using WaterLily: inside_u

# Regression test for the udf advecting-velocity fix (branch udf-advecting-velocity):
# `mom_step!` now passes the velocity field the convective flux is evaluated on to the udf:
# `a.u⁰` in the predictor and the projected `a.u` in the corrector.
# The general udf machinery (body force, rotating frame) is already covered
# in test_flow.jl ("increasing body force", "rotating reference frame"); here we only check the
# new 3-arg advecting-velocity signature and that the 2-arg force-only fallback still dispatches.
@testset "udf advecting velocity" begin
    # 3-arg udf supplies the velocity the convective flux uses each phase
    saw = Tuple{Float64,Float64}[]
    rec!(flow, u, t; kw...) = (push!(saw, (maximum(abs, @view u[inside_u(u),:]),
                                           maximum(abs, @view flow.u[inside_u(flow.u),:]))); nothing)
    sim = Simulation((16,16),(1.0,0.0),16; U=1.0, T=Float64, mem=Array)
    empty!(saw); sim_step!(sim; udf=rec!)
    @test saw[1][1] > 1e-8       # predictor udf sees nonzero u⁰ (the fix)
    @test saw[1][2] < 1e-8       # while flow.u interior is zeroed (the old bug source)
    @test saw[end][1] > 1e-8     # corrector udf sees the nonzero projected field

    # 2-arg force-only udf still dispatches and runs once per phase (predictor + corrector)
    NG = Ref(0)
    grav!(flow, t; g=0.5) = (WaterLily.@loop flow.f[Ii] += g over Ii in CartesianIndices(flow.f); NG[]+=1; nothing)
    sim2 = Simulation((16,16),(1.0,0.0),16; U=1.0, T=Float64, mem=Array)
    sim_step!(sim2; udf=grav!, g=0.5)
    @test NG[] == 2 && all(isfinite, sim2.flow.u)
end

using WaterLily: sgs!, inside, δ
@inline function smagorinsky(I::CartesianIndex{D};S,Cs,Δ) where D
    s = zero(eltype(S))
    for i in 1:D, j in 1:D
        s += S[I,i,j]^2
    end
    (Cs*Δ)^2*√(2s)
end
@inline uniform_νₜ(I;ν₀) = ν₀
# Vreman (2004), doi:10.1063/1.1785131, from the velocity gradient ∇u[I,i,j] = ∂uᵢ/∂xⱼ
@inline function vreman(I::CartesianIndex{D};∇u,C,Δ) where D
    T = eltype(∇u)
    α = SMatrix{D,D,T}(ntuple(k->∇u[I,(k-1)÷D+1,(k-1)%D+1],Val(D*D)))  # α[j,i] = ∂uᵢ/∂xⱼ
    β = Δ^2*α'*α; αα = sum(abs2,α); Bᵦ = zero(T)
    for i in 1:D, j in i+1:D
        Bᵦ += β[i,i]*β[j,j]-β[i,j]^2
    end
    αα>eps(T) ? C*√(max(Bᵦ,zero(T))/αα) : zero(T)
end
sgs_u2(i,x,k) = i==1 ? sin(k*x[1])*cos(2k*x[2])+0.3cos(k*x[2]) : -cos(k*x[1])*sin(2k*x[2])/2+0.2sin(2k*x[1])
sgs_u3(i,x,k) = i==1 ? sin(k*x[1])*cos(k*x[2])*cos(2k*x[3]) : i==2 ? -cos(k*x[1])*sin(k*x[2])*cos(2k*x[3])+0.3sin(k*x[3]) : 0.2sin(2k*x[1])*cos(k*x[2])
# Periodic box with a smooth velocity field that is not solenoidal
function sgs_box(N,D,mem)
    k = 2π/N; u0 = D==2 ? (i,x)->sgs_u2(i,x,k) : (i,x)->sgs_u3(i,x,k)
    Simulation(ntuple(_->N,D),ntuple(_->0,D),N;U=1,ν=0.01,u0,perdir=ntuple(identity,D),T=Float64,mem)
end
sgs_buffer(sim) = (D = ndims(sim.flow.p); WaterLily.zeros_on(WaterLily.get_backend(sim.flow.p),Float64,(size(sim.flow.p)...,D,D)))
function sgs_force(sim;kw...)
    fill!(sim.flow.f,0); sgs!(sim.flow,sim.flow.u,0;kw...)
    Array(sim.flow.f)[inside_u(size(sim.flow.p))]
end

@testset "sgs!" begin
    for mem in arrays, D in (2,3)
        @testset "periodic box $(D)D $mem" begin
            N = D==2 ? 32 : 16; R = inside_u((N+2,N+2,N+2)[1:D])
            sim = sgs_box(N,D,mem); S = sgs_buffer(sim); u = Array(sim.flow.u)
            f = sgs_force(sim;νₜ=smagorinsky,S,Cs=0.2,Δ=1)
            # the force is a flux divergence, so it conserves momentum, and it dissipates energy
            @test all(abs(sum(selectdim(f,D+1,i))) < 1e-12 for i in 1:D)
            @test sum(u[R].*f) < 0
            # the same field shifted across the periodic boundaries gives the same force, shifted
            sh = (ntuple(_->N÷2-3,D)...,0)
            sim₂ = sgs_box(N,D,mem); u₂ = copy(u); u₂[R] .= circshift(u[R],sh)
            copyto!(sim₂.flow.u,u₂); BC!(sim₂.flow.u,ntuple(zero,D),false,sim₂.flow.perdir)
            @test maximum(abs,sgs_force(sim₂;νₜ=smagorinsky,S,Cs=0.2,Δ=1).-circshift(f,sh)) < 1e-13
            # a uniform νₜ gives ν₀(∇²u+∇(∇·u)), both terms of the strain
            f = sgs_force(sim;νₜ=uniform_νₜ,ν₀=0.5); ref = zero(f)
            dv(I) = sum(u[I+δ(j,I),j]-u[I,j] for j in 1:D)
            for (k,Ii) in enumerate(R)
                I,i = CartesianIndex(Base.front(Ii.I)),last(Ii.I)
                ref[k] = (sum(u[I+δ(j,I),i]+u[I-δ(j,I),i]-2u[Ii] for j in 1:D)+dv(I)-dv(I-δ(i,I)))/2
            end
            @test maximum(abs,f.-ref) < 1e-13
            # a model built on the velocity gradient
            fᵥ = sgs_force(sim;νₜ=vreman,∇u=sgs_buffer(sim),C=0.07,Δ=1)
            @test all(≥(0),Array(sim.flow.σ)[inside(sim.flow.σ)]) && sum(u[R].*fᵥ) < 0
        end
    end
    @testset "pure shear" begin
        # Uniform shear between two slip boundaries: Smagorinsky gives (CₛΔ)²G and no force away from
        # the boundaries, and Vreman gives no eddy viscosity
        G = 0.1
        sim = Simulation((16,16,8),(0,0,0),16;U=1,ν=0.01,u0=(i,x)->i==1 ? 0.1x[2] : 0.,perdir=(1,3),T=Float64)
        ∇u,S = sgs_buffer(sim),sgs_buffer(sim); R = CartesianIndices((2:17,4:15,2:9))
        f = sgs_force(sim;νₜ=smagorinsky,S,Cs=0.2,Δ=1)
        @test all(≈(0.2^2*G),sim.flow.σ[R]) && maximum(abs,f[:,3:end-2,:,:]) < 1e-14
        sgs_force(sim;νₜ=vreman,∇u,C=0.07,Δ=1)
        @test all(≈(G),∇u[R,1,2]) && maximum(abs,∇u[R,2,1]) < 1e-14 && maximum(abs,sim.flow.σ[R]) < 1e-12
    end
end
