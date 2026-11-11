using WaterLily: wall_model!, log_law, interp_per, interp, inside_u

# Plane channel of height `H` periodic in x, with the walls shifted `s` cells relative to the grid.
# Also returns the wall distance of the rows of `u₁`.
# The walls slide along themselves at the speed `Uw`.
function wm_channel(mem;N=(8,48),H=32,s=0,ν=0.01,g=nothing,u0=(1,0),Uw=0,T=Float64)
    yc,H2,Uw = T(N[2]/2+s),T(H/2),T(Uw)
    body = AutoBody((x,t)->H2-abs(x[2]-yc),(x,t)->x-SA[Uw*t,0])
    sim = Simulation(N,(0,0),H;U=1,ν,body,u0,g,perdir=(1,),T,mem)
    sim,[H2-abs(j-T(1.5)-yc) for j in 1:N[2]+2]
end
# Channel of height `H` at 45° to the grid, periodic in both directions
function wm_diag_channel(mem;N=64,H=24,s=0,ν=0.01,λ=quick,T=Float64)
    P,r,H2,s = T(N/√2),T(√0.5),T(H/2),T(s)
    body = AutoBody((x,t)->H2-abs(mod((x[1]-x[2])*r-s+P/2,P)-P/2))
    Simulation((N,N),(0,0),H;U=1,ν,body,u0=(r,r),perdir=(1,2),λ,T,mem)
end
# Sheared flow U0+G*d along the walls of a channel with the normal (a,b)/√(a²+b²), periodic in both
# directions. Also returns the flow direction and the total wall length.
function wm_shear_channel(a,b,s;N=64,H=16,U0=0.5,G=0.05,λ=quick)
    m = √(a^2+b^2); n̂,t̂ = SA[a,b]/m,SA[-b,a]/m; P,H2 = N/m,H/2
    dist(x) = H2-abs(mod(x[1]*n̂[1]+x[2]*n̂[2]-s+P/2,P)-P/2)
    u0(i,x) = (U0+G*max(dist(x),0))*t̂[i]
    Simulation((N,N),(0,0),H;U=1,ν=1e-8,body=AutoBody((x,t)->dist(x)),u0,perdir=(1,2),T=Float64,λ),t̂,2N*m
end
wm_profile(sim) = (u = Array(sim.flow.u); vec(sum(u[2:end-1,:,1],dims=1))/(size(u,1)-2))
wm_step!(sim,n;kw...) = for _ in 1:n
    sim_step!(sim;remeasure=false,udf=wall_model!,body=sim.body,ϵ=sim.ϵ,kw...)
end
no_stress(U,y,ν) = zero(U)
const wm_τ = 1e-3
fixed_stress(U,y,ν) = oftype(U,√wm_τ)

@testset "WallModel.jl" begin
    @testset "log law" begin
        κ,B = 0.41,5.2
        for (U,y,ν) in ((1.,3.,1e-4),(1.,3.,1e-6),(0.3,2.,1e-5))
            uτ = log_law(U,y,ν)
            @test U/uτ ≈ log(y*uτ/ν)/κ+B
        end
        @test log_law(1.,3.,1.) ≈ √(1/3)                  # viscous sublayer, u⁺=y⁺
        y⁺ = 11.062; uτ = 0.1; y = 3.; ν = y*uτ/y⁺         # continuous across the two branches
        @test log_law(0.999y⁺*uτ,y,ν) ≈ log_law(1.001y⁺*uτ,y,ν) rtol=2e-3
        @test log_law(1f0,3f0,1f-5) isa Float32
        @test log_law(1.,3.,1e-5;κ=0.38,B=4.1) > log_law(1.,3.,1e-5;κ=0.38,B=5.2)
    end
    @testset "periodic sampling" begin
        u = rand(10,12,2); BC!(u,(0.,0.),false,(1,2))
        x = SA[3.3,4.7]
        @test interp_per(x,u,(1,2)) ≈ interp(x,u)
        @test interp_per(x,u,()) ≈ interp(x,u)
        @test interp_per(SA[-0.7,10.2],u,(1,2)) ≈ interp(SA[7.3,0.2],u)
        @test interp_per(SA[-0.7,10.2],u,()) ≈ interp(SA[-0.7,10.2],u) # clamped
    end
    for mem in arrays
        @testset "free slip $mem" begin
            # a viscous plug flow is untouched by a stress-free wall, also inside the kernel
            for s in (0,0.25,0.5)
                sim,d = wm_channel(mem;s,ν=0.1)
                wm_step!(sim,50;wall_law=no_stress)
                @test all(u->u≈1,wm_profile(sim)[d.>-1])
                @test maximum(abs,Array(sim.flow.u)[:,:,2]) < 1e-12
            end
            sim = wm_diag_channel(mem;s=0.3,ν=0.1)
            wm_step!(sim,50;wall_law=no_stress)
            @test all(u->u≈√0.5,Array(sim.flow.u)[Array(sim.flow.μ₀).>0])
            @test_throws AssertionError sim_step!(Simulation((8,8),(1,0),8;body=sim.body);udf=wall_model!,body=sim.body)
        end
        @testset "3D Float32, remeasured $mem" begin
            H,yc = 20f0,16.3f0
            body = AutoBody((x,t)->H/2-abs(x[2]-yc))
            sim = Simulation((8,32,8),(0,0,0),H;U=1,ν=0.1f0,body,u0=(1,0,0),perdir=(1,3),T=Float32,mem)
            for _ in 1:10
                sim_step!(sim;udf=wall_model!,body,ϵ=sim.ϵ,wall_law=no_stress)
            end
            u = Array(sim.flow.u)
            @test eltype(u)==Float32 && all(≈(1),u[:,:,:,1][Array(sim.flow.μ₀)[:,:,:,1].>0])
            @test maximum(abs,u[:,:,:,2:3]) < 1f-6
        end
        @testset "wall stress $mem" begin
            rowμ₀(sim) = Array(sim.flow.μ₀)[3,:,1]
            # A stress-free forced plug: after one step each fluid-side row has gained μ₀gΔt,
            # ie. the kernel rows are not relaxed, only slowed down
            g₀ = 1e-3
            for s in (0,0.25)
                sim,d = wm_channel(mem;s,ν=1e-8,g=(i,x,t)->i==1 ? g₀ : 0.)
                wm_step!(sim,1;wall_law=no_stress)
                @test all(d.≤0 .|| wm_profile(sim).-1 .≈ rowμ₀(sim)*g₀*sim.flow.Δt[1])
            end
            # Without viscosity the wall-parallel momentum Σu/μ₀ of the fluid-side rows decays by 2τ
            # per unit length: exactly when no row is cut by the kernel (s=0.5). Otherwise Σu/μ₀
            # over-reads by up to 4.5%, so the 5% is a regression bound, not an accuracy claim
            for (s,rtol) in ((0.5,1e-6),(0,0.05),(0.25,0.05))
                sim,d = wm_channel(mem;s,ν=1e-8); fluid = d.>0
                M() = sum(wm_profile(sim)[fluid]./rowμ₀(sim)[fluid])
                wm_step!(sim,10;wall_law=fixed_stress)
                M₀,t₀ = M(),WaterLily.time(sim)
                wm_step!(sim,40;wall_law=fixed_stress)
                @test (M₀-M())/(WaterLily.time(sim)-t₀) ≈ 2wm_τ rtol=rtol
            end
            # On a 45° wall with central convection the plain sum Σu of the fluid-side faces decays
            # at 1.02 to 1.05 times that rate. Regression bound: it is 1.33 to 1.43 when the
            # wall-normal velocity is estimated differently on the faces of a cell
            sim = wm_diag_channel(mem;s=0.25,ν=1e-8,λ=cds)
            R = inside_u(size(sim.flow.p)); side = [sdf(sim.body,loc(Ii,Float64),0.)>0 for Ii in R]
            Σu() = sum(Array(sim.flow.u)[R][side])*√0.5
            wm_step!(sim,10;wall_law=fixed_stress)
            Σu₀,t₀ = Σu(),WaterLily.time(sim)
            wm_step!(sim,40;wall_law=fixed_stress)
            @test (Σu₀-Σu())/(WaterLily.time(sim)-t₀)/(2wm_τ*size(R,1)*√2) ≈ 1.04 atol=0.04
        end
    end
    @testset "sheared flow, stress-free walls" begin
        # Steady without viscosity. Spurious stress as the momentum gain of the fluid-side faces per
        # unit wall length, over U0*G: none on aligned walls. On oblique walls the zero-gradient
        # extension of the slip velocity leaves a kink that convection turns into a force, largest
        # near slopes of 1:2 and 1:3, where it is 0.007 to 0.035 (regression bound, not a target)
        U0,G = 0.5,0.05
        for (a,b,bound) in ((0,1,1e-6),(1,-2,0.05)), λ in (quick,cds), s in (0,0.6)
            sim,t̂,Lw = wm_shear_channel(a,b,s;U0,G,λ)
            R = inside_u(size(sim.flow.p)); side = [sdf(sim.body,loc(Ii,Float64),0.)>0 for Ii in R]
            M() = sum((sim.flow.u[R].*[t̂[last(Ii)] for Ii in R])[side])
            wm_step!(sim,10;wall_law=no_stress); M₀,t₀ = M(),WaterLily.time(sim)
            wm_step!(sim,40;wall_law=no_stress)
            @test abs(M()-M₀)/(WaterLily.time(sim)-t₀)/Lw/(U0*G) < bound
        end
    end
    @testset "sliding walls" begin
        # Galilean invariance: the model acts on the velocity relative to the wall
        Uw = 0.5
        for (law,ν) in ((no_stress,0.1),(fixed_stress,1e-8),(log_law,1e-3)), s in (0,0.25)
            a,d = wm_channel(Array;s,ν); b,_ = wm_channel(Array;s,ν,Uw,u0=(1+Uw,0))
            for sim in (a,b), _ in 1:30 # same time step for both
                sim.flow.Δt[end] = 0.2; wm_step!(sim,1;wall_law=law)
            end
            @test wm_profile(b)[d.>-1].-Uw ≈ wm_profile(a)[d.>-1]
            law==no_stress && @test all(≈(1+Uw),wm_profile(b)[d.>-1])
        end
    end
    @testset "driven laminar channel" begin
        # The viscous branch of the wall law closes the loop: the wall stress balances the forcing.
        # The law assumes a linear profile up to h, which overshoots Poiseuille flow by 4h²/H²
        H = 32; ν = 1.; g₀ = 8ν/H^2; h = 3.
        for s in (0,0.25)
            sim,d = wm_channel(Array;H,s,ν,u0=(0,0),g=(i,x,t)->i==1 ? g₀ : 0.)
            wm_step!(sim,round(Int,40H^2/π^2/ν))
            u = wm_profile(sim)
            j = findfirst(>(h),d); a = h-d[j-1]
            τ = log_law((1-a)*u[j-1]+a*u[j],h,ν)^2
            @test 2τ/(g₀*H) ≈ 1 rtol=0.015
            @test maximum(u) ≈ 1+4h^2/H^2 rtol=0.005
        end
    end
end
