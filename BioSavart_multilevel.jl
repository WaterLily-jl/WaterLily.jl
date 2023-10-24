using WaterLily,StaticArrays

# 2D Multi-level Biot-Savart functions 
function MLArray(x)
    levels = [x]
    N = size(x)
    while all(N .|> WaterLily.divisible)
        N = @. 1+N÷2
        y = similar(x,N); fill!(y,0)
        push!(levels,y)
    end
    return levels
end

function ml_ω!(ml,a::Flow)
    # cell-centered vorticity on finest grid
    @inside ml[1][I] = ω(I,a.u)
    # pool values at each level
    for l ∈ 2:lastindex(ml)
        WaterLily.restrict!(ml[l],ml[l-1])
    end
end
ω(I::CartesianIndex{2},u) = WaterLily.permute((j,k)->WaterLily.∂(k,j,I,u),3)

biotsavart(r,j) = (3-2j)*r[j]/(2π*r'*r)
r(x,I,dx) = x-dx*WaterLily.loc(0,I) .+ 1.5*(dx-1)

function u_ω(i,x,ml)
    # initialize at coarsest level
    ui = zero(eltype(x)); j = i%2+1
    l = lastindex(ml)
    R = inside(ml[l])
    Imax,dx,ω = 0,0,0

    # loop levels
    while l>=1
        # set grid scale and index nearest to x
        ω = ml[l]
        dx = 2^(l-1)
        Imax = argmin(I->sum(abs2,r(x,I,dx)),R)

        # get contributions other than Imax
        for I in R
            I != Imax && (ui += ω[I]*biotsavart(r(x,I,dx),j))
        end

        # move "up" one level near Imax
        l -= 1
        R = WaterLily.up(Imax)
    end

    # add Imax contribution
    return ui + ω[Imax]*biotsavart(r(x,Imax,dx),j)
end

function biotBC!(u,U,ml)
    N,n = WaterLily.size_u(u)
    for j ∈ 1:n, i ∈ 1:n
        for s ∈ (1,2,N[j])
            for I ∈ WaterLily.slice(N,s,j)
                x = WaterLily.loc(i,I)
                u[I,i] = u_ω(i,x,ml)+U[i]
            end
        end
    end
end

using SpecialFunctions,ForwardDiff
function lamb_dipole(N;D=3N/4,U=1)
    β = 2.4394π/D
    C = -2U/(β*besselj0(β*D/2))
    function ψ(x,y)
        r = √(x^2+y^2)
        ifelse(r ≥ D/2, U*((D/2r)^2-1)*y, C*besselj1(β*r)*y/r)
    end
    center = SA[N/2,N/2] .+ 1.5
    function uλ(i,xy)
        x,y = xy-center
        ifelse(i==1,ForwardDiff.derivative(y->ψ(x,y),y)+1+U,-ForwardDiff.derivative(x->ψ(x,y),x))
    end
    Simulation((N, N), (1,0), D; uλ) # Don't overwrite ghosts with BCs
end

include("examples/TwoD_plots.jl")
sim = lamb_dipole(16);σ = sim.flow.σ;
@inside σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
flood(σ[2:end,2:end],clims=(-20,20))
flood(sim.flow.u[2:end,:,1])
flood(sim.flow.u[:,2:end,2])

ml = MLArray(σ); ml_ω!(ml,sim.flow);
flood(ml[1],clims=(-5,5))
flood(ml[2],clims=(-5,5))
flood(ml[3],clims=(-5,5))
flood(ml[4],clims=(-5,5))

u = sim.flow.u;
WaterLily.BC!(u,sim.flow.U);
sum(abs2,u-sim.flow.u⁰) # Around N[1]/2!
biotBC!(u,sim.flow.U,ml);
sum(abs2,u-sim.flow.u⁰) # Like 1/N[1]^3?!?
flood(u[2:end,:,1])
flood(u[:,2:end,2])
