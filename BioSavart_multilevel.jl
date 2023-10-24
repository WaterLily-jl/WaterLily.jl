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
    # vorticity on finest grid
    @inside ml[1][I] = WaterLily.curl(3,I,a.u)
    # pool values at each level
    for l ∈ 2:lastindex(ml)
        WaterLily.restrict!(ml[l],ml[l-1])
    end
end

@inline @fastmath function biotsavart(x,j,ω,I,dx)
    r = x-dx*(WaterLily.loc(I)-0.5); s = 3-2j
    s*ω[I]*r[j]/(2π*r'*r)
end

function u_ω(i,x,ml)
    # initialize at coarsest level
    ui = zero(eltype(x)); j = i%2+1
    l = lastindex(ml.levels)
    R = inside(ml[l])
    Imax,dx,ω = 0,0,0

    # loop levels
    while l>=1
        # set grid scale and index nearest to x
        ω = ml[l]
        dx = 2^(l-1)
        Imax = CI(round.(Int,x/dx .+0.5)...)

        # get contributions other than Imax
        for I in R
            I != Imax && (ui += biotsavart(x,j,ω,I,dx))
        end

        # move "up" one level near Imax
        l -= 1
        R = up(Imax)
    end

    # add Imax contribution
    return ui + biotsavart(x,j,ω,Imax,dx)
end

using SpecialFunctions,ForwardDiff
function lamb_dipole(N;D=N/3,U=1)
    β = 2.4394π/D
    C = -2U/(β*besselj0(β*D/2))
    function ψ(x,y)
        r = √(x^2+y^2)
        ifelse(r ≥ D/2, U*((D/2r)^2-1)*y, C*besselj1(β*r)*y/r)
    end
    center = SA[N/2,N/2] .+ 1
    function uλ(i,xy)
        x,y = xy-center
        ifelse(i==1,ForwardDiff.derivative(y->ψ(x,y),y)+1+U,-ForwardDiff.derivative(x->ψ(x,y),x))
    end
    Simulation((N, N), (1,0), D; uλ)
end

include("examples/TwoD_plots.jl")
sim = lamb_dipole(64,D=48);
σ = sim.flow.σ;
@inside σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
flood(σ[inside(σ)],clims=(-20,20))