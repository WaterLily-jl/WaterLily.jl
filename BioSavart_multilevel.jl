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

biotsavart(r,j) = Float32((3-2j)*r[j]/(2π*r'*r))
r(x,I,dx) = x-dx*WaterLily.loc(0,I) .+ 1.5f0*(dx-1)
# inR(x,dx,R) = CartesianIndex(clamp.(Tuple.((round.(Int,x .+ 1.5f0*(dx-1)),first(R),last(R)))...))
function inR(x,dx,R) 
    y = x .+ 1.5f0*(dx-1)
    z = round.(Int,y/dx)
    CartesianIndex(clamp.(Tuple.((z,first(R),last(R)))...))
end
function u_ω(i,x,ml,dis=2.5f0)
    # initialize at coarsest level
    ui = zero(eltype(x)); j = i%2+1

    # for I ∈ inside(ml[1]) # single level method
    #     ui += ml[1][I]*biotsavart(r(x,I,1),j)
    # end
    # return ui
    
    l = lastindex(ml)
    ω = ml[l]
    R = WaterLily.inside(ω)
    dx = 2^(l-1)

    # loop levels
    while l>1
        # find Region close to x
        Rclose = inR(x .-dx*dis,dx,R):inR(x .+dx*dis,dx,R)
        # Rclose2 = filter(I->sum(abs2,r(x,I,dx))<dis*(dx^2),R)
        # @assert mapreduce(I->I∈Rclose,&,Rclose2)

        # get contributions outside Rclose
        for I ∈ R
            !(I ∈ Rclose) && (ui += ω[I]*biotsavart(r(x,I,dx),j))
        end

        # move "up" one level within Rclose
        l -= 1
        R = first(WaterLily.up(first(Rclose))):last(WaterLily.up(last(Rclose)))
        ω = ml[l]
        dx = 2^(l-1)
    end

    # add Imax contribution
    for I ∈ R
        ui += ω[I]*biotsavart(r(x,I,dx),j)
    end
    return ui
end

function biotBC!(u,U,ml)
    N,n = WaterLily.size_u(u)
    for j ∈ 1:n, i ∈ 1:n
        for s ∈ (1,2,N[j])
            for I ∈ WaterLily.slice(N,s,j)
    # i,I = 1,CartesianIndex(5,1)
                x = WaterLily.loc(i,I)
                # @show i,j,I,x
                u[I,i] = u_ω(i,x,ml)+U[i]
            end
        end
    end
end

# Check reconstruction on lamb dipole
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

begin
    sim = lamb_dipole(128);σ = sim.flow.σ;
    ml = MLArray(σ); ml_ω!(ml,sim.flow);
    u = sim.flow.u;
    @time WaterLily.BC!(u,sim.flow.U);
    @time biotBC!(u,sim.flow.U,ml); # lots of allocations!
    @assert sum(abs2,u-sim.flow.u⁰)/sim.L<2e-4
end