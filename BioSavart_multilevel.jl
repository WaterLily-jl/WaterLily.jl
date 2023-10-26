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

function ml_ω!(ml,u)
    # cell-centered vorticity on finest grid
    @inside ml[1][I] = ω(I,u)
    # pool values at each level
    for l ∈ 2:lastindex(ml)
        WaterLily.restrict!(ml[l],ml[l-1])
    end
end
ω(I::CartesianIndex{2},u) = WaterLily.permute((j,k)->WaterLily.∂(k,j,I,u),3)

function u_ω(i,x,ml,dis=2.5f0)
    # initialize at bottom level
    ui = zero(eltype(x)); j = i%2+1
    l = lastindex(ml)
    ω = ml[l]
    R = WaterLily.inside(ω)
    dx = 2^(l-1)

    # loop levels
    while l>1
        # find Region close to x
        Rclose = inR(x .-dx*dis,dx,R):inR(x .+dx*dis,dx,R)

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

    # top level contribution
    for I ∈ R
        ui += ω[I]*biotsavart(r(x,I,dx),j)
    end
    return ui
end
biotsavart(r,j) = Float32((3-2j)*r[j]/(2π*r'*r))
r(x,I,dx) = x-dx*WaterLily.loc(0,I) .+ 1.5f0*(dx-1)
inR(x,dx,R) = clamp(CartesianIndex(round.(Int,x/dx .+ 1.5f0*(1-1/dx))...),R)
Base.clamp(I::CartesianIndex,R::CartesianIndices) = CartesianIndex(clamp.(I.I,first(R).I,last(R).I))

function biotBC!(u,U,ml)
    ml_ω!(ml,u)
    N,n = WaterLily.size_u(u)
    for j ∈ 1:n, i ∈ 1:n
        for s ∈ ifelse(i==j,(1,2,N[j]),(1,N[j]))
            for I ∈ WaterLily.slice(N,s,j)
                x = WaterLily.loc(i,I)
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
    sim = lamb_dipole(128); σ = sim.flow.σ; u = sim.flow.u;
    ml = MLArray(σ);
    @time biotBC!(u,sim.flow.U,ml);
    @assert sum(abs2,u-sim.flow.u⁰)/sim.L<2e-4
    @time BC!(u,sim.flow.U); #only x10 faster
end

# Check pressure solver convergence on circle
circ(N;D=3N/4,U=1) = Simulation((N, N), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x .- (N/2+1.5))-D/2))
begin 
    sim = circ(128); a = sim.flow; a.u .= 0; ml = MLArray(a.σ);
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν);
    WaterLily.BDIM!(a);
    for i in 1:30
        biotBC!(a.u,a.U,ml);
        @inside a.σ[I] = WaterLily.div(I,a.u)
        @show L₂(a.σ),maximum(a.u)
        L₂(a.σ)<1e-3 && (@show i; break)
        WaterLily.project!(a,sim.pois;itmx=1);
    end
    @assert abs(maximum(a.u)/2-1)<1e-2
end