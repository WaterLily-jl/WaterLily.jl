using WaterLily
using HYPRE
using Test
using StaticArrays

function make_poisson(N::NTuple{D};psolver=:MultiLevelPoisson,T=Float32,mem=Array) where D
    # generate field
    x,z = zeros(T,N.+2) |> mem, zeros(T,N.+2) |> mem
    μ⁰ = ones(T,((N.+2)...,D)) |> mem
    
    # apply zero Neumann BC
    WaterLily.BC!(μ⁰,zeros(D))
   
    # create Poisson solver
    eval(psolver)(x,μ⁰,z)
end
myL₂(a,b;R=inside(a)) = (a.-=b; √WaterLily.L₂(a)/length(a[R]))
L∞(a,b;R=inside(a)) = maximum(abs.(a[R].-b[R]))

# test poisson solver in parallel
@testset "Manufactured solution 1" begin
    for psolver in [:Poisson,:MultiLevelPoisson,:HyprePoisson]
        for f ∈ arrays, T in [Float32,Float64], L in [32,64]
            #@info "Testing $psolver solver with T=$T on a domain L=$L"
            psolver==:HyprePoisson && T==Float32 && continue # skip this test
            # create Poisson solver
            pois = make_poisson((L,L);psolver,T,mem=f)
            MG = isa(pois,MultiLevelPoisson)

            # source term with solution
            # u := cos(2πx/L) cos(2πy/L)
            uₑ = copy(pois.x); apply!(x->cos(2π*x[1]/L)*cos(2π*x[2]/L),uₑ)
            # ∂u²/∂x² + ∂u²/∂y² = f = -8π²/L² cos(2πx/L) cos(2πy/L)
            apply!(x->-8π^2/L^2*cos(2π*x[1]/L)*cos(2π*x[2]/L),pois.z)

            # solver
            MG ? solver!(pois;tol=100eps(T),itmx=32) : solver!(pois;tol=100eps(T),itmx=1e3)

            # show stats and save
            @test WaterLily.L₂(pois) ≤ 100eps(T) # have we converged?
            #@info "Iters $(pois.n), r⋅r=$(WaterLily.L₂(pois))"
            L2 = myL₂(pois.x,uₑ); Linf = L∞(pois.x,uₑ)
            #@info "L₂-norm of error $L2, L∞-norm of error $Linf"
        end
    end
end
hyrostatic!(p) = @inside p.z[I] = WaterLily.∂(1,I,p.L) # zero v contribution everywhere
# test poisson solver in parallel
@testset "hydrostatic pressure" begin
    for psolver in [:Poisson,:MultiLevelPoisson,:HyprePoisson]
        for f ∈ arrays, T in [Float32,Float64], L in [32,64]    #@info "Testing $psolver solver with T=$T on a domain L=$L"
            psolver==:HyprePoisson && T==Float32 && continue # skip this test
            # create Poisson solver
            pois = make_poisson((L,L);psolver,T,mem=f)
            MG = isa(pois,MultiLevelPoisson)

            # make circle and solver
            hyrostatic!(pois); WaterLily.update!(pois)
            MG ? solver!(pois;tol=100eps(T),itmx=32) : solver!(pois;tol=100eps(T),itmx=1e3)

            # show stats and save
            @test WaterLily.L₂(pois) ≤ 100eps(T) # have we converged?
            #@info "Iters $(pois.n), r⋅r=$(WaterLily.L₂(pois))"
            uₑ = copy(pois.x); apply!(x->x[1]-L/2,uₑ)
            L2 = myL₂(pois.x,uₑ); Linf = L∞(pois.x,uₑ)
            #@info "L₂-norm of error $L2, L∞-norm of error $Linf"
        end
    end
end
# potential flow around a circle in the middle of a domain
function circle!(p;L=last(size(p.x)),T=eltype(p.x))
    f(i,x) = WaterLily.μ₀(√sum(abs2,x.-L/2)-L/8,one(T)) # sdf circle
    apply!(f,p.L); BC!(p.L,[one(T),zero(T)]) # BC for the velocity field
    @inside p.z[I] = WaterLily.∂(1,I,p.L) # zero v contribution everywhere
    BC!(p.L,zeros(last(size(p.L)))) # correct BC on μ₀
end
@testset "Potential flow" begin
    for psolver in [:Poisson,:MultiLevelPoisson]
        for f ∈ arrays, T in [Float32,Float64], L in [32,64]    #@info "Testing $psolver solver with T=$T on a domain L=$L"
            # create Poisson solver
            pois = make_poisson((L,L);psolver,T,mem=f)
            MG = isa(pois,MultiLevelPoisson)
            
            # make circle and solve
            circle!(pois); WaterLily.update!(pois) # update the Poisson solver
            MG ? solver!(pois;tol=100eps(T),itmx=32) : solver!(pois;tol=100eps(T),itmx=1e3)
            #@info "Iters $(pois.n), r⋅r=$(WaterLily.L₂(pois))"
            @test WaterLily.L₂(pois) ≤ 100eps(T) # have we converged?
        end
    end
end
function capacitor!(p;L=last(size(p.x)))
    # 2 parallel plate capacitors with potential ±1
    apply!(x->ifelse(abs(x[1]-L/2)≤5 && abs(x[2]-L/2)≈5.5,sign(x[2]-L/2),0),p.z)
end
@testset "parallel capacitor" begin
    for psolver in [:Poisson,:MultiLevelPoisson,:HyprePoisson]
        for f ∈ arrays, T in [Float32,Float64], L in [32,64]    #@info "Testing $psolver solver with T=$T on a domain L=$L"
            psolver==:HyprePoisson && T==Float32 && continue # skip this test
            # create Poisson solver
            pois = make_poisson((L,L);psolver,T,mem=f)
            MG = isa(pois,MultiLevelPoisson)
            
            # make circle and solve
            capacitor!(pois); WaterLily.update!(pois)
            MG ? solver!(pois;tol=100eps(T),itmx=32) : solver!(pois;tol=100eps(T),itmx=1e3)
            #@info "Iters $(pois.n), r⋅r=$(WaterLily.L₂(pois))"
            @test WaterLily.L₂(pois) ≤ 100eps(T) # have we converged?
        end
    end
end
function pn_junction!(p;L=last(size(p.x)))
    # charge densiy f(x,y) = ifelse(|x-L/2|<10 && |y-L/2|<5, sech(x)tanh(x), 0)
    apply!(x->ifelse(abs(x[1]-L/2)≤10 && abs(x[2]-L/2)≤5,sech(x[1]-L/2)*tanh(x[1]-L/2),0),p.z)
end
@testset "p-n junction" begin
    for psolver in [:Poisson,:MultiLevelPoisson,:HyprePoisson]
        for f ∈ arrays, T in [Float32,Float64], L in [32,64]    #@info "Testing $psolver solver with T=$T on a domain L=$L"
            psolver==:HyprePoisson && T==Float32 && continue # skip this test
            # create Poisson solver
            pois = make_poisson((L,L);psolver,T,mem=f)
            MG = isa(pois,MultiLevelPoisson)
            
            # make circle and solve
            pn_junction!(pois); WaterLily.update!(pois)
            MG ? solver!(pois;tol=100eps(T),itmx=32) : solver!(pois;tol=100eps(T),itmx=1e3)
            #@info "Iters $(pois.n), r⋅r=$(WaterLily.L₂(pois))"
            @test WaterLily.L₂(pois) ≤ 100eps(T) # have we converged?
        end
    end
end