#mpiexecjl --project=examples/ -n 4 julia PoissonMPI.jl
using MPI,WaterLily
using StaticArrays
using FileIO,JLD2
include("WaterLilyMPI.jl")

# domain and fields
L = 128
r = init_mpi((L,L))
N,D,T = (L,L),2,Float64
Ng = N .+ 4
x,z = zeros(T, Ng) |> Array, zeros(T, Ng) |> Array
μ⁰ = ones(T, (Ng..., D)) |> Array
# apply zero Neumann BC
BC!(μ⁰,zeros(2))

# test poisson solver in parallel
for Pois in [:Poisson,:MultiLevelPoisson]

    # create Poisson solver
    pois = eval(Pois)(x,μ⁰,z)
    MG = isa(pois,MultiLevelPoisson)

    # source term with solution
    # u := cos(2πx/L) cos(2πy/L)
    # ∂u²/∂x² + ∂u²/∂y² = f = -8π²/L² cos(2πx/L) cos(2πy/L)
    apply!(x->-π^2/L^2*cos(π*x[1]/L)*cos(π*x[2]/L),pois.z)
    save("test_$(string(Pois))_$(me())_source.jld2","C",pois.z)

    # solver
    MG ? solver!(pois;tol=10eps(T),itmx=32) : solver!(pois;tol=10eps(T),itmx=1e4)
    @show pois.n
    me()==0 && println("Iters $(pois.n)")
    save("test_$(string(Pois))_$(me()).jld2","C",pois.x)
    uₑ = copy(pois.x); apply!(x->cos(π*x[1]/L)*cos(π*x[2]/L),uₑ)
    save("test_$(string(Pois))_$(me())_sol.jld2","C",uₑ)
    L2 = WaterLily.L₂(uₑ .- pois.x); Linf = WaterLily.L∞(uₑ .- pois.x)
    me()==0 && println("L₂-norm of error $L2")
    me()==0 && println("L∞-norm of error $Linf")
end
finalize_mpi()