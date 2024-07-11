#mpiexecjl --project=examples/ -n 4 julia PoissonMPI.jl
using MPI,WaterLily
using StaticArrays
using FileIO,JLD2
include("WaterLilyMPI.jl")

WaterLily.L₂(ml::MultiLevelPoisson) = WaterLily.L₂(ml.levels[1])
WaterLily.L∞(ml::MultiLevelPoisson) = WaterLily.L∞(ml.levels[1])

# domain and fields
L = 16
r = init_mpi((L,L))
N,D,T = (L,L),2,Float64
Ng = N .+ 4 # double ghosts
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
    uₑ = copy(pois.x); apply!(x->cos(2π*x[1]/L)*cos(2π*x[2]/L),uₑ)
    # ∂u²/∂x² + ∂u²/∂y² = f = -8π²/L² cos(2πx/L) cos(2πy/L)
    apply!(x->-8π^2/L^2*cos(2π*x[1]/L)*cos(2π*x[2]/L),pois.z)

    # solver
    MG ? solver!(pois;tol=10eps(T),itmx=32) : solver!(pois;tol=10eps(T),itmx=1e4)

    # show stats and save
    me()==0 && println("Iters $(pois.n)")
    save("test_$(string(Pois))_$(me()).jld2","C",pois.x)
    save("test_$(string(Pois))_$(me())_error.jld2","C",pois.x.-uₑ)
    MG ? pois.levels[1].r .= pois.levels[1].x.-uₑ : pois.r .= pois.x.-uₑ
    L2 = √WaterLily.L₂(pois)/length(x)
    Linf = √WaterLily.L∞(pois)/length(x)
    me()==0 && println("L₂-norm of error $L2")
    me()==0 && println("L∞-norm of error $Linf")
end
finalize_mpi()