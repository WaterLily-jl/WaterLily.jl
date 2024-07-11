using WaterLily
using StaticArrays
using Plots
include("../examples/TwoD_plots.jl")

WaterLily.L₂(ml::MultiLevelPoisson) = WaterLily.L₂(ml.levels[1])
WaterLily.L∞(ml::MultiLevelPoisson) = WaterLily.L∞(ml.levels[1])

# domain and fields
L = 32
N,D,T = (L,L),2,Float64
Ng = N .+ 4
x,z = zeros(T, Ng) |> Array, zeros(T, Ng) |> Array
μ⁰ = ones(T, (Ng..., D)) |> Array

# apply zero Neumann BC
BC!(μ⁰,zeros(2))

MG = false

# construct Poisson problem
pois = MG ? MultiLevelPoisson(x,μ⁰,z) : Poisson(x,μ⁰,z)
R = inside(pois.x)

# source term for the solution
# u := cos(2πx/L) cos(2πy/L)
# ∂u²/∂x² + ∂u²/∂y² = f = -8π²/L² cos(2πx/L) cos(2πy/L)
apply!(x->-8π^2/L^2*cos(2π*x[1]/L)*cos(2π*x[2]/L),pois.z)
flood(pois.z[R])

# solver
MG ? solver!(pois;tol=10eps(T),itmx=32) : solver!(pois;tol=10eps(T),itmx=1e4)
@show pois.n
flood(pois.x[R])
uₑ = copy(pois.x); apply!(x->cos(2π*x[1]/L)*cos(2π*x[2]/L),uₑ)
flood(uₑ[R])

@show WaterLily.L₂(pois)
@show WaterLily.L∞(pois)
flood(log10.(abs.(pois.x[R].-uₑ[R])),title="Log10 error")
