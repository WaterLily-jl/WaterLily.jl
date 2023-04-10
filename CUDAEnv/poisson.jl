using WaterLily
using BenchmarkTools
using CUDA

function Poisson_setup(poisson,N;f=identity,T=Float32,D=length(N))
    c = ones(T,N...,D) |> f|> OA(D)
    BC!(c, ntuple(zero,D), WaterLily.bc_indices(N) |> f)
    x = zeros(T,N) |> f |> OA()
    soln = map(I->T(I.I[1]),CartesianIndices(N)) |> f |> OA()
    return poisson(x,c),soln
end
pois,soln = Poisson_setup(Poisson,(2^10+2,2^10+2)); 
b = mult(pois,soln);
# 653.800 μs (0 allocations: 0 bytes) # 636.200 μs (215 allocations: 21.38 KiB) # 4.200 μs (83 allocations: 4.66 KiB)
@btime WaterLily.residual!($pois,$b);
# 640.100 μs (0 allocations: 0 bytes) # <-same # 191.000 μs (130 allocations: 5.77 KiB)
@btime L₂($pois.r)
# 619.000 μs (0 allocations: 0 bytes) # 695.700 μs (422 allocations: 38.19 KiB) # 7.275 μs (138 allocations: 6.86 KiB)
@btime WaterLily.increment!($pois)
# 969.700 μs (0 allocations: 0 bytes) # 1.064 ms (634 allocations: 57.12 KiB) # 10.900 μs (207 allocations: 10.14 KiB)
@btime WaterLily.Jacobi!($pois)
# 8.033 ms (0 allocations: 0 bytes) # 11.259 ms (422 allocations: 38.19 KiB) # false
typeof(pois.r.parent)<:Array && @btime WaterLily.SOR!($pois,it=1)

ml,soln = Poisson_setup(MultiLevelPoisson,(2^10+2,2^10+2),f=cu);
fine,coarse = ml.levels[1],ml.levels[2];
WaterLily.residual!(fine,mult(fine,soln));
# 637.800 μs (0 allocations: 0 bytes) # 135.500 μs (207 allocations: 17.83 KiB) # 2.737 μs (62 allocations: 2.78 KiB)
@btime WaterLily.restrict!($coarse.r,$fine.r)
# 446.200 μs (0 allocations: 0 bytes) # 198.100 μs (208 allocations: 17.84 KiB) # 2.850 μs (62 allocations: 2.78 KiB)
@btime WaterLily.prolongate!($fine.ϵ,$coarse.x)
# 11.833 ms (0 allocations: 0 bytes) # 14.754 ms (19673 allocations: 1.68 MiB) # 2.089 ms (41036 allocations: 1.97 MiB) SLOW
@btime WaterLily.Vcycle!($ml)
# Vcycle! with (2^8+2)^3
# 172.724 ms (0 allocations: 0 bytes) # 140.696 ms (17696 allocations: 1.65 MiB) # 1.924 ms (32015 allocations: 1.90 MiB)