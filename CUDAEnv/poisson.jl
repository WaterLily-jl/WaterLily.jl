using WaterLily
using BenchmarkTools
using CUDA: CuArray

function Poisson_setup(poisson,N::NTuple{D};f=Array,T=Float32) where D
    c = ones(T,N...,D) |> f; BC!(c,ntuple(zero,D))
    x = zeros(T,N) |> f
    p = poisson(x,c)
    soln = map(I->T(I.I[1]),CartesianIndices(N)) |> f
    WaterLily.residual!(p,mult(p,soln))
    return p
end

@fastmath function poisson_benchmark!(pois::AbstractPoisson)
    @btime WaterLily.residual!($pois,$pois.ϵ);
    @btime L₂($pois)
    @btime WaterLily.increment!($pois)
    @btime WaterLily.Jacobi!($pois,it=1)
    @btime WaterLily.pcg!($pois,it=1)
    typeof(pois.r)<:Array && @btime WaterLily.SOR!($pois,it=1)
end

#SERIAL BASELINE
# 482.100 μs (0 allocations: 0 bytes)
# 62.200 μs (0 allocations: 0 bytes)
# 603.300 μs (0 allocations: 0 bytes)
# 931.700 μs (0 allocations: 0 bytes)
# 1.497 ms (0 allocations: 0 bytes)
# 8.099 ms (0 allocations: 0 bytes)
pois = Poisson_setup(Poisson,(2^10+2,2^10+2));
poisson_benchmark!(pois)
# 239.900 μs (207 allocations: 18.12 KiB) 2x speed-up
# 62.500 μs (0 allocations: 0 bytes)
# 357.600 μs (412 allocations: 34.91 KiB) 1.5x
# 600.500 μs (621 allocations: 52.44 KiB) 1.5x
# 1.442 ms (834 allocations: 70.31 KiB)   NOTHING?!?
# 8.286 ms (414 allocations: 34.97 KiB)
pois = Poisson_setup(Poisson,(2^10+2,2^10+2),f=CuArray);
poisson_benchmark!(pois)
# 3.262 μs (65 allocations: 3.30 KiB)     150x speed-up
# 53.800 μs (130 allocations: 5.77 KiB)  
# 6.100 μs (122 allocations: 5.55 KiB)    100x
# 9.000 μs (183 allocations: 8.25 KiB)    100x
# 602.800 μs (287 allocations: 12.31 KiB) 2x ONLY!
# false

@fastmath function mlpoisson_benchmark!(ml::MultiLevelPoisson)
    fine,coarse = ml.levels[1],ml.levels[2];
    @btime WaterLily.restrict!($coarse.r,$fine.r)
    @btime WaterLily.prolongate!($fine.ϵ,$coarse.x)
    @btime WaterLily.Vcycle!($ml)
end

# SERIAL BASELINE
# 583.000 μs (0 allocations: 0 bytes)
# 427.900 μs (0 allocations: 0 bytes)
# 6.779 ms (0 allocations: 0 bytes)
ml = Poisson_setup(MultiLevelPoisson,(2^10+2,2^10+2));
mlpoisson_benchmark!(ml)
# 136.300 μs (205 allocations: 17.11 KiB) 4x speed-up
# 103.200 μs (207 allocations: 17.16 KiB) 4x
# 8.078 ms (29898 allocations: 2.47 MiB) !! Too many allocations
ml = Poisson_setup(MultiLevelPoisson,(2^10+2,2^10+2),f=CuArray);
mlpoisson_benchmark!(ml)
# 2.862 μs (59 allocations: 2.50 KiB)     200x speed-up
# 3.100 μs (59 allocations: 2.50 KiB)     140x
# 4.129 ms (9262 allocations: 396.66 KiB) 1.5x ONLY!! 