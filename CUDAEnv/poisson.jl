using WaterLily
using BenchmarkTools
using CUDA: CuArray

function Poisson_setup(poisson,N;f=Array,T=Float32,D=length(N))
    c = ones(T,N...,D) |> f; BC!(c, zeros(T,D)|>f)
    x = zeros(T,N) |> f
    soln = map(I->T(I.I[1]),CartesianIndices(N)) |> f
    return poisson(x,c),soln
end

@fastmath function poisson_benchmark!(pois::AbstractPoisson,soln::AbstractArray)
    b = mult(pois,soln)
    @btime WaterLily.residual!($pois,$b);
    @btime L₂($pois.r)
    @btime WaterLily.increment!($pois)
    @btime WaterLily.Jacobi!($pois,it=1)
    typeof(pois.r)<:Array && @btime WaterLily.SOR!($pois,it=1)
end

#SERIAL BASELINE
# 482.100 μs (0 allocations: 0 bytes)
# 465.200 μs (0 allocations: 0 bytes)
# 603.300 μs (0 allocations: 0 bytes)
# 931.700 μs (0 allocations: 0 bytes)
# 8.099 ms (0 allocations: 0 bytes)
pois,soln = Poisson_setup(Poisson,(2^10+2,2^10+2));
poisson_benchmark!(pois,soln)
# 283.300 μs (207 allocations: 18.12 KiB) 2x speed-up
# 480.400 μs (0 allocations: 0 bytes)
# 353.300 μs (413 allocations: 34.94 KiB) 2x
# 626.100 μs (621 allocations: 52.44 KiB) 1.5x
# 8.541 ms (414 allocations: 34.97 KiB)
pois,soln = Poisson_setup(Poisson,(2^10+2,2^10+2),f=CuArray);
poisson_benchmark!(pois,soln)
# 3.225 μs (65 allocations: 3.30 KiB)     150x speed-up
# 188.000 μs (130 allocations: 5.77 KiB)  2.5x
# 6.575 μs (122 allocations: 5.55 KiB)    100x 
# 10.100 μs (183 allocations: 8.25 KiB)   93x

@fastmath function mlpoisson_benchmark!(ml::MultiLevelPoisson,soln::AbstractArray)
    fine,coarse = ml.levels[1],ml.levels[2];
    WaterLily.residual!(fine,mult(fine,soln));
    @btime WaterLily.restrict!($coarse.r,$fine.r)
    @btime WaterLily.prolongate!($fine.ϵ,$coarse.x)
    @btime WaterLily.Vcycle!($ml)
end

# SERIAL BASELINE
# 583.000 μs (0 allocations: 0 bytes)
# 427.900 μs (0 allocations: 0 bytes)
# 11.771 ms (0 allocations: 0 bytes)
ml,soln = Poisson_setup(MultiLevelPoisson,(2^10+2,2^10+2));
mlpoisson_benchmark!(ml,soln)
# 136.300 μs (205 allocations: 17.11 KiB) 4x speed-up
# 103.200 μs (207 allocations: 17.16 KiB) 4x
# 14.007 ms (19250 allocations: 1.55 MiB) NOTHING! (Maybe need to use fewer levels)
ml,soln = Poisson_setup(MultiLevelPoisson,(2^10+2,2^10+2),f=CuArray);
mlpoisson_benchmark!(ml,soln)
# 2.862 μs (59 allocations: 2.50 KiB)     200x speed-up
# 3.100 μs (59 allocations: 2.50 KiB)     140x
# 2.129 ms (36302 allocations: 1.61 MiB)  5x 