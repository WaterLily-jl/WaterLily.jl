using WaterLily
using BenchmarkTools
using CUDA: CuArray,@sync

function Poisson_setup(poisson,N::NTuple{D};f=Array,T=Float32) where D
    c = ones(T,N...,D) |> f; BC!(c,ntuple(zero,D))
    x = zeros(T,N) |> f
    p = poisson(x,c)
    soln = map(I->T(I.I[1]),CartesianIndices(N)) |> f
    WaterLily.residual!(p,mult(p,soln))
    return p
end

using LinearAlgebra: ⋅
using KernelAbstractions
function pcg_benchmark!(p::Poisson)
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    alpha = π
    @btime @sync @inside $z[I] = $r[I]*$p.iD[I]
    @btime @sync @inside $z[I] = $ϵ[I] = $r[I]*$p.iD[I]
    @btime @sync @inside $z[I] = mult(I,$p.L,$p.D,$ϵ)
    @btime @sync WaterLily.@loop ($x[I] += $alpha*$ϵ[I]; 
                                  $r[I] -= $alpha*$z[I]) over I ∈ inside($x)
    @btime @sync $r ⋅ $z
    return
end
# SERIAL BASELINE
# 6.662 ms (0 allocations: 0 bytes)
# 9.666 ms (0 allocations: 0 bytes)
# 23.709 ms (0 allocations: 0 bytes)
# 9.187 ms (0 allocations: 0 bytes)
# 3.852 ms (0 allocations: 0 bytes)
##
# 202.400 μs (0 allocations: 0 bytes)
# 374.700 μs (0 allocations: 0 bytes)
# 416.700 μs (0 allocations: 0 bytes)
# 340.100 μs (0 allocations: 0 bytes)
# 109.200 μs (0 allocations: 0 bytes)
pois = Poisson_setup(Poisson,(2^8+2,2^8+2,2^8+2));
pcg_benchmark!(pois)
# 4.173 ms (207 allocations: 18.12 KiB)
# 6.810 ms (207 allocations: 18.44 KiB)
# 9.520 ms (207 allocations: 18.44 KiB)
# 6.795 ms (207 allocations: 18.44 KiB)
# 3.919 ms (0 allocations: 0 bytes)
##
# 79.800 μs (207 allocations: 17.47 KiB)   2.5x speed-up
# 107.400 μs (207 allocations: 17.81 KiB)  3.5x
# 135.300 μs (207 allocations: 17.81 KiB)  3x
# 112.400 μs (207 allocations: 17.81 KiB)  3x
# 108.500 μs (0 allocations: 0 bytes)
pois = Poisson_setup(Poisson,(2^8+2,2^8+2,2^8+2),f=CuArray);
pois = Poisson_setup(Poisson,(2^10+2,2^10+2),f=CuArray);
pcg_benchmark!(pois)
# 1.345 ms (61 allocations: 3.20 KiB)
# 1.744 ms (112 allocations: 6.78 KiB)
# 3.625 ms (112 allocations: 6.83 KiB)
# 2.638 ms (111 allocations: 6.75 KiB)
# 829.600 μs (1 allocation: 16 bytes)
##
# 98.700 μs (61 allocations: 2.70 KiB)
# 125.400 μs (63 allocations: 3.02 KiB)
# 153.000 μs (63 allocations: 3.05 KiB)
# 187.400 μs (63 allocations: 3.02 KiB)
# 95.000 μs (1 allocation: 16 bytes)

@fastmath function poisson_benchmark!(pois::AbstractPoisson)
    @btime @sync WaterLily.residual!($pois,$pois.ϵ);
    @btime @sync L₂($pois)
    @btime @sync WaterLily.increment!($pois)
    @btime @sync WaterLily.Jacobi!($pois,it=1)
    @btime @sync WaterLily.pcg!($pois,it=1)
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
# 142.700 μs (207 allocations: 18.12 KiB)
# 62.700 μs (0 allocations: 0 bytes)
# 214.100 μs (414 allocations: 34.97 KiB)
# 320.000 μs (621 allocations: 52.44 KiB)
# 786.500 μs (625 allocations: 53.81 KiB)
# 8.187 ms (413 allocations: 34.94 KiB)
pois = Poisson_setup(Poisson,(2^10+2,2^10+2),f=CuArray);
poisson_benchmark!(pois)
# 181.100 μs (65 allocations: 3.30 KiB)
# 55.800 μs (1 allocation: 16 bytes)
# 270.100 μs (122 allocations: 5.55 KiB)
# 365.500 μs (183 allocations: 8.25 KiB)
# 618.900 μs (200 allocations: 9.27 KiB)
# false

@fastmath function mlpoisson_benchmark!(ml::MultiLevelPoisson)
    fine,coarse = ml.levels[1],ml.levels[2];
    @btime @sync WaterLily.restrict!($coarse.r,$fine.r)
    @btime @sync WaterLily.prolongate!($fine.ϵ,$coarse.x)
    @btime @sync WaterLily.Vcycle!($ml)
end

# SERIAL BASELINE
# 10.799 ms (0 allocations: 0 bytes)
# 7.750 ms (0 allocations: 0 bytes)
# 148.623 ms (0 allocations: 0 bytes)
##
# 583.000 μs (0 allocations: 0 bytes)
# 427.900 μs (0 allocations: 0 bytes)
# 6.779 ms (0 allocations: 0 bytes)
ml = Poisson_setup(MultiLevelPoisson,(2^8+2,2^8+2,2^8+2));
ml = Poisson_setup(MultiLevelPoisson,(2^10+2,2^10+2));
mlpoisson_benchmark!(ml)
# 2.286 ms (207 allocations: 17.78 KiB)
# 2.291 ms (207 allocations: 17.78 KiB)
# 57.975 ms (20389 allocations: 1.75 MiB)
##
# 120.000 μs (205 allocations: 17.11 KiB)
# 91.100 μs (206 allocations: 17.12 KiB)
# 5.711 ms (24864 allocations: 2.07 MiB) !! Too many allocations
ml = Poisson_setup(MultiLevelPoisson,(2^8+2,2^8+2,2^8+2),f=CuArray);
ml = Poisson_setup(MultiLevelPoisson,(2^10+2,2^10+2),f=CuArray);
mlpoisson_benchmark!(ml)
# 548.500 μs (59 allocations: 2.84 KiB)
# 941.500 μs (107 allocations: 6.12 KiB)
# 17.288 ms (2789 allocations: 145.70 KiB)
##
# 58.600 μs (59 allocations: 2.50 KiB)
# 66.000 μs (59 allocations: 2.50 KiB)
# 4.890 ms (7734 allocations: 345.75 KiB)