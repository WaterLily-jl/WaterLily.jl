using WaterLily
using BenchmarkTools
using CUDA: CuArray, @allowscalar, allowscalar, @sync
allowscalar(false)

@fastmath function mom_step_benchmark!(a::Flow,b::AbstractPoisson)
    @btime @sync WaterLily.conv_diff!($a.f,$a.u⁰,$a.σ,ν=$a.ν) 
    @btime @sync WaterLily.BDIM!($a)
    @btime @sync BC!($a.u,$a.U) 
    @btime @sync WaterLily.project!($a,$b) 
    @btime @sync WaterLily.CFL($a)
end

U, N = (0, 0), (2^10, 2^10)
flowCPU = Flow(N, U, T=Float32);
flowGPU = Flow(N, U; f=CuArray, T=Float32);

## SERIAL BASELINE using @inbounds @simd instead of @kernel
# 18.359 ms (0 allocations: 0 bytes)
# 3.456 ms (0 allocations: 0 bytes)
# 7.600 μs (0 allocations: 0 bytes)
# 2.219 ms (0 allocations: 0 bytes)
# 1.021 ms (0 allocations: 0 bytes)

mom_step_benchmark!(flowCPU, MultiLevelPoisson(flowCPU.p, flowCPU.μ₀, flowCPU.σ))
# 1.977 ms (3240 allocations: 280.97 KiB)   # 10x speed-up
# 955.100 μs (416 allocations: 37.25 KiB)   # 3.5x
# 146.000 μs (1954 allocations: 165.02 KiB)
# 741.600 μs (845 allocations: 73.03 KiB)   # 3x
# 851.600 μs (218 allocations: 17.59 KiB)

mom_step_benchmark!(flowGPU, MultiLevelPoisson(flowGPU.p, flowGPU.μ₀, flowGPU.σ))
# 1.182 ms (1108 allocations: 50.75 KiB)    # 15x speed-up
# 662.800 μs (133 allocations: 7.23 KiB)    # 5x
# 137.000 μs (611 allocations: 25.17 KiB)
# 604.700 μs (270 allocations: 12.47 KiB)   # 3.5x
# 298.000 μs (149 allocations: 6.73 KiB)    # 5x