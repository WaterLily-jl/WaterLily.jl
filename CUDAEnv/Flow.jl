using WaterLily
using BenchmarkTools
using CUDA: cu, @allowscalar, allowscalar
allowscalar(false)

@fastmath function mom_step_benchmark!(a::Flow,b::AbstractPoisson)
    @btime WaterLily.conv_diff!($a.f,$a.u⁰,$a.σ,ν=$a.ν) 
    @btime WaterLily.BDIM!($a) 
    @btime BC!($a.u,$a.U) 
    @btime WaterLily.project!($a,$b) 
    @btime WaterLily.CFL($a)
end

U, N = (0, 0), (2^10, 2^10)
flowCPU = Flow(N, U, T=Float32);
flowGPU = Flow(N, U; f=cu, T=Float32);

## SERIAL BASELINE using @inbounds @simd instead of @kernel
# mom_step_benchmark!(flowCPU, MultiLevelPoisson(flowCPU.p, flowCPU.μ₀))
# 18.359 ms (0 allocations: 0 bytes)
# 3.456 ms (0 allocations: 0 bytes)
# 7.600 μs (0 allocations: 0 bytes)
# 2.219 ms (0 allocations: 0 bytes)
# 1.021 ms (0 allocations: 0 bytes)

mom_step_benchmark!(flowCPU, MultiLevelPoisson(flowCPU.p, flowCPU.μ₀))
# 2.789 ms (4052 allocations: 348.88 KiB) # 6.5x speed-up of the most expensive routine !!
# 3.910 ms (829 allocations: 72.53 KiB)  
# 136.800 μs (1941 allocations: 164.52 KiB) # 18x slower !!
# 2.090 ms (834 allocations: 72.59 KiB)
# 852.300 μs (207 allocations: 17.16 KiB)

mom_step_benchmark!(flowGPU, MultiLevelPoisson(flowGPU.p, flowGPU.μ₀))
# 74.500 μs (1287 allocations: 56.44 KiB) # 250x
# 22.200 μs (360 allocations: 26.66 KiB)  # 150x
# 33.900 μs (611 allocations: 25.17 KiB)  # 224x
# 720.600 μs (399 allocations: 18.22 KiB) # 3x
# 260.900 μs (149 allocations: 6.73 KiB)  # 4x