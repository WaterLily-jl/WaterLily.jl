using WaterLily
using CUDA: cu, @allowscalar
using BenchmarkTools

@fastmath function mom_step_benchmark!(a::Flow,b::AbstractPoisson)
    a.u⁰ .= a.u
    a.u .= 0
    # predictor u → u'
    @btime conv_diff!($a.f,$a.u⁰,$a.σ,ν=$a.ν) # CPU: 10.308 ms (2730 allocations: 195.81 KiB). GPU: 177.927 μs (1287 allocations: 56.44 KiB)
    @btime BDIM!($a) # CPU: 11.231 ms (541 allocations: 39.53 KiB). GPU: 50.584 μs (360 allocations: 26.66 KiB)
    @btime BC!($a.u,$a.U) # CPU: 139.339 μs (1356 allocations: 97.41 KiB). GPU: 67.421 μs (611 allocations: 26.11 KiB)
    @btime project!($a,$b) # CPU: 9.187 ms (550 allocations: 39.72 KiB). GPU: 435.650 ms (1285003 allocations: 56.93 MiB)
    @btime BC!($a.u,$a.U) # CPU: 138.663 μs (1355 allocations: 97.38 KiB). GPU: 68.190 μs (611 allocations: 26.11 KiB)
    # corrector u → u¹
    @btime conv_diff!($a.f,$a.u,$a.σ,ν=$a.ν) # CPU: 11.308 ms (2728 allocations: 195.75 KiB). GPU: 181.429 μs (1287 allocations: 56.44 KiB)
    @btime BDIM!($a) # CPU: 11.354 ms (542 allocations: 39.56 KiB). GPU: 57.634 μs (360 allocations: 26.66 KiB)
    @btime BC!($a.u,$a.U,2) # CPU: 137.151 μs (1355 allocations: 97.38 KiB). GPU:  68.276 μs (611 allocations: 26.11 KiB)
    @btime project!($a,$b,2) # CPU: 8.200 ms (550 allocations: 39.72 KiB). GPU: 424.575 ms (1285013 allocations: 56.93 MiB)
    a.u ./= 2
    @btime BC!($a.u,$a.U) # CPU: 137.820 μs (1356 allocations: 97.41 KiB). GPU: 68.100 μs (611 allocations: 26.11 KiB)
    @btime push!($a.Δt,CFL($a)) # CPU: 4.298 ms (9 allocations: 528 bytes). GPU: N/A
end

U, N = (2/3, -1/3), (2^10, 2^10)
flowCPU = Flow(N, U)
flowGPU = Flow(N, U; f=cu)

# mom_step_benchmark!(flowCPU, MultiLevelPoisson(flowCPU.p, flowCPU.μ₀))
# mom_step_benchmark!(flowGPU, MultiLevelPoisson(flowGPU.p, flowGPU.μ₀))

mom_step!(flowCPU, MultiLevelPoisson(flowCPU.p, flowCPU.μ₀))
mom_step!(flowGPU, MultiLevelPoisson(flowGPU.p, flowGPU.μ₀))

println(L₂(flowCPU.u[:,:,1].-U[1]) < 2e-5) # true
println(L₂(flowCPU.u[:,:,2].-U[2]) < 1e-5) # true
@allowscalar println(L₂(flowGPU.u[:,:,1].-U[1]) < 2e-5) # false
@allowscalar println(L₂(flowGPU.u[:,:,2].-U[2]) < 1e-5) # false