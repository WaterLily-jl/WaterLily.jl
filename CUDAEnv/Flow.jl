using WaterLily
using CUDA: cu
using BenchmarkTools

U, N = (2/3, -1/3), (2^10, 2^10)
flowCPU = Flow(N, U)
flowGPU = Flow(N, U; f=cu)

@btime mom_step!($flowCPU, MultiLevelPoisson($flowCPU.p, $flowCPU.μ₀))
@btime mom_step!($flowGPU, MultiLevelPoisson($flowGPU.p, $flowGPU.μ₀))

# mom_step!(flowCPU, MultiLevelPoisson(flowCPU.p, flowCPU.μ₀))
# mom_step!(flowGPU, MultiLevelPoisson(flowGPU.p, flowGPU.μ₀))

# println(L₂(flowCPU.u[:,:,1].-U[1]) < 2e-5)
# println(L₂(flowCPU.u[:,:,2].-U[2]) < 1e-5)
# @allowscalar println(L₂(flowGPU.u[:,:,1].-U[1]) < 2e-5)
# @allowscalar println(L₂(flowGPU.u[:,:,2].-U[2]) < 1e-5)

return nothing;