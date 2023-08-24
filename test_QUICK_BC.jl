using WaterLily

# # overwrites the function so we can see where we aplly it
# @inline function WaterLily.ϕu∂(a,I,f,u,λ=WaterLily.quick) 
#     if u>=0
#         println("CD at ", I," ",a)
#         return u*WaterLily.ϕ(a,I,f)
#     else
#         println("Quick at ", I," ",a)
#         return u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I])
#     end
# end

# # Initialize simulation
# # sim = Simulation((L, 1), U, L; ν=L/Re, T=Float64)
# flow = Flow((4,4),(1,0);uλ=(i,x)->i==1 ? x[1] : 0)
# flow.u⁰[1,:,1] .= 0
# # update flow
# WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ;ν=0.)
# # should use quick on the top and right boundaries, and CD on the bottom and left
# display(flow.u⁰)
# display(flow.f)


# Impulsive flow in a box
U = (2/3, -1/3)
N = (2^4, 2^4)
a = Flow(N, U; T=Float32)
mom_step!(a, MultiLevelPoisson(a.p,a.μ₀,a.σ))
println(L₂(a.u[:,:,1].-U[1]) < 2e-5)
println(L₂(a.u[:,:,2].-U[2]) < 1e-5)
