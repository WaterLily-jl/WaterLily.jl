#mpiexecjl --project=../examples/ -np 4 julia test_mpi.jl
using WaterLily
using MPI
using StaticArrays
using FileIO,JLD2

# using WaterLily: inside,@loop,mult,⋅
# function WaterLily.pcg!(p::Poisson{T};it=6) where T
#     me()==0 && println("pcg! started $it iterations")
#     x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
#     @inside z[I] = ϵ[I] = r[I]*p.iD[I]
#     insideI = inside(x) # [insideI]
#     rho = T(r⋅z)
#     println("pcg! computed inital residuals $rho")
#     abs(rho)<10eps(T) && return
#     for i in 1:it
#         me()==0 && print("pcg! iteration: $i, $rho")
#         BC!(ϵ;perdir=p.perdir)
#         @inside z[I] = mult(I,p.L,p.D,ϵ)
#         alpha = rho/T(z[insideI]⋅ϵ[insideI])
#         @loop (x[I] += alpha*ϵ[I];
#                r[I] -= alpha*z[I]) over I ∈ inside(x)
#         (i==it || abs(alpha)<1e-2) && return
#         @inside z[I] = r[I]*p.iD[I]
#         rho2 = T(r⋅z)
#         me()==0 && println(", rho2 $rho2")
#         abs(rho2)<10eps(T) && return
#         beta = rho2/rho
#         @inside ϵ[I] = beta*ϵ[I]+z[I]
#         rho = rho2
#     end
# end

"""Flow around a circle"""
function circle(n,m,center,radius;Re=250,U=1)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, psolver=Poisson)
end

# local grid size
nx = 2^6
ny = 2^5

# init the MPI grid and the simulation
r = init_mpi((nx,ny))
sim = circle(nx,ny,SA[ny,ny],nx/4)

(me()==0) && println("nx=$nx, ny=$ny")

# check global coordinates
xs = loc(0,CartesianIndex(3,3))
println("I am rank $r, at global coordinate $xs")

# first we chack s imple rank matrix
# sim.flow.σ .= NaN
# sim.flow.μ₀ .= NaN
# sim.flow.σ[inside(sim.flow.σ)] .= reshape(collect(1:length(inside(sim.flow.σ))),size(inside(sim.flow.σ)))

# global_loc_function(i,x) = x[i]
# apply!(global_loc_function,sim.flow.μ₀)
# check that the measure uses the correct loc function
# measure_sdf!(sim.flow.σ,sim.body,0.0)
# save("waterlily_$me.jld2", "sdf", sim.flow.σ)

# second check is to check the μ₀
# sim.flow.σ .= sim.flow.μ₀[:,:,2]

# updating the halos should not do anything
save("waterlily_1_$me.jld2", "sdf", sim.flow.u⁰)

# # BC!(sim.flow.μ₀,zeros(SVector{2,Float64}))
# # BC!(sim.flow.σ)

# # sim.flow.σ .= sim.flow.μ₀[:,:,2]

# # sim_step!(sim, 10.0; verbose=true)
# # mom_step!(sim.flow,sim.pois)
sim.flow.u⁰ .= sim.flow.u; WaterLily.scale_u!(sim.flow,0)
# predictor u → u'
U = WaterLily.BCTuple(sim.flow.U,WaterLily.time(sim.flow),2)
(me == 0) && println("U = $U")
save("waterlily_2_$me.jld2", "sdf", sim.flow.u)
WaterLily.conv_diff!(sim.flow.f,sim.flow.u⁰,sim.flow.σ,ν=sim.flow.ν)
WaterLily.BDIM!(sim.flow); BC!(sim.flow.u,U)
save("waterlily_3_$me.jld2", "sdf", sim.flow.f)
WaterLily.project!(sim.flow,sim.pois)
# dt = sim.flow.Δt[end]
# @inside sim.pois.z[I] = WaterLily.div(I,sim.flow.u); sim.pois.x .*= dt # set source term & solution IC
# # solver!(b)
# BC!(sim.pois.x)
# WaterLily.residual!(sim.pois); r₂ = L₂(sim.pois)
# # save("waterlily_4_$me.jld2", "sdf", sim.pois.r)
# save("waterlily_4_$me.jld2", "sdf", sim.pois.L[:,:,2])
# WaterLily.smooth!(sim.pois)
# for i ∈ 1:2  # apply solution and unscale to recover pressure
#     @WaterLily.loop sim.flow.u[I,i] -= sim.pois.L[I,i]*WaterLily.∂(i,I,sim.pois.x) over I ∈ inside(sim.pois.x)
# end
# sim.pois.x ./= dt
BC!(sim.flow.u,U)

println("L₂(pois) in rank $me : $(L₂(sim.pois))")


sim.pois.r .= 0.0
me() == 2 && (sim.pois.r[32,32] = 123.456789) # make this the only non-zero element
println("L∞(pois) : $(WaterLily.L∞(sim.pois))")

# WaterLily.smooth!(sim.pois.levels[1])

# @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U

finalize_mpi()
