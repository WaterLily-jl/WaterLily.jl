using MPI,WaterLily
using StaticArrays
using FileIO,JLD2

const NDIMS_MPI = 3           # Internally, we set the number of dimensions always to 3 for calls to MPI. This ensures a fixed size for MPI coords, neigbors, etc and in general a simple, easy to read code.
const NNEIGHBORS_PER_DIM = 2


"""return the CI of the halos must only point to halos, otherwise it messes-up 
the reconstruction"""
function halos(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (1:2) : (dims[i]-1:dims[i]) : (1:dims[i]), N))
end
# return the CI of the buff 
function buff(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (3:4) : (dims[i]-3:dims[i]-2) : (1:dims[i]), N))
end


function mpi_swap!(send1,recv1,send2,recv2,neighbor,comm)
    reqs=MPI.Request[]
    # Send to / receive from neighbor 1 in dimension d
    push!(reqs,MPI.Isend(send1,  neighbor[1], 0, comm))
    push!(reqs,MPI.Irecv!(recv1, neighbor[1], 1, comm))
    # Send to / receive from neighbor 2 in dimension d
    push!(reqs,MPI.Irecv!(recv2, neighbor[2], 0, comm))
    push!(reqs,MPI.Isend(send2,  neighbor[2], 1, comm))
    # wair for all transfer to be done
    MPI.Waitall!(reqs)
end

WaterLily.perBC!(a,::Tuple{})          = perBC!(a, size(a), true)
WaterLily.perBC!(a, perdir, N=size(a)) = perBC!(a, N, true)
perBC!(a, N, mpi::Bool) = for d ∈ eachindex(N)
    # get data to transfer @TODO use @views
    send1 = a[buff(N,-d)]; send2 = a[buff(N,+d)]
    recv1 = zero(send1);   recv2 = zero(send2)
    # swap 
    mpi_swap!(send1,recv1,send2,recv2,mpi_grid().neighbors[:,d],mpi_grid().comm)

    # this sets the BCs
    if mpi_grid().neighbors[1,d]!=MPI.PROC_NULL # halo swap
        a[halos(N,-d)] .= recv1
    end
    if mpi_grid().neighbors[2,d]!=MPI.PROC_NULL # halo swap
        a[halos(N,+d)] .= recv2
    end
end

function WaterLily.BC!(a,A,saveexit=false,perdir=())
    N,n = WaterLily.size_u(a)
    for i ∈ 1:n, j ∈ 1:n
        # get data to transfer @TODO use @views
        send1 = a[buff(N,-j),i]; send2 = a[buff(N,+j),i]
        recv1 = zero(send1);     recv2 = zero(send2)
        # swap 
        mpi_swap!(send1,recv1,send2,recv2,mpi_grid().neighbors[:,j],mpi_grid().comm)

        # this sets the BCs on the domain boundary and transfers the data
        if mpi_grid().neighbors[1,j]==MPI.PROC_NULL # left wall
            if i==j # set flux
                a[halos(N,-j),i] .= A[i]
                a[WaterLily.slice(N,3,j),i] .= A[i]
            else # zero gradient
                a[halos(N,-j),i] .= reverse(send1; dims=j)
            end
        else # neighbor on the left
            a[halos(N,-j),i] .= recv1
        end
        if mpi_grid().neighbors[2,j]==MPI.PROC_NULL # right wall
            if i==j && (!saveexit || i>1) # convection exit
                a[halos(N,+j),i] .= A[i]
            else # zero gradient
                a[halos(N,+j),i] .= reverse(send2; dims=j)
            end
        else # neighbor on the right
            a[halos(N,+j),i] .= recv2
        end
    end
end

struct MPIGrid #{I,C<:MPI.Comm,N<:AbstractVector,M<:AbstractArray,G<:AbstractVector}
    me::Int                    # rank
    comm::MPI.Comm             # communicator
    coords::AbstractVector     # coordinates
    neighbors::AbstractArray   # neighbors
    global_loc::AbstractVector # the location of the lower left corner in global index space
end
const MPI_GRID_NULL = MPIGrid(-1,MPI.COMM_NULL,[-1,-1,-1],[-1 -1 -1; -1 -1 -1],[0,0,0])

let
    global MPIGrid, set_mpi_grid, mpi_grid, mpi_initialized, check_mpi

    # allows to access the global mpi grid
    _mpi_grid::MPIGrid          = MPI_GRID_NULL
    mpi_grid()::MPIGrid         = (check_mpi(); _mpi_grid::MPIGrid)
    set_mpi_grid(grid::MPIGrid) = (_mpi_grid = grid;)
    mpi_initialized()           = (_mpi_grid.comm != MPI.COMM_NULL)
    check_mpi()                 = !mpi_initialized() && error("MPI not initialized")
end

function init_mpi(Dims::NTuple{D};dims=[0, 0, 0],periods=[0, 0, 0],comm::MPI.Comm=MPI.COMM_WORLD,
                  disp::Integer=1,reorder::Bool=true) where D
    # MPI
    MPI.Init()
    nprocs = MPI.Comm_size(comm)
    # create cartesian communicator
    MPI.Dims_create!(nprocs, dims)
    comm_cart = MPI.Cart_create(comm, dims, periods, reorder)
    me     = MPI.Comm_rank(comm_cart)
    coords = MPI.Cart_coords(comm_cart)
    # make the cart comm
    neighbors = fill(MPI.PROC_NULL, NNEIGHBORS_PER_DIM, NDIMS_MPI);
    for i = 1:NDIMS_MPI
        neighbors[:,i] .= MPI.Cart_shift(comm_cart, i-1, disp);
    end
    # global index coordinate in grid space
    global_loc = SVector([coords[i]*Dims[i] for i in 1:D]...)
    set_mpi_grid(MPIGrid(me,comm_cart,coords,neighbors,global_loc))
    return me; # this is the most usefull MPI vriable to have in the local space
end
finalize_mpi() = MPI.Finalize()

# global coordinate in grid space
grid_loc(;grid=MPI_GRID_NULL) = 0
grid_loc(;grid=mpi_grid()) = grid.global_loc
me()= mpi_grid().me

# every process must redifine the loc to be global
@inline function WaterLily.loc(i,I::CartesianIndex{N},T=Float64) where N
    # global position in the communicator
    SVector{N,T}(grid_loc() .+ I.I .- 2.5 .- 0.5 .* δ(i,I).I)
end

function WaterLily.L₂(a)
    MPI.Allreduce(sum(abs2,@inbounds(a[I]) for I ∈ inside(a)),+,mpi_grid().comm)
end
function WaterLily.L₂(p::Poisson)
    s = zero(eltype(p.r))
    for I ∈ inside(p.r)
        @inbounds s += p.r[I]*p.r[I]
    end
    MPI.Allreduce(s,+,mpi_grid().comm)
end
WaterLily.L∞(p::Poisson) = MPI.Allreduce(maximum(abs.(p.r)),Base.max,mpi_grid().comm)
function WaterLily._dot(a::AbstractArray{T},b::AbstractArray{T}) where T
    s = zero(T)
    for I ∈ inside(a)
        @inbounds s += a[I]*b[I]
    end
    MPI.Allreduce(s,+,mpi_grid().comm)
end

function WaterLily.CFL(a::Flow;Δt_max=10)
    @inside a.σ[I] = WaterLily.flux_out(I,a.u)
    MPI.Allreduce(min(Δt_max,inv(maximum(a.σ)+5a.ν)),Base.min,mpi_grid().comm)
end

function WaterLily.sim_step!(sim::Simulation,t_end;remeasure=true,max_steps=typemax(Int),verbose=false)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure)
        (verbose && me()==0) && println("tU/L=",round(sim_time(sim),digits=4),
                                        ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

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
sim.flow.σ .= NaN
sim.flow.μ₀ .= NaN
sim.flow.σ[inside(sim.flow.σ)] .= reshape(collect(1:length(inside(sim.flow.σ))),size(inside(sim.flow.σ)))
save("waterlily_0_$(me()).jld2", "sdf", sim.flow.σ)

global_loc_function(i,x) = x[i]
apply!(global_loc_function,sim.flow.μ₀)
# check that the measure uses the correct loc function
measure_sdf!(sim.flow.σ,sim.body,0.0)
save("waterlily_1_$(me()).jld2", "sdf", sim.flow.σ)

# test norm functions
sim.pois.r .= 0.0
me() == 2 && (sim.pois.r[32,32] = 123.456789) # make this the only non-zero element
println("L∞(pois) : $(WaterLily.L∞(sim.pois))")

# updating the halos should not do anything
# save("waterlily_1_$(me()).jld2", "sdf", sim.flow.u⁰[inside(sim.flow.σ),:])
# # save("waterlily_2_$(me()).jld2", "sdf", sim.flow.f)

# # # BC!(sim.flow.μ₀,zeros(SVector{2,Float64}))
# # # BC!(sim.flow.σ)

# # # sim.flow.σ .= sim.flow.μ₀[:,:,2]

# sim_step!(sim, 10.0; verbose=true)

# # let
# #     sim.flow.u⁰ .= sim.flow.u; WaterLily.scale_u!(sim.flow,0)
# #     # predictor u → u'
# #     U = WaterLily.BCTuple(sim.flow.U,@view(sim.flow.Δt[1:end-1]),2)
# #     save("waterlily_2_$(me()).jld2", "sdf", sim.flow.u)
# #     WaterLily.conv_diff!(sim.flow.f,sim.flow.u⁰,sim.flow.σ,ν=sim.flow.ν)
# #     WaterLily.BDIM!(sim.flow);
# #     WaterLily.BC!(sim.flow.u,U)
# #     @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
# #     save("waterlily_3_$(me()).jld2", "sdf", sim.flow.σ)
# #     (me()==0) && println("projecting")
# #     WaterLily.project!(sim.flow,sim.pois)
# #     me()==0 && println(sim.pois.n)
# #     println("L2-pois", WaterLily.L₂(sim.pois))
# #     WaterLily.BC!(sim.flow.u,U)
# #     # save("waterlily_3_$(me()).jld2", "sdf", sim.flow.p)
# # end
# @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
# WaterLily.perBC!(sim.flow.σ,())
# save("waterlily_2_$(me()).jld2", "sdf", sim.flow.σ[inside(sim.flow.σ)])
# save("waterlily_3_$(me()).jld2", "sdf", sim.flow.p[inside(sim.flow.p)])
# save("waterlily_4_$(me()).jld2", "sdf", sim.flow.u[inside(sim.flow.σ),:])

finalize_mpi()