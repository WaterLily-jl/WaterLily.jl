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


function WaterLily.BC!(a;perdir=(0,))
    N = size(a)
    for d ∈ eachindex(N)    # this is require because scalar and vector field are located 
                            # at different location
        # get data to transfer
        send1 = a[buff(N,-d)]; send2 = a[buff(N,+d)]
        recv1 = zero(send1);   recv2 = zero(send2)
        # swap 
        mpi_swap!(send1,recv1,send2,recv2,mpi_grid().neighbors[:,d],mpi_grid().comm)

        # this sets the BCs
        if mpi_grid().neighbors[1,d]==MPI.PROC_NULL # right wall
            a[halos(N,-d)] .= reverse(send1; dims=d)
        else # halo swap
            a[halos(N,-d)] .= recv1
        end
        if mpi_grid().neighbors[2,d]==MPI.PROC_NULL # right wall
            a[halos(N,+d)] .= reverse(send2; dims=d)
        else # halo swap
            a[halos(N,+d)] .= recv2
        end
    end
end

function WaterLily.BC!(a,A,saveexit=false,perdir=(0,))
    N,n = WaterLily.size_u(a)
    for i ∈ 1:n, j ∈ 1:n
        # get data to transfer
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

macro mpi_initialized() :(mpi_initialized();) end
let
    global MPIGrid, set_mpi_grid, mpi_grid ,mpi_initialized

    # allows to access the global mpi grid
    _mpi_grid::MPIGrid          = MPI_GRID_NULL
    mpi_grid()::MPIGrid         = (_mpi_grid::MPIGrid)
    set_mpi_grid(grid::MPIGrid) = (_mpi_grid = grid;)
    mpi_initialized()           = (_mpi_grid.comm != MPI.COMM_NULL)
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
# macro grid_loc() esc(:(0)) end
macro grid_loc() esc(:( mpi_grid().global_loc )) end
macro me() esc(:( mpi_grid().rank )) end

# every process must redifine the loc to be global
@inline function WaterLily.loc(i,I::CartesianIndex{N},T=Float64) where N
    # global position in the communicator
    SVector{N,T}(@grid_loc() .+ I.I .- 2.5 .- 0.5 .* δ(i,I).I)
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
me = init_mpi((nx,ny))
sim = circle(nx,ny,SA[ny,ny],nx/4)

(me ==0) && println("nx=$nx, ny=$ny")

# check global coordinates
xs = loc(0,CartesianIndex(3,3))
println("I am rank $me, at global coordinate $xs")

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
@WaterLily.inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
BC!(sim.flow.σ)
save("waterlily_4_$me.jld2", "sdf", sim.flow.σ)
WaterLily.project!(sim.flow,sim.pois)
BC!(sim.flow.u,U)

# WaterLily.smooth!(sim.pois.levels[1])

# @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U

finalize_mpi()
