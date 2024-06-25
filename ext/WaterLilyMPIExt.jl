module WaterLilyMPIExt

if isdefined(Base, :get_extension)
    using MPI
else
    using ..MPI
end

using StaticArrays
using WaterLily
import WaterLily: init_mpi,me,BC!,L₂,L∞,loc,⋅,finalize_mpi

const NDIMS_MPI = 3           # Internally, we set the number of dimensions always to 3 for calls to MPI. This ensures a fixed size for MPI coords, neigbors, etc and in general a simple, easy to read code.
const NNEIGHBORS_PER_DIM = 2

"""
    halos(dims,d)

Return the CartesianIndices of the halos in dimension `±d` of an array of size `dims`.
"""
function halos(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (1:2) : (dims[i]-1:dims[i]) : (1:dims[i]), N))
end
"""
    buff(dims,d)

Return the CartesianIndices of the buffer in dimension `±d` of an array of size `dims`.
"""
function buff(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (3:4) : (dims[i]-3:dims[i]-2) : (1:dims[i]), N))
end

"""
    mpi_swap!(send1,recv1,send2,recv2,neighbor,comm)

This function swaps the data between two MPI processes. The data is sent from `send1` to `neighbor[1]` and received in `recv1`.
The data is sent from `send2` to `neighbor[2]` and received in `recv2`. The function is non-blocking and returns when all data 
has been sent and received. 
"""
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

"""
    BC!(a)

This function sets the boundary conditions of the array `a` using the MPI grid.
"""
function BC!(a;perdir=(0,))
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

function BC!(a,A,saveexit=false,perdir=(0,))
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
# grid_loc(;grid=MPI_GRID_NULL) = 0
# grid_loc() = mpi_grid().global_loc
me()= mpi_grid().me
neighbor(d,i) = mpi_grid().neighbors[i,d]
neighbor(d) = mpi_grid().neighbors[:,d]

# every process must redifine the loc to be global
@inline function loc(i,I::CartesianIndex{N},T=Float64) where N
    # global position in the communicator
    SVector{N,T}(mpi_grid().global_loc .+ I.I .- 2.5 .- 0.5 .* δ(i,I).I)
end

L₂(a) = MPI.Allreduce(sum(abs2,@inbounds(a[I]) for I ∈ inside(a)),+,mpi_grid().comm)
function L₂(p::Poisson)
    s = zero(eltype(p.r))
    for I ∈ inside(p.r)
        @inbounds s += p.r[I]*p.r[I]
    end
    MPI.Allreduce(s,+,mpi_grid().comm)
end
L∞(p::Poisson) = MPI.Allreduce(maximum(abs.(p.r)),Base.max,mpi_grid().comm)
function ⋅(a::AbstractArray{T},b::AbstractArray{T}) where T
    s = zero(T)
    for I ∈ inside(a)
        @inbounds s += a[I]*b[I]
    end
    MPI.Allreduce(s,+,mpi_grid().comm)
end

end # module