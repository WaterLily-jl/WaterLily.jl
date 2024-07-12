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
    send1 = a[buff(N,-d)]; send2 =a[buff(N,+d)]
    recv1 = zero(send1);   recv2 = zero(send2)
    # swap 
    mpi_swap!(send1,recv1,send2,recv2,neighbors(d),mpi_grid().comm)

    # this sets the BCs
    !mpi_wall(d,1) && (a[halos(N,-d)] .= recv1) # halo swap
    !mpi_wall(d,2) && (a[halos(N,+d)] .= recv2) # halo swap
end

function WaterLily.BC!(a,A,saveexit=false,perdir=())
    N,n = WaterLily.size_u(a)
    for i ∈ 1:n, d ∈ 1:n
        # get data to transfer @TODO use @views
        send1 = a[buff(N,-d),i]; send2 = a[buff(N,+d),i]
        recv1 = zero(send1);     recv2 = zero(send2)
        # swap 
        mpi_swap!(send1,recv1,send2,recv2,neighbors(d),mpi_grid().comm)

        # this sets the BCs on the domain boundary and transfers the data
        if mpi_wall(d,1) # left wall
            if i==d # set flux
                a[halos(N,-d),i] .= A[i]
                a[WaterLily.slice(N,3,d),i] .= A[i]
            else # zero gradient
                a[halos(N,-d),i] .= reverse(send1; dims=d)
            end
        else # neighbor on the left
            a[halos(N,-d),i] .= recv1
        end
        if mpi_wall(d,2) # right wall
            if i==d && (!saveexit || i>1) # convection exit
                a[halos(N,+d),i] .= A[i]
            else # zero gradient
                a[halos(N,+d),i] .= reverse(send2; dims=d)
            end
        else # neighbor on the right
            a[halos(N,+d),i] .= recv2
        end
    end
end

function WaterLily.exitBC!(u,u⁰,U,Δt)
    N,_ = size_u(u)
    exitR = slice(N.-2,N[1]-2,1,3) # exit slice excluding ghosts
    if mpi_wall(1,2) #right wall
        @loop u[I,1] = u⁰[I,1]-U[1]*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ exitR
        ∮udA = sum(u[exitR,1])/length(exitR)-U[1]   # mass flux imbalance
    else
        ∮udA = 0
    end
    ∮u = MPI.Allreduce(⨕udA,+,mpi_grid().comm)           # domain imbalance
    mpi_wall(1,2) && (@loop u[I,1] -= ∮u over I ∈ exitR) # correct flux only on right wall
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
neighbors(dim) = mpi_grid().neighbors[:,dim]
mpi_wall(dim,i) = mpi_grid().neighbors[i,dim]==MPI.PROC_NULL

# every process must redifine the loc to be global
@inline function WaterLily.loc(i,I::CartesianIndex{N},T=Float64) where N
    # global position in the communicator
    SVector{N,T}(grid_loc() .+ I.I .- 2.5 .- 0.5 .* δ(i,I).I)
end

function WaterLily.L₂(a)
    MPI.Allreduce(sum(abs2,@inbounds(a[I]) for I ∈ inside(a)),+,mpi_grid().comm)
end
function WaterLily.L₂(p::Poisson) # won't work on the GPU
    s = zero(eltype(p.r))
    for I ∈ inside(p.r)
        @inbounds s += p.r[I]*p.r[I]
    end
    MPI.Allreduce(s,+,mpi_grid().comm)
end
WaterLily.L∞(a::AbstractArray) = MPI.Allreduce(maximum(abs.(a)),Base.max,mpi_grid().comm)
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
# this actually add a global comminutation every time residual is called
function WaterLily.residual!(p::Poisson) 
    WaterLily.perBC!(p.x,p.perdir)
    @inside p.r[I] = ifelse(p.iD[I]==0,0,p.z[I]-WaterLily.mult(I,p.L,p.D,p.x))
    # s = sum(p.r)/length(inside(p.r))
    s = MPI.Allreduce(sum(p.r)/length(inside(p.r)),+,mpi_grid().comm)
    abs(s) <= 2eps(eltype(s)) && return
    @inside p.r[I] = p.r[I]-s
end

function WaterLily.sim_step!(sim::Simulation,t_end;remeasure=true,max_steps=typemax(Int),verbose=false)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure)
        (verbose && me()==0) && println("tU/L=",round(sim_time(sim),digits=4),
                                        ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
