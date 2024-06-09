using MPI,FileIO
using WaterLily

const NDIMS_MPI = 3           # Internally, we set the number of dimensions always to 3 for calls to MPI. This ensures a fixed size for MPI coords, neigbors, etc and in general a simple, easy to read code.
const NNEIGHBORS_PER_DIM = 2

function halos(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (1:1) : (dims[i]:dims[i]) : (2:dims[i]-1), N))
end
# return the CI of the buff 
function buff(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (2:2) : (dims[i]-1:dims[i]-1) : (2:dims[i]-1), N))
end

function update_halo!(d, A, neighbors, comm)
    # Send to / receive from neighbor 1 in dimension d
    sendbuf = A[buff(size(A),-d)]
    recvbuf = zeros(length(sendbuf))
    MPI.Send(sendbuf,  neighbors[1,d], 0, comm)
    MPI.Recv!(recvbuf, neighbors[1,d], 1, comm)
    A[halos(size(A),-d)] .= reshape(recvbuf,size(halos(size(A),-d)))
    # Send to / receive from neighbor 2 in dimension d
    sendbuf = A[buff(size(A),+d)]
    recvbuf = zeros(length(sendbuf))
    MPI.Recv!(recvbuf, neighbors[2,d], 0, comm)
    MPI.Send(sendbuf,  neighbors[2,d], 1, comm)
    A[halos(size(A),+d)] .= reshape(recvbuf,size(halos(size(A),+d)))
end


function update_halo_asynch!(d, A, neighbors, comm)
    reqs=MPI.Request[]
    # get data to transfer
    send1 = A[buff(size(A),-d)]; send2 = A[buff(size(A),+d)]
    recv1 = zero(send1);         recv2 = zero(send2)
    # Send to / receive from neighbor 1 in dimension d
    push!(reqs,MPI.Isend(send1,  neighbors[1,d], 0, comm))
    push!(reqs,MPI.Irecv!(recv1, neighbors[1,d], 1, comm))
    # Send to / receive from neighbor 2 in dimension d
    push!(reqs,MPI.Irecv!(recv2, neighbors[2,d], 0, comm))
    push!(reqs,MPI.Isend(send2,  neighbors[2,d], 1, comm))
    # wair for all transfer to be done
    MPI.Waitall!(reqs)
    # put back in place
    A[halos(size(A),-d)] .= recv1 #reshape(recv1,size(halos(size(A),-d)))
    A[halos(size(A),+d)] .= recv2 #reshape(recv2,size(halos(size(A),+d)))
end

function diffusion_2D_mpi()
    # MPI
    MPI.Init()
    dims   = [0, 0, 0]
    comm   = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)
    periods = [0, 0, 0]
    disp::Integer=1
    reorder::Bool=true

    MPI.Dims_create!(nprocs, dims)
    comm_cart = MPI.Cart_create(comm, dims, periods, reorder)
    me     = MPI.Comm_rank(comm_cart)
    coords = MPI.Cart_coords(comm_cart)
    # make the cart comm
    neighbors = fill(MPI.PROC_NULL, NNEIGHBORS_PER_DIM, NDIMS_MPI);
    for i = 1:NDIMS_MPI
        neighbors[:,i] .= MPI.Cart_shift(comm_cart, i-1, disp);
    end
    (me == 0) && println("nprocs=$(nprocs), dims[1]=$(dims[1]), dims[2]=$(dims[2])")

    # Physics
    lx, ly = 10.0, 10.0
    D      = 1.0
    nt     = 100
    # Numerics
    nx, ny = 16 ,8                             # local number of grid points
    nx_g, ny_g = dims[1] * (nx - 2) + 2, dims[2] * (ny - 2) + 2 # global number of grid points

    # Derived numerics
    dx, dy = lx / nx_g, ly / ny_g                   # global
    dt     = min(dx, dy)^2 / D / 4.1
    # Array allocation
    qx     = zeros(nx - 1, ny - 2)
    qy     = zeros(nx - 2, ny - 1)
    # Initial condition
    x0, y0 = coords[1] * (nx - 2) * dx, coords[2] * (ny - 2) * dy
    xc     = [x0 + ix * dx - dx / 2 - 0.5 * lx for ix = 1:nx]
    yc     = [y0 + iy * dy - dy / 2 - 0.5 * ly for iy = 1:ny]
    C      = exp.(.-xc .^ 2 .- yc' .^ 2)
    save("diffusion_init_$me.jld2","C", C)

    # Time loop
    for it = 1:nt
        qx .= .-D * diff(C[:, 2:end-1], dims=1) / dx
        qy .= .-D * diff(C[2:end-1, :], dims=2) / dy
        C[2:end-1, 2:end-1] .= C[2:end-1, 2:end-1] .- dt * (diff(qx, dims=1) / dx .+ diff(qy, dims=2) / dy)
        for d in 1:2
            # update_halo!(d, C, neighbors, comm_cart)
            update_halo_asynch!(d, C, neighbors, comm_cart)
        end
    end

    # Save to visualise
    save("diffusion_$me.jld2","C", C)
    MPI.Finalize()
    return
end

diffusion_2D_mpi()
