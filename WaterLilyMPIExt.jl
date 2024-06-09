using MPI,WaterLily
using StaticArrays
using FileIO,JLD2

const NDIMS_MPI = 3           # Internally, we set the number of dimensions always to 3 for calls to MPI. This ensures a fixed size for MPI coords, neigbors, etc and in general a simple, easy to read code.
const NNEIGHBORS_PER_DIM = 2


"""return the CI of the halos must only point to halos, otherwise it messes-up 
the reconstruction"""
function halos(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (2:3) : (dims[i]-1:dims[i]) : (3:dims[i]-2), N))
end
# return the CI of the buff 
function buff(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (4:5) : (dims[i]-3:dims[i]-2) : (3:dims[i]-2), N))
end

# function update_halo!(d, A, neighbors, comm)
#     reqs=MPI.Request[]
#     # Send to / receive from neighbor 1 in dimension d
#     sendbuf = A[buff(size(A),-d)]
#     recvbuf = zeros(length(sendbuf))
#     push!(reqs,MPI.Isend(sendbuf,  neighbors[1,d], 0, comm))
#     push!(reqs,MPI.Irecv!(recvbuf, neighbors[1,d], 1, comm))
#     A[halos(size(A),-d)] .= reshape(recvbuf,size(halos(size(A),-d)))
#     # Send to / receive from neighbor 2 in dimension d
#     sendbuf = A[buff(size(A),+d)]
#     recvbuf = zeros(length(sendbuf))
#     push!(reqs,MPI.Irecv!(recvbuf, neighbors[2,d], 0, comm))
#     push!(reqs,MPI.Isend(sendbuf,  neighbors[2,d], 1, comm))
#     A[halos(size(A),+d)] .= reshape(recvbuf,size(halos(size(A),+d)))
#     MPI.Waitall!(reqs)
# end

function update_halo!(d, A, neighbors, comm)
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
    # put back in place if the neightbor exists
    (neighbors[1,d] != MPI.PROC_NULL) && (A[halos(size(A),-d)] .= recv1)
    (neighbors[2,d] != MPI.PROC_NULL) && (A[halos(size(A),+d)] .= recv2)
end

# global coordinate in grid space
_global() = SA[coords[1]*nx, coords[2]*ny]

# every process must redifine the loc to be global
@inline function WaterLily.loc(i,I::CartesianIndex{N},T=Float64) where N
    # global position in the communicator
    SVector{N,T}(_global() .+ I.I .- 2.5 .- 0.5 .* δ(i,I).I)
end

"""Flow around a circle"""
function circle(n,m,center,radius;Re=250,U=1)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body)
end

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
println("I am rank $me, at coordinate $coords")

nx = 2^4
ny = 2^4
center = SA[nx,ny]
sim = circle(nx,ny,center,nx/4)

(me ==0) && println("nx=$nx, ny=$ny")

# check global coordinates
xs = loc(0,CartesianIndex(3,3))
println("I am rank $me, at global coordinate $xs")

sim.flow.σ .= NaN
sim.flow.σ[inside(sim.flow.σ)] .= me

# check that the measure uses the correct loc function
# measure_sdf!(sim.flow.σ,sim.body,0.0)
# save("waterlily_$me.jld2", "sdf", sim.flow.σ)

# updating the halos should not do anything
save("waterlily_$me.jld2", "sdf", sim.flow.σ)
for d in 1:2
    update_halo!(d, sim.flow.σ, neighbors, comm)
    # @show tmp[1:8,1:8]
    # sim.flow.σ .= tmp
    # sim.flow.μ₀[halos(size(tmp),-d),d] .= 1.0
    # sim.flow.μ₀[halos(size(tmp),+d),d] .= 1.0
end
save("waterlily_haloupdate_$me.jld2","sdf",sim.flow.σ)

MPI.Finalize()
