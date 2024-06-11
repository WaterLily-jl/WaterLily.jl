using MPI,WaterLily
using StaticArrays
using FileIO,JLD2

const NDIMS_MPI = 3           # Internally, we set the number of dimensions always to 3 for calls to MPI. This ensures a fixed size for MPI coords, neigbors, etc and in general a simple, easy to read code.
const NNEIGHBORS_PER_DIM = 2


"""return the CI of the halos must only point to halos, otherwise it messes-up 
the reconstruction"""
function halos(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (2:3) : (dims[i]-1:dims[i]) : (1:dims[i]), N))
end
# return the CI of the buff 
function buff(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (4:5) : (dims[i]-3:dims[i]-2) : (1:dims[i]), N))
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
    return
end


function BC!(a)
    N = size(a)
    for d ∈ eachindex(N)# this is require because scalar and vector field are located at different location
        Ii = δ(d,CartesianIndex(0,0))
        # get data to transfer
        send1 = a[buff(N,-d).-Ii]; send2 = a[buff(N,+d)]
        recv1 = zero(send1);   recv2 = zero(send2)
        # swap 
        mpi_swap!(send1,recv1,send2,recv2,neighbors[:,d],comm)

        # this sets the BCs
        if neighbors[1,d] != MPI.PROC_NULL # right wall
            a[halos(N,-d).-Ii] .= recv1
        else
            a[halos(N,-d).-Ii] .= reverse(send1; dims=d)
        end
        if neighbors[2,d] != MPI.PROC_NULL # right wall
            a[halos(N,+d)] .= recv2
        else
            a[halos(N,+d)] .= reverse(send2; dims=d)
        end
    end
end

function BC!(a,A,neighbors,comm,saveexit=false)
    N,n = WaterLily.size_u(a)
    for i ∈ 1:n, j ∈ 1:n
        # get data to transfer
        send1 = a[buff(N,-j),i]; send2 = a[buff(N,+j),i]
        recv1 = zero(send1);     recv2 = zero(send2)
        # swap 
        mpi_swap!(send1,recv1,send2,recv2,neighbors[:,j],comm)

        # domain boundaries
        if neighbors[1,j]==MPI.PROC_NULL # left wall
            if i==j # set flux
                a[halos(N,-j),i] .= A[i]
            else # zero gradient
                a[halos(N,-j),i] .= reverse(send1; dims=j)
            end
        else
            a[halos(N,-j),i] .= recv1
        end
        if neighbors[2,j]==MPI.PROC_NULL # right wall
            if i==j #&& (!saveexit || i>1) # convection exit
                a[halos(N,+j),i] .= A[i]
            else # zero gradient
                a[halos(N,+j),i] .= reverse(send2; dims=j)
            end
        else
            a[halos(N,+j),i] .= recv2
        end
        # this sets the BCs
        # (neighbors[1,j] != MPI.PROC_NULL) && (a[halos(N,-j),i] .= recv1)
        # (neighbors[2,j] != MPI.PROC_NULL) && (a[halos(N,+j),i] .= recv2)
    end
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

# first we chack s imple rank matrix
sim.flow.σ .= NaN
sim.flow.σ[inside(sim.flow.σ)] .= me

# check that the measure uses the correct loc function
# measure_sdf!(sim.flow.σ,sim.body,0.0)
# save("waterlily_$me.jld2", "sdf", sim.flow.σ)

# second check is to check the μ₀
sim.flow.σ .= sim.flow.μ₀[:,:,1]

# updating the halos should not do anything
save("waterlily_$me.jld2", "sdf", sim.flow.σ)

BC!(sim.flow.μ₀, zeros(SVector{2,Float64}), neighbors, comm)

BC!(sim.flow.σ)

sim.flow.σ .= sim.flow.μ₀[:,:,1]

save("waterlily_haloupdate_$me.jld2","sdf",sim.flow.σ)

MPI.Finalize()
