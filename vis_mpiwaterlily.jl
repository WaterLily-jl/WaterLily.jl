using FileIO,JLD2,Plots
using WaterLily
d = 1

rank_0 = load("/home/marin/Workspace/WaterLily/waterlily_1_0.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_1 = load("/home/marin/Workspace/WaterLily/waterlily_1_1.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_2 = load("/home/marin/Workspace/WaterLily/waterlily_1_2.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_3 = load("/home/marin/Workspace/WaterLily/waterlily_1_3.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]

# @assert all(rank_0[:,end-1:end] .≈ rank_0[:,3:4])
# @assert all(rank_2[:,end-1:end] .≈ rank_3[:,3:4])


C = vcat(hcat(rank_0,rank_1),hcat(rank_2,rank_3))

p1 = contourf(C', cmap=:imola10, aspect_ratio=:equal, lw=0, levels=3)

rank_1_1 = load("/home/marin/Workspace/WaterLily/waterlily_2_1.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_2_1 = load("/home/marin/Workspace/WaterLily/waterlily_2_2.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_3_1 = load("/home/marin/Workspace/WaterLily/waterlily_2_3.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_0_1 = load("/home/marin/Workspace/WaterLily/waterlily_2_0.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]

C = vcat(hcat(rank_0_1,rank_1_1),hcat(rank_2_1,rank_3_1))

# p2 = contourf(C[inside_u(size(C),1)]', cmap=:imola10, aspect_ratio=:equal, levels=3)
p2 = contourf(C', cmap=:imola10, aspect_ratio=:equal, lw=0, levels=3)


rank_1_2 = load("/home/marin/Workspace/WaterLily/waterlily_3_1.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_2_2 = load("/home/marin/Workspace/WaterLily/waterlily_3_2.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_3_2 = load("/home/marin/Workspace/WaterLily/waterlily_3_3.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_0_2 = load("/home/marin/Workspace/WaterLily/waterlily_3_0.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]

C = vcat(hcat(rank_0_2,rank_1_2),hcat(rank_2_2,rank_3_2))

p3 = contourf(C', cmap=:imola10, aspect_ratio=:equal, lw=0, levels=3)


rank_1_3 = load("/home/marin/Workspace/WaterLily/waterlily_4_1.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_2_3 = load("/home/marin/Workspace/WaterLily/waterlily_4_2.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_3_3 = load("/home/marin/Workspace/WaterLily/waterlily_4_3.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]
rank_0_3 = load("/home/marin/Workspace/WaterLily/waterlily_4_0.jld2")["sdf"][:,:,d]#[4:end-2,4:end-2]

C = vcat(hcat(rank_0_3,rank_1_3),hcat(rank_2_3,rank_3_3))

p4 = contourf(C', cmap=:imola10, aspect_ratio=:equal, lw=0, levels=10)


plot(p1,p2,p3,p4,layout = @layout [a b; c d])
# plot(p1,p2,layout = @layout [a b])




# OLD FUNCTIONS
# @views function update_halo!(A, neighbors, comm)
#     # Send to / receive from neighbor 1 in dimension x ("left neighbor")
#     if neighbors[1,1] != MPI.PROC_NULL
#         sendbuf = A[2,:]
#         recvbuf = zeros(length(sendbuf))
#         MPI.Send(sendbuf,  neighbors[1,1], 0, comm)
#         MPI.Recv!(recvbuf, neighbors[1,1], 1, comm)
#         A[1,:] .= recvbuf
#     end
#     # Send to / receive from neighbor 2 in dimension x ("right neighbor")
#     if neighbors[2,1] != MPI.PROC_NULL
#         sendbuf = A[end-1,:]
#         recvbuf = zeros(length(sendbuf))
#         MPI.Recv!(recvbuf, neighbors[2,1], 0, comm)
#         MPI.Send(sendbuf,  neighbors[2,1], 1, comm)
#         A[end,:] .= recvbuf
#     end
#     # Send to / receive from neighbor 1 in dimension y ("bottom neighbor")
#     if neighbors[1,2] != MPI.PROC_NULL
#         sendbuf = A[:,2]
#         recvbuf = zeros(length(sendbuf))
#         MPI.Send(sendbuf,  neighbors[1,2], 0, comm)
#         MPI.Recv!(recvbuf, neighbors[1,2], 1, comm)
#         A[:,1] .= recvbuf
#     end
#     # Send to / receive from neighbor 2 in dimension y ("top neighbor")
#     if neighbors[2,2] != MPI.PROC_NULL
#         sendbuf = A[:,end-1]
#         recvbuf = zeros(length(sendbuf))
#         MPI.Recv!(recvbuf, neighbors[2,2], 0, comm)
#         MPI.Send(sendbuf,  neighbors[2,2], 1, comm)
#         A[:,end] .= recvbuf
#     end
#     return
# end


# function update_halo!(d, A, neighbors, comm)
#     # Send to / receive from neighbor 1 in dimension d
#     # if neighbors[1,d] != MPI.PROC_NULL
#         # sendbuf = d==1 ? A[2,:] : A[:,2]
#         sendbuf = A[buff(size(A),-d)]
#         recvbuf = zeros(length(sendbuf))
#         MPI.Send(sendbuf,  neighbors[1,d], 0, comm)
#         MPI.Recv!(recvbuf, neighbors[1,d], 1, comm)
#         # d==1 ? A[1,:] .= recvbuf : A[:,1] .= recvbuf
#         A[halos(size(A),-d)] .= reshape(recvbuf,size(halos(size(A),-d)))
#     # end
#     # Send to / receive from neighbor 2 in dimension d
#     # if neighbors[2,d] != MPI.PROC_NULL
#         # sendbuf = d==1 ? A[end-1,:] : A[:,end-1]
#         sendbuf = A[buff(size(A),+d)]
#         recvbuf = zeros(length(sendbuf))
#         MPI.Recv!(recvbuf, neighbors[2,d], 0, comm)
#         MPI.Send(sendbuf,  neighbors[2,d], 1, comm)
#         # d==1 ? A[end,:] .= recvbuf : A[:,end] .= recvbuf
#         A[halos(size(A),+d)] .= reshape(recvbuf,size(halos(size(A),+d)))
#     # end
# end

# function update_halo!(d, A, neighbors, comm)
#     reqs=MPI.Request[]
#     # get data to transfer
#     send1 = A[buff(size(A),-d)]; send2 = A[buff(size(A),+d)]
#     recv1 = zero(send1);         recv2 = zero(send2)
#     # Send to / receive from neighbor 1 in dimension d
#     push!(reqs,MPI.Isend(send1,  neighbors[1,d], 0, comm))
#     push!(reqs,MPI.Irecv!(recv1, neighbors[1,d], 1, comm))
#     # Send to / receive from neighbor 2 in dimension d
#     push!(reqs,MPI.Irecv!(recv2, neighbors[2,d], 0, comm))
#     push!(reqs,MPI.Isend(send2,  neighbors[2,d], 1, comm))
#     # wair for all transfer to be done
#     MPI.Waitall!(reqs)
#     # put back in place if the neightbor exists
#     (neighbors[1,d] != MPI.PROC_NULL) && (A[halos(size(A),-d)] .= recv1)
#     (neighbors[2,d] != MPI.PROC_NULL) && (A[halos(size(A),+d)] .= recv2)
# end
# struct MPIgrid{T}
#     comm :: MPI.COMM_WORLD
#     periods :: AbstractVector{}
#     me :: T
#     coords :: AbstractVector{T}
#     neighbors :: AbstractArray{T}
# end

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


# # Pototype for boundary conditions update
# function update_halo!(d, A, neighbors, comm)
#     # get data to transfer
#     send1 = A[buff(size(A),-d)]; send2 = A[buff(size(A),+d)]
#     recv1 = zero(send1);         recv2 = zero(send2)
#     # swap the array
#     mpi_swap!(send1,recv1,send2,recv2,neighbors[:,d],comm)
#     # put back in place if the neightbor exists
#     (neighbors[1,d] != MPI.PROC_NULL) && (A[halos(size(A),-d)] .= recv1)
#     (neighbors[2,d] != MPI.PROC_NULL) && (A[halos(size(A),+d)] .= recv2)
# end


# MPI
# MPI.Init()
# dims   = [0, 0, 0]
# comm   = MPI.COMM_WORLD
# nprocs = MPI.Comm_size(comm)
# periods = [0, 0, 0]
# disp::Integer=1
# reorder::Bool=true

# MPI.Dims_create!(nprocs, dims)
# comm_cart = MPI.Cart_create(comm, dims, periods, reorder)
# me     = MPI.Comm_rank(comm_cart)
# coords = MPI.Cart_coords(comm_cart)
# # make the cart comm
# neighbors = fill(MPI.PROC_NULL, NNEIGHBORS_PER_DIM, NDIMS_MPI);
# for i = 1:NDIMS_MPI
#     neighbors[:,i] .= MPI.Cart_shift(comm_cart, i-1, disp);
# end
# (me == 0) && println("nprocs=$(nprocs), dims[1]=$(dims[1]), dims[2]=$(dims[2])")
# println("I am rank $me, at coordinate $coords")
