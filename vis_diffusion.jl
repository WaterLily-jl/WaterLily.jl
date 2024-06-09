using FileIO,JLD2,Plots

rank_0 = load("/home/marin/Workspace/WaterLily/diffusion_init_0.jld2")["C"][2:end-1,2:end-1]
rank_1 = load("/home/marin/Workspace/WaterLily/diffusion_init_1.jld2")["C"][2:end-1,2:end-1]
rank_2 = load("/home/marin/Workspace/WaterLily/diffusion_init_2.jld2")["C"][2:end-1,2:end-1]
rank_3 = load("/home/marin/Workspace/WaterLily/diffusion_init_3.jld2")["C"][2:end-1,2:end-1]

C = vcat(hcat(rank_0,rank_1),hcat(rank_2,rank_3))

p1 = contourf(C', cmap=:imola10, aspect_ratio=:equal)

rank_0 = load("/home/marin/Workspace/WaterLily/diffusion_0.jld2")["C"][2:end-1,2:end-1]
rank_1 = load("/home/marin/Workspace/WaterLily/diffusion_1.jld2")["C"][2:end-1,2:end-1]
rank_2 = load("/home/marin/Workspace/WaterLily/diffusion_2.jld2")["C"][2:end-1,2:end-1]
rank_3 = load("/home/marin/Workspace/WaterLily/diffusion_3.jld2")["C"][2:end-1,2:end-1]

C = vcat(hcat(rank_0,rank_1),hcat(rank_2,rank_3))

p2 = contourf(C', cmap=:imola10, aspect_ratio=:equal)
plot(p1, p2, layout = @layout [a; b])


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