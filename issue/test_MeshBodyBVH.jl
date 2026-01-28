using WaterLily,StaticArrays
using GeometryBasics,CUDA,WriteVTK

function main(; L=256, MEMORY=CuArray, T=Float32)
    map(x,t) = x .- L÷2
    # different meshes
    # body = MeshBody("/home/marin/Workspace/WaterLilyPreCICE/meshes/xyzrgb_dragon.obj";map,scale=L/215,boundary=true,mem=MEMORY,T)
    body = MeshBody("//home/marin/Workspace/WaterLilyPreCICE/meshes/sphere.stl";map,scale=L/2,boundary=true,mem=MEMORY,T)
    # body = MeshBody("/home/marin/Workspace/WaterLilyPreCICE/meshes/cube.stl";map,scale=L/2,boundary=true,mem=MEMORY,T)
    return Simulation((L,L,L),(1,0,0),L;body,mem=MEMORY,T);
end

function make_airfoil(; L=64, Re=1000, α=-π/20, U=1, MEMORY=Array, T=Float32)
    # angle of attack
    R = SA{T}[cos(α) -sin(α) 0; sin(α) cos(α) 0; 0 0 1]
    # make the body from the stl mesh
    body = MeshBody("/home/marin/Workspace/WaterLilyPreCICE/meshes/IWS_kessi/InflatableWing.inp";scale=1.11L,
                    map=(x,t)->R*(x.-SA[L/2,L,L/4]),mem=MEMORY,T)
    return Simulation((3L,2L,L÷2), (U,0,0), L; body, ν=U*L/Re, mem=MEMORY, T)
end

sim = main(L=128,MEMORY=Array,T=Float32)
# sim = make_airfoil(L=128,MEMORY=CuArray,T=Float32)
sim.flow.σ .= 0;
@time measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim.flow))

# make a writer with some attributes to output to the file
vtk_velocity(a::AbstractSimulation) = a.flow.u |> Array;
vtk_pressure(a::AbstractSimulation) = a.flow.p |> Array;
vtk_body(a::AbstractSimulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); a.flow.σ |> Array;)
vtk_mu0(a::AbstractSimulation) = a.flow.μ₀ |> Array;
vtk_normal(a::AbstractSimulation) = (WaterLily.@loop a.flow.f[I,:] .= measure(a.body,loc(0,I),0)[2] over I in CartesianIndices(a.flow.p);
                                     a.flow.f |> Array;)
# custom_attrib = Dict("u"=>vtk_velocity, "p"=>vtk_pressure, "d"=>vtk_body, "μ₀"=>vtk_mu0, "n"=>vtk_normal)
custom_attrib = Dict("d"=>vtk_body,"μ₀"=>vtk_mu0,"n"=>vtk_normal)
vtu_normal(a) = [1.0 for tri in Array(a.mesh)]

# make the paraview writer
wr = vtkWriter("BVHBody";attrib=custom_attrib)
wr_mesh = vtkWriter("BVHBody_mesh";attrib=Dict("n"=>vtu_normal))
@time save!(wr, sim);
@time save!(wr_mesh, sim.body)
close(wr)
close(wr_mesh)

# # traverse and find the closest leave
# @inline function closest(x::SVector{D,T},bvh::ImplicitBVH.BVH,mesh) where {D,T}
#     tree = bvh.tree; length_nodes = length(bvh.nodes)
#     u=Int32(0);a=d=d⁰=T(64) # initial guess @TODO sensitive to initial a
#     # Depth-First-Search
#     i=2; for _ in 1:4tree.levels^2 # prevent infinite loops
#         @inbounds j = memory_index(tree,i)
#         if j ≤ length_nodes # we are on a node
#             if inside(x, bvh.nodes[j]) # we are in this box
#                 i = 2i; continue # → go deeper
#             end # we are not inside this one, how far are we?
#             k = memory_index(tree,i+1)
#             unsafe_isvirtual(tree, k) && break
#             if inside(x, bvh.nodes[k])
#                 i = 2k; continue # go deeper in the neighbor
#             end
#             if dist(x, bvh.nodes[j]) < dist(x, bvh.nodes[k])
#                 i = 2i; continue
#             else
#                 i = 2k; continue
#             end
#         else # we reached a leaf
#             @inbounds j = bvh.leaves[j-length_nodes].index # correct index in mesh
#             d = d²_fast(x, mesh[j])
#             d<a && (a=d; u=Int32(j))  # Replace current best
#         end
#         i = i>>trailing_ones(i)+1 # go to sibling, or uncle etc.
#         (i==1 || unsafe_isvirtual(tree, i)) && break # search complete!
#     end
#     return u,a
# end