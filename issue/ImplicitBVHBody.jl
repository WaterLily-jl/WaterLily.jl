using WaterLily,StaticArrays
using GeometryBasics,CUDA,WriteVTK
using Plots; gr()
using ImplicitBVH
using ImplicitBVH: BBox, BSphere

struct BVHBody{M,B,F<:Function} <: AbstractBody
    mesh::M
    bvh::B
    map::F
end
BVHBody(mesh::M,bvh::B;map=(x,t)->x) where {M,B} = BVHBody{M,B,typeof(map)}(mesh,bvh,map)
using Adapt
# make it GPU compatible
Adapt.@adapt_structure BVHBody

# Relative squared-distance from bounding volumes
using ImplicitBVH: BBox, BSphere, BoundingVolume
@fastmath @inline dist(x,bb::BSphere) = sum(abs2,x .- bb.x) - bb.r
@fastmath @inline function dist(x,bb::BBox)
    c = (bb.up .+ bb.lo) ./ 2
    r = (bb.up .- bb.lo) ./ 2
    q = abs.(x .- c) .- r
    sum(abs2,max.(q,0))
end
@fastmath @inline dist(x,bb::BoundingVolume) = dist(x,bb.volume)
@fastmath @inline inside(x::SVector,b::BoundingVolume) = inside(x,b.volume)
@fastmath @inline inside(x::SVector,b::BBox) = all(b.lo.-4 .≤ x) && all(x .≤ b.up.+4)
@fastmath @inline inside(x::SVector,b::BSphere) = sum(abs2,x .- b.x) < b.r^2

@fastmath @inline sibling(I::Int) = I%2==0 ? I+1 : I-1
@fastmath @inline parent(I::Int) = fld(I,2)
@fastmath @inline center(tri::SMatrix) = SVector(sum(tri,dims=2)/3)
@fastmath @inline d²_fast(x::SVector,tri::SMatrix) = sum(abs2,x-center(tri))
# @fastmath @inline d²_fast(x::SVector,tri::SMatrix) = sum(abs2,x-SVector(locate(x,tri))) # accurate method

@inline function closest(x::SVector,mesh)
    u=Int32(1); a=b=d²_fast(x,mesh[1]) # fast method
    for I in 2:length(mesh)
        b = d²_fast(x,mesh[I])
        b<a && (a=b; u=I) # Replace current best
    end
    # compute the normal and distance
    n,p = normal(mesh[u]),SVector(locate(x,mesh[u]))
    # signed Euclidian distance
    s = x-p; d = sign(sum(s.*n))*√sum(abs2,s)
    return d,n
end

# traverse the BVH
using ImplicitBVH: memory_index,unsafe_isvirtual
@inline function closest(x::SVector{D,T},bvh::ImplicitBVH.BVH,mesh;verbose=false) where {D,T}
    tree = bvh.tree; length_nodes = length(bvh.nodes)
    u=Int32(1);a=d²_fast(x,mesh[1]) # initial guess
    # Depth-First-Search
    i = 2; for _ in 1:tree.levels^(2+1) # prevent infinite loops
        @inbounds j = memory_index(tree,i)
        if j ≤ length_nodes
            inside(x,bvh.nodes[j]) && (i = 2i; continue) # go deeper
        else # use the leaf value
            @inbounds j = bvh.leaves[j-length_nodes].index # correct index in mesh
            d=d²_fast(x,mesh[j])
            d<a && (a=d; u=Int32(j))  # Replace current best
        end
        i = i>>trailing_ones(i)+1
        (i==1 || unsafe_isvirtual(tree, i)) && break # search complete!
    end
    # compute the normal and distance
    n,p = normal(mesh[u]),SVector(locate(x,mesh[u]))
    # signed Euclidian distance
    s = x-p; d = sign(sum(s.*n))*√sum(abs2,s)
    return d,n
end

using Printf
get_points = Base.get_extension(WaterLily, :WaterLilyGeometryBasicsExt).get_points
function WaterLily.save!(w,a::BVHBody,t=w.count[1]) #where S<:AbstractSimulation{A,B,C,D,MeshBody}
    k = w.count[1]
    points = get_points(a.mesh)
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, TriangleFace{Int}(3i+1,3i+2,3i+3)) for i in 0:length(a.mesh)-1]
    vtk = vtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), points, cells)
    for (name,func) in w.output_attrib
        # point/vector data must be oriented in the same way as the mesh
        vtk[name] = ndims(func(a))==1 ? func(a) : permutedims(func(a))
    end
    vtk_save(vtk); w.count[1]=k+1
    w.collection[round(t,digits=4)]=vtk
end

# signed distance function
WaterLily.sdf(body::BVHBody,x,t;kwargs...) = measure(body,x,t;kwargs...)[1]

using ForwardDiff
# measure function for the mesh body

using LinearAlgebra: cross
@fastmath @inline normal(tri::SMatrix) = hat(SVector(cross(tri[:,2]-tri[:,1],tri[:,3]-tri[:,1])))
@fastmath @inline hat(v) = v/(√(v'*v)+eps(eltype(v)))
function locate(x::SVector{T},tri::SMatrix{T}) where T
    # unpack the triangle vertices
    a,b,c = tri[:,1],tri[:,2],tri[:,3]
    ab,ac,ap = b.-a,c.-a,x.-a
    d1,d2 = sum(ab.*ap),sum(ac.*ap)
    # is point `a` closest?
    ((d1 ≤ 0) && (d2 ≤ 0)) && (return a)
    # is point `b` closest?
    bp = x.-b
    d3,d4 = sum(ab.*bp),sum(ac.*bp)
    ((d3 ≥ 0) && (d4 ≤ d3)) && (return b)
    # is point `c` closest?
    cp = x.-c
    d5,d6 = sum(ab.*cp),sum(ac.*cp)
    ((d6 ≥ 0) && (d5 ≤ d6)) && (return c)
    # is segment 'ab' closest?
    vc = d1*d4 - d3*d2
    ((vc ≤ 0) && (d1 ≥ 0) && (d3 ≤ 0)) && (return a .+ ab.*d1 ./ (d1 - d3))
    #  is segment 'ac' closest?
    vb = d5*d2 - d1*d6
    ((vb ≤ 0) && (d2 ≥ 0) && (d6 ≤ 0)) && (return a .+ ac.*d2 ./ (d2 - d6))
    # is segment 'bc' closest?
    va = d3*d6 - d5*d4
    ((va ≤ 0) && (d4 ≥ d3) && (d5 ≥ d6)) && (return b .+ (c .- b) .* (d4 - d3) ./ ((d4 - d3) + (d5 - d6)))
    # closest is interior to `abc`
    denom = one(T) / (va + vb + vc)
    v,w= vb*denom,vc*denom
    return a .+ ab .* v .+ ac .* w
end

function WaterLily.measure(body::BVHBody,x::SVector{D,T},t;kwargs...) where {D,T}
    # # map to correct location
    ξ = body.map(x,t)
    # before we try the bvh
    !inside(ξ,body.bvh.nodes[1]) && return (T(8),zero(SVector{D,T}),zero(SVector{D,T}))
    # # locate the point on the mesh
    d,n = closest(ξ,body.bvh,body.mesh)
    # velocity at the mesh point
    dξdx = ForwardDiff.jacobian(x->body.map(x,t), ξ)
    dξdt = -ForwardDiff.derivative(t->body.map(x,t), t)
    return (d,dξdx\n,dξdx\dξdt)
end

# dims
L = 128
MEMORY = Array

# read in the mesh
using FileIO, MeshIO
T = Float32
scale = L/2
mesh_0 = load("/home/marin/Workspace/WaterLilyPreCICE/meshes/sphere.stl")
# mesh_0 = load("/home/marin/Workspace/WaterLilyPreCICE/meshes/cube.stl")
# scale and map the points to the correct location
points = Point{3,T}[]
X_ = 4 .*rand(3)
for pnt in mesh_0.position
    push!(points,  Point{3,T}(SA{T}[pnt.data...]*T(scale)).+L/2.f0 .+ X_)
end
# make the scaled mesh
mesh = GeometryBasics.Mesh(points,GeometryBasics.faces(mesh_0))
# Generate bounding spheres around each triangle in the mesh
Primitives = BSphere
bounding_boxes = [Primitives{T}(tri) for tri in mesh] |> MEMORY;

# make a bvh
bvh = BVH(bounding_boxes, Primitives{T})

# surrogate of the mesh
cu_mesh = [hcat(vcat([mesh[i]...])...) for i in 1:length(mesh)] |> MEMORY
body = BVHBody(cu_mesh,bvh)

# test closest
x  = SA{T}[100,100,100]
# d,n = closest(x,body.bvh,body.mesh;verbose=true)

# d,n = closest(x,body.mesh)


# # test
sim = Simulation((L,L,L),(1,0,0),L;body,mem=MEMORY,T);
@time measure!(sim)
flood(sim.flow.σ[:,:,size(sim.flow.σ,3)÷2-1])

# make a writer with some attributes to output to the file
vtk_velocity(a::AbstractSimulation) = a.flow.u |> Array;
vtk_pressure(a::AbstractSimulation) = a.flow.p |> Array;
vtk_body(a::AbstractSimulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); a.flow.σ |> Array;)
vtk_mu0(a::AbstractSimulation) = a.flow.μ₀ |> Array;
vtk_normal(a::AbstractSimulation) = (WaterLily.@loop a.flow.f[I,:] .= measure(a.body,loc(0,I),0)[2] over I in CartesianIndices(a.flow.p);
                                     a.flow.f |> Array;)
custom_attrib = Dict("u"=>vtk_velocity, "p"=>vtk_pressure, "d"=>vtk_body, "μ₀"=>vtk_mu0, "n"=>vtk_normal)
vtu_normal(a) = [normal(tri) for tri in Array(a.mesh)]

# make the paraview writer
wr = vtkWriter("BVHBody";attrib=custom_attrib)
wr_mesh = vtkWriter("BVHBody_mesh";attrib=Dict("n"=>vtu_normal))
@time save!(wr, sim);
@time save!(wr_mesh, sim.body)
close(wr)
close(wr_mesh)
# save!(sim.body.bvh, "bvh")
