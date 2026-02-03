module WaterLilyGeometryBasicsExt

using WaterLily
import WaterLily: AbstractBody, MeshBody, save!
using FileIO, MeshIO, StaticArrays
using ImplicitBVH, GeometryBasics

struct Meshbody{T,M,B,F<:Function} <: AbstractBody
    mesh::M
    bvh::B
    map::F
    scale::T
    boundary::Bool
    half_thk::T
end
MeshBody(mesh::M,bvh::B;map=(x,t)->x,scale::T=1,boundary=false,half_thk::T=0) where {T,M,B} = Meshbody{T,M,B,typeof(map)}(mesh,bvh,map,scale,boundary,half_thk)
using Adapt
# make it GPU compatible
Adapt.@adapt_structure Meshbody

"""
    MeshBody(mesh::Union{Mesh, String};
             map::Function=(x,t)->x, boundary::Bool=true, half_thk::T=0.f0,
             scale::T=1.f0, mem=Array, primitives::Union{BBox, BSphere}) where T

Constructor for a MeshBody:

  - `mesh::Union{Mesh, String}`: the GeometryBasics.Mesh or path to the mesh file to use to define the geometry.
  - `map(x::AbstractVector,t::Real)::AbstractVector`: coordinate mapping function.
  - `boundary::Bool`: whether the mesh is a boundary or not.
  - `half_thk::T`: half thickness to apply if the mesh is not a boundary, the type defines the base type of the MeshBody, default is Float32.
  - `scale::T`: scale factor to apply to the mesh points, the type defines the base type of the MeshBody, default is Float32.
  - `mem`: memory location. `Array`, `CuArray`, `ROCm` to run on CPU, NVIDIA, or AMD devices, respectively.
  - `primitive::Union{BBox, BSphere}`: bounding volume primitive to use in the ImplicitBVH. Default is Axis-Aligned Bounding Box.

"""
function MeshBody(file_name::String;map=(x,t)->x,scale::T=1.f0,boundary=true,half_thk=0,mem=Array,primitive=ImplicitBVH.BBox) where T
    # read in the mesh
    mesh = load(file_name)
    # scale and map the points to the correct location
    points = Point{3,T}[]
    for pnt in mesh.position
        push!(points,  Point{3,T}(SA{T}[pnt.data...]*T(scale)))
    end
    # make the scaled mesh
    mesh = GeometryBasics.Mesh(points,GeometryBasics.faces(mesh))
    return MeshBody(mesh;map,scale,boundary,half_thk,mem,primitive)
end
function MeshBody(mesh::Mesh;map=(x,t)->x,scale::T=1.f0,boundary=true,half_thk=0,mem=Array,primitive=ImplicitBVH.BBox) where T
    # make the BVH
    bounding_boxes = [primitive{T}(el) for el in mesh] |> mem;
    bvh = BVH(bounding_boxes, primitive{T})
    # device array of the mesh that we store
    mesh = [hcat(vcat([mesh[i]...])...) for i in 1:length(mesh)] |> mem
    # make the mesh and return
    MeshBody(mesh,bvh;map,scale=T(scale),boundary=boundary,half_thk=T(half_thk))
end

using LinearAlgebra: cross
# @fastmath @inline d²_fast(x::SVector,tri::SMatrix) = sum(abs2,x-center(tri))
@fastmath @inline d²_fast(x::SVector,tri::SMatrix) = sum(abs2,x-locate(x,tri))
@fastmath @inline normal(tri::SMatrix) = hat(SVector(cross(tri[:,2]-tri[:,1],tri[:,3]-tri[:,1])))
@fastmath @inline hat(v) = v/(√(v'*v)+eps(eltype(v)))
@fastmath @inline center(tri::SMatrix) = SVector(sum(tri,dims=2)/3)

import ImplicitBVH: BoundingVolume,BBox,BSphere
@fastmath @inline inside(x::SVector, b::BoundingVolume) = inside(x, b.volume)
@fastmath @inline inside(x::SVector, b::BBox) = all(b.lo.-4 .≤ x) && all(x .≤ b.up.+4)
@fastmath @inline inside(x::SVector, b::BSphere) = sum(abs2,x .- b.x) - b.r^2 ≤ 4

# traverse the BVH
import ImplicitBVH: memory_index,unsafe_isvirtual
@inline function closest(x::SVector{D,T},bvh::ImplicitBVH.BVH,mesh) where {D,T}
    tree = bvh.tree; length_nodes = length(bvh.nodes)
    u=Int32(0);a=d=T(64) # initial guess @TODO sensitive to initial a
    # Depth-First-Search
    i=2; for _ in 1:4tree.levels^2 # prevent infinite loops
        @inbounds j = memory_index(tree,i)
        if j ≤ length_nodes # we are on a node
            inside(x, bvh.nodes[j]) && (i = 2i; continue) # go deeper
        else # we reached a leaf
            @inbounds j = bvh.leaves[j-length_nodes].index # correct index in mesh
            d = d²_fast(x, mesh[j])
            d<a && (a=d; u=Int32(j))  # Replace current best
        end
        i = i>>trailing_ones(i)+1 # go to sibling, or uncle etc.
        (i==1 || unsafe_isvirtual(tree, i)) && break # search complete!
    end
    return u,a
end

# locate the closest point p to x on triangle tri
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

# signed distance function
WaterLily.sdf(body::Meshbody,x,t;kwargs...) = measure(body,x,t;kwargs...)[1]

using ForwardDiff
# measure
function WaterLily.measure(body::Meshbody,x::SVector{D,T},t;fastd²=Inf) where {D,T}
    # map to correct location
    ξ = body.map(x,t)
    # before we try the bvh
    !inside(ξ,body.bvh.nodes[1]) && return (T(8),zero(x),zero(x))
    # locate the point on the mesh
    u,d⁰ = closest(ξ,body.bvh,body.mesh)
    u==0 && return (T(8),zero(x),zero(x)) # no closest found
    # compute the normal and distance
    n,p = normal(body.mesh[u]),SVector(locate(ξ,body.mesh[u]))
    # signed Euclidian distance
    s = ξ-p; d = sign(sum(s.*n))*√sum(abs2,s)
    !body.boundary && (d = abs(d)-body.half_thk) # if the mesh is not a boundary, we need to adjust the distance
    d^2>fastd² && return (d,zero(x),zero(x)) # skip n,V
    # velocity at the mesh point
    dξdx = ForwardDiff.jacobian(x->body.map(x,t), ξ)
    dξdt = -ForwardDiff.derivative(t->body.map(x,t), t)
    return (d,dξdx\n,dξdx\dξdt)
end

import WriteVTK: MeshCell, VTKCellTypes, vtk_grid, vtk_save
using Printf: @sprintf
# access the WaterLily writer to save the file
function save!(w,a::Meshbody,t=w.count[1]) #where S<:AbstractSimulation{A,B,C,D,MeshBody}
    k = w.count[1]
    points = zeros(Float32, 3, 3length(a.mesh))
    for (i,el) in enumerate(Array(a.mesh))
        points[:,3i-2:3i] = el
    end
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, TriangleFace{Int}(3i+1,3i+2,3i+3)) for i in 0:length(a.mesh)-1]
    vtk = vtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), points, cells)
    for (name,func) in w.output_attrib
        # point/vector data must be oriented in the same way as the mesh
        vtk[name] = ndims(func(a))==1 ? func(a) : permutedims(func(a))
    end
    vtk_save(vtk); w.count[1]=k+1
    w.collection[round(t,digits=4)]=vtk
end

end # module