module WaterLilyGeometryBasicsExt

using WaterLily
import WaterLily: AbstractBody, MeshBody, save!
using FileIO, MeshIO, StaticArrays
using GeometryBasics
using ImplicitBVH

# @TODO these two functions will live somewhere else
# read .inp files
function load_inp(fname; facetype=GLTriangleFace, pointtype=Point3f)
    #INP file format
    @assert endswith(fname,".inp") "file type not supported"
    fs = open(fname)

    points = pointtype[]
    faces = facetype[]
    node_idx = Int[]
    cnt = 0

    # read the first 3 lines if there is the "*heading" keyword
    line = readline(fs)
    contains(line,"*heading") && (line = readline(fs))
    BlockType = contains(line,"*NODE") ? Val{:NodeBlock}() : Val{:DataBlock}()

    # read the file
    while !eof(fs)
        line = readline(fs)
        contains(line,"*ELSET, ELSET=") && (cnt+=1)
        BlockType, line = parse_blocktype!(BlockType, fs, line)
        if BlockType == Val{:NodeBlock}()
            push!(node_idx, parse(Int,split(line,",")[1])) # keep track of the node index of the inp file
            push!(points, pointtype(parse.(eltype(pointtype),split(line,",")[2:4])))
        elseif BlockType == Val{:ElementBlock}()
            nodes = parse.(Int,split(line,",")[2:end])
            push!(faces, TriangleFace{Int}(facetype([findfirst(==(node),node_idx) for node in nodes])...)) # parse the face
        else
            continue
        end
    end
    close(fs) # close file stream
    return Mesh(points, faces)
end
function parse_blocktype!(block, io, line)
    contains(line,"*NODE") && return block=Val{:NodeBlock}(),readline(io)
    contains(line,"*ELEMENT") && return block=Val{:ElementBlock}(),readline(io)
    return block, line
end

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
    MeshBody(file_name; map=(x,t)->x, boundary=true, half_thk=0, scale=1, mem=Array, T=Float32)

Create a `MeshBody` from a mesh file. The mesh file can be in `.inp` format or any other format supported by `MeshIO`.
The mesh is scaled and mapped to the correct location using the `map` function.
The `boundary` flag indicates if the mesh is a boundary or not, and `half_thk` is used to adjust the distance for non-boundary meshes.
The `scale` parameter is used to scale the mesh points, and `mem` specifies the memory type for the `Simulation`.
"""
function MeshBody(file_name::String;map=(x,t)->x,scale=1,boundary=true,half_thk=0,T=Float32,mem=Array,primitive=ImplicitBVH.BBox)
    # read in the mesh
    mesh = endswith(file_name,".inp") ? load_inp(file_name) : load(file_name)
    # scale and map the points to the correct location
    points = Point{3,T}[]
    for pnt in mesh.position
        push!(points,  Point{3,T}(SA{T}[pnt.data...]*T(scale)))
    end
    # make the scaled mesh
    mesh = GeometryBasics.Mesh(points,GeometryBasics.faces(mesh))
    return MeshBody(mesh;map,scale,boundary,half_thk,T,mem,primitive)
end
function MeshBody(mesh::Mesh;map=(x,t)->x,scale=1,boundary=true,half_thk=0,T=Float32,mem=Array,primitive=ImplicitBVH.BBox)
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
@fastmath @inline inside(x::SVector, b::ImplicitBVH.BoundingVolume) = inside(x, b.volume)
@fastmath @inline inside(x::SVector, b::ImplicitBVH.BBox) = all(b.lo.-4 .≤ x) && all(x .≤ b.up.+4)
@fastmath @inline inside(x::SVector, b::ImplicitBVH.BSphere) = sum(abs2,x .- b.x) - b.r^2 ≤ 4

# compute the distance to primitive
dist(x, b::ImplicitBVH.BSphere) = sum(abs2,x .- b.x) - b.r
function dist(x, b::ImplicitBVH.BBox)
    c = (b.up .+ b.lo) ./ 2
    r = (b.up .- b.lo) ./ 2
    sum(abs2, max.(abs.(x .- c) .- r, 0))
end
dist(x, b::ImplicitBVH.BoundingVolume) = dist(x, b.volume)

@inline function closest(x::SVector,mesh)
    u=Int32(1); a=b=d²_fast(x, mesh[1]) # fast method
    for I in 2:length(mesh)
        b = d²_fast(x, mesh[I])
        b<a && (a=b; u=I) # Replace current best
    end
    return u,a
end

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
    u==Int32(0) && return (d⁰,zero(x),zero(x))
    # compute the normal and distance
    n,p = normal(body.mesh[u]),SVector(locate(ξ,body.mesh[u]))
    # signed Euclidian distance
    s = ξ-p; d = sign(sum(s.*n))*√sum(abs2,s)
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

function get_points(mesh)
    pts = zeros(Float32, 3, 3length(mesh))
    for (i,el) in enumerate(Array(mesh))
        pts[:,3i-2:3i] = el
    end
    return pts
end

end # module