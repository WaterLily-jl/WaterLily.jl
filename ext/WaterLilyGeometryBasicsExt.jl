module WaterLilyGeometryBasicsExt

using WaterLily
import WaterLily: MeshBody, measure, sdf, save!
using FileIO, MeshIO
using GeometryBasics
using StaticArrays
using Adapt

struct Meshbody{T,A<:AbstractArray,S<:AbstractVector{T},F<:Function} <: AbstractBody
    mesh :: A
    origin :: S
    width :: S
    map   :: F
    scale :: T
    boundary :: Bool
    half_thk :: T
end
function MeshBody(mesh::AbstractArray,origin::SVector{3,T},width::SVector{3,T},map=(x,t)->x,
                  scale=1,boundary=true,half_thk=0) where T
    return Meshbody(mesh,origin,width,map,T(scale),boundary,T(half_thk))
end
# make it GPU compatible
Adapt.@adapt_structure Meshbody

"""
    MeshBody(file_name; map=(x,t)->x, boundary=true, half_thk=0, scale=1, mem=Array, T=Float32)

Create a `MeshBody` from a mesh file. The mesh file can be in `.inp` format or any other format supported by `MeshIO`.
The mesh is scaled and mapped to the correct location using the `map` function.
The `boundary` flag indicates if the mesh is a boundary or not, and `half_thk` is used to adjust the distance for non-boundary meshes.
The `scale` parameter is used to scale the mesh points, and `mem` specifies the memory type for the `Simulation`.
"""
function MeshBody(file_name::String;map=(x,t)->x,scale=1,boundary=true,half_thk=0,T=Float32,mem=Array)
    # read in the mesh
    mesh = endswith(file_name,".inp") ? load_inp(file_name) : load(file_name)
    # scale and map the points to the correct location
    points = Point3f[]
    for pnt in mesh.position
        push!(points, Point3f(SA{T}[pnt.data...]*T(scale)))
    end
    # make the scaled mesh
    mesh = GeometryBasics.Mesh(points,GeometryBasics.faces(mesh))
    return MeshBody(mesh;map,scale,boundary,half_thk,T,mem)
end
function MeshBody(mesh::Mesh;map=(x,t)->x,scale=1,boundary=true,half_thk=0,T=Float32,mem=Array)
    # bounding box
    box = Rect(mesh.position)
    origin,width = SVector{3,T}(box.origin...),SVector{3,T}(box.widths...)
    # device array of the mesh that we store
    mesh = [hcat(vcat([mesh[i]...])...) for i in 1:length(mesh)] |> mem
    # make the mesh and return
    MeshBody(mesh,origin.-max(4,2half_thk),width.+max(8,4half_thk),map,T(scale),boundary,T(half_thk))
end

# closest triangle index in the mesh
#TODO without the `@inline``, an out of bounds error is thrown on the GPU...
# this is interesting, also passing the T to the MeshBody constructor throws that error
@inline function closest(mesh,x::SVector{T};kwargs...) where T
    u=1; a=b=d²_fast(mesh[1],x) # fast method
    for I in 2:length(mesh)
        b = d²_fast(mesh[I],x)
        b<a && (a=b; u=I) # Replace current best
    end
    return u
end

# signed distance function
sdf(body::Meshbody,x,t;kwargs...) = measure(body,x,t;kwargs...)[1]

# measure function for the mesh body
using ForwardDiff
function measure(body::Meshbody,x::SVector{D,T},t;kwargs...) where {D,T}
    # map to correct location
    ξ = body.map(x,t)
    # we don't need to worry if the geom is a boundary or not
    outside(ξ,body.origin,body.width) && return (max(8,2body.half_thk),zeros(SVector{D,T}),zeros(SVector{D,T}))
    # locate the point on the mesh
    #TODO this is what we replace with the BVH
    u = closest(body.mesh,ξ;kwargs...)
    # compute the normal and distance
    n,p = normal(body.mesh[u]),SVector(locate(body.mesh[u],x))
    # signed Euclidian distance
    s = ξ-p; d = sign(sum(s.*n))*√sum(abs2,s)
    # velocity at the mesh point
    dξdx = ForwardDiff.jacobian(x->body.map(x,t), x)
    dξdt = -ForwardDiff.derivative(t->body.map(x,t), t)
    # if the mesh is not a boundary, we need to adjust the distance
    !body.boundary && (d = abs(d)-body.half_thk)
    return (d,dξdx\n/body.scale,dξdx\dξdt)
end

using LinearAlgebra: cross
@fastmath @inline outside(x::SVector,origin,width) = !(all(origin .≤ x) && all(x .≤ origin+width))
@fastmath @inline d²_fast(tri::SMatrix,x::SVector) = sum(abs2,x-SVector(sum(tri,dims=2)/3))
@fastmath @inline normal(tri::SMatrix) = hat(SVector(cross(tri[:,2]-tri[:,1],tri[:,3]-tri[:,1])))
@fastmath @inline hat(v) = v/(√(v'*v)+eps(eltype(v)))

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

function locate(tri::SMatrix{T},p::SVector{T}) where T
    # unpack the triangle vertices
    a,b,c = tri[:,1],tri[:,2],tri[:,3]
    ab = b.-a
    ac = c.-a
    ap = p.-a
    d1 = sum(ab.*ap)
    d2 = sum(ac.*ap)
    # is point `a` closest?
    if ((d1 ≤ 0) && (d2 ≤ 0))
        return a
    end
    # is point `b` closest?
    bp = p.-b
    d3 = sum(ab.*bp)
    d4 = sum(ac.*bp)
    if ((d3 ≥ 0) && (d4 ≤ d3))
        return b
    end
    # is point `c` closest?
    cp = p.-c
    d5 = sum(ab.*cp)
    d6 = sum(ac.*cp)
    if ((d6 ≥ 0) && (d5 ≤ d6))
        return c
    end
    # is segment 'ab' closest?
    vc = d1*d4 - d3*d2
    if ((vc ≤ 0) && (d1 ≥ 0) && (d3 ≤ 0))
        x =  a .+ ab.*d1 ./ (d1 - d3)
        return x
    end
    #  is segment 'ac' closest?
    vb = d5*d2 - d1*d6
    if ((vb ≤ 0) && (d2 ≥ 0) && (d6 ≤ 0))
        x =  a .+ ac.*d2 ./ (d2 - d6)
        return x
    end
    # is segment 'bc' closest?
    va = d3*d6 - d5*d4
    if ((va ≤ 0) && (d4 ≥ d3) && (d5 ≥ d6))
        x =  b .+ (c .- b) .* (d4 - d3) ./ ((d4 - d3) + (d5 - d6))
        return x
    end
    # closest is interior to `abc`
    denom = one(T) / (va + vb + vc)
    v= vb*denom
    w = vc*denom
    x = a .+ ab .* v .+ ac .* w
    return x
end

import WriteVTK: MeshCell, VTKCellTypes, vtk_grid, vtk_save
using Printf: @sprintf
# access the WaterLily writer to save the file
function save!(w,a::Meshbody,t=w.count[1]) #where S<:AbstractSimulation{A,B,C,D,MeshBody}
    k = w.count[1]
    # points = hcat([[p.data...] for p ∈ a.mesh.position]...)
    points = hcat(Array(a.mesh)...)
    # cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, Base.to_index.(face)) for face in faces(a.mesh)]
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