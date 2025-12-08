module WaterLilyGeometryBasicsExt

using WaterLily
import WaterLily: MeshBody, measure, sdf, save!
using FileIO, MeshIO
using GeometryBasics
using StaticArrays
using Adapt

# split a vector in two along it's longest dimension
split_w(width::SVector{N},j) where N = SA[ntuple(i -> i==j ? width[i]/2 : width[i], N)...]
function Base.split(O::SVector,R::SVector)
    # split the longest side
    w = split_w(R, argmax(R))
    return (O,w), (O+(R-w),w)
end

struct Meshbody{T,A<:AbstractArray,P<:AbstractVector,S<:AbstractVector{T},F<:Function} <: AbstractBody
    mesh :: A
    bvh :: P
    sub_mesh
    sub_mesh_id
    origin :: S
    width :: S
    map   :: F
    scale :: T
    boundary :: Bool
    half_thk :: T
    use_bvh :: Bool
end
function MeshBody(mesh::AbstractArray,bvh,sub_mesh,sub_mesh_id,origin::SVector{3,T},width::SVector{3,T},map=(x,t)->x,
                  scale=1,boundary=true,half_thk=0,use_bvh=true) where T
    return Meshbody(mesh,bvh,sub_mesh,sub_mesh_id,origin,width,map,T(scale),boundary,T(half_thk),use_bvh)
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
function MeshBody(file_name::String;map=(x,t)->x,scale=1,boundary=true,half_thk=0,level=5,T=Float32,mem=Array,use_bvh=true)
    # read in the mesh
    mesh = endswith(file_name,".inp") ? load_inp(file_name) : load(file_name)
    # scale and map the points to the correct location
    points = Point{3,T}[]
    for pnt in mesh.position
        push!(points,  Point{3,T}(SA{T}[pnt.data...]*T(scale)))
    end
    # make the scaled mesh
    mesh = GeometryBasics.Mesh(points,GeometryBasics.faces(mesh))
    return MeshBody(mesh;map,scale,boundary,half_thk,level,T,mem,use_bvh)
end
function MeshBody(mesh::Mesh;map=(x,t)->x,scale=1,boundary=true,half_thk=0,level=5,T=Float32,mem=Array,use_bvh=true)
    # bounding box
    box = Rect(mesh.position)
    origin,width = SVector{3,T}(box.origin...).-max(4,2half_thk),SVector{3,T}(box.widths...).+max(8,4half_thk)
    # device array of the mesh that we store
    mesh = [hcat(vcat([mesh[i]...])...) for i in 1:length(mesh)]
    bvh,sub_mesh,sub_mesh_id = make_bvh(mesh,origin,width,level,mem)
    mesh = mesh |> mem
    # make the mesh and return
    MeshBody(mesh,bvh,sub_mesh,sub_mesh_id,bvh[1]...,  # unpack the root box
            map,T(scale),boundary,T(half_thk),use_bvh)
end

#TODO the new mesh is organised with the points as the row of the SMatrix
# this means that point 1 is el[1,:], point 2 is el [2.:], etc.
function make_bvh(mesh,origin::SVector{3,T},width::SVector{3,T},lvl::Int,mem) where T
    # store the subdivisions in a binary tree fashion
    box_array = Vector{NTuple{2,SVector{3,T}}}(undef, 2^lvl-1)
    box_array[1]=(origin,width)
    # make the subdivisions
    for i in 1:2^(lvl-1)-1
        left,right = split(box_array[i]...)
        box_array[2i] = left
        box_array[2i+1] = right
    end
    # now for each leave, we make it tight to the mesh
    # WaterLily.@loop box_array[I] = subset(I,box_array,mesh) over I ∈ leafs(lvl)
    sub_mesh_id,sub_mesh = [],[]
    for I in leafs(lvl)
        S = split_fit!(I,box_array,mesh)
        push!(sub_mesh_id,S)
        push!(sub_mesh, mesh[S])
    end
    # we can now go back and add children together to make the true parent
    # WaterLily.@loop box_array[I] = merge(I,box_array) over I ∈ parents(lvl)
    for I in parents(lvl)
        box_array[I] = merge(I,box_array)
    end
    return box_array |> mem, sub_mesh |> mem, sub_mesh_id |> mem
end

# small helper functions
@fastmath @inline leafs(lvl) = CartesianIndices((2^(lvl-1):2^lvl-1,))
@fastmath @inline parents(lvl) = reverse(CartesianIndices((1:2^(lvl-1)-1,)))
@fastmath @inline children(I) = 2I:2I+oneunit(I)
@fastmath @inline sibling(I::Int) = I%2==0 ? I+1 : I-1
@fastmath @inline parent(I::Int) = fld(I,2)
@inline subset(I,boxes) = boxes[I]

# split the mesh to sub meshed and fit the box tight to it
function split_fit!(I,box_list,mesh,δ=4)
    (O,W) = box_list[I]
    T = eltype(O)
    sub_mesh = Int[] # populate with dummy
    vmin,vmax = O,O.+W
    for (i,el) in enumerate(mesh)
        # if the center oft he elementi is inside
        if all(O .≤ center(el) .≤ O+W)
            # remember it
            push!(sub_mesh,i)
            # new box, minimum of coordinates sets bounding box
            vmin = min.(minimum(el,dims=2), vmin)
            vmax = max.(maximum(el,dims=2), vmax)
        end
    end
    o = vmin .- T(δ)
    r = (vmax - vmin) .+ T(2δ) # make it a bit bigger
    box_list[I] = (o,r) # update the box
    return sub_mesh
end

# merge two child boxes into a parent box
@inline @fastmath function merge(I,bvh)
    left,right = bvh[children(I)]
    O = min.(left[1], right[1])
    W = max.(sum(left), sum(right)) - O
    return (O,W)
end

# closest triangle index in the mesh
#TODO without the `@inline``, an out of bounds error is thrown on the GPU...
# this is interesting, also passing the T to the MeshBody constructor throws that error
@inline function closest(x::SVector,mesh;kwargs...)
    u=1; a=b=d²_fast(x,mesh[1]) # fast method
    for I in 2:length(mesh)
        b = d²_fast(x,mesh[I])
        b<a && (a=b; u=I) # Replace current best
    end
    return u,a
end

# taverse a bvh to find the closest triangle and distance
@inline function traverse_fsm(x::SVector{D,T},bvh::AbstractVector,
                              sub_meshes::AbstractVector,sub_meshes_id)::Tuple{Int,T} where {D,T}
    # how deep we go and where we start
    N,state,current = fld(length(bvh),2),:fromParent,1
    u,a = (0, T(64)) # this one doesn't exist, and we use a square distance
    off = length(bvh) - length(sub_meshes)  # offset to access subsets
    while true
        if state==:fromChild
            if current ==1
                break
            elseif current==2parent(current)
                current = sibling(current)
                state = :fromSibling
            else
                current = parent(current)
                state = :fromChild
            end
        elseif state==:fromSibling
            hit = !outside(x,bvh[current]...)
            if !hit
                current = parent(current)
                state = :fromChild
            elseif current > N
                v,b = closest(x,subset(current-off,sub_meshes))
                abs(b)<abs(a) && (a=b; u=subset(current-off,sub_meshes_id)[v])
                current = parent(current)
                state = :fromChild
            else
                current = 2current
                state = :fromParent
            end
        elseif state==:fromParent
            hit = !outside(x,bvh[current]...)
            if !hit && current ==1
                break
            elseif !hit
                current = sibling(current)
                state = :fromSibling
            elseif current > N
                v,b = closest(x,subset(current-off,sub_meshes))
                abs(b)<abs(a) && (a=b; u=subset(current-off,sub_meshes_id)[v])
                current = sibling(current)
                state = :fromSibling
            else
                current = 2current
                state = :fromParent
            end
        end
    end
    return u,a
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
    u,a = body.use_bvh ? traverse_fsm(ξ,body.bvh,body.sub_mesh,body.sub_mesh_id) : closest(ξ,body.mesh;kwargs...)
    # check that we have found something
    u == 0 && return (max(8,2body.half_thk),zeros(SVector{D,T}),zeros(SVector{D,T}))
    # compute the normal and distance
    n,p = normal(body.mesh[u]),SVector(locate(ξ,body.mesh[u]))
    # signed Euclidian distance
    s = ξ-p; d = sign(sum(s.*n))*√sum(abs2,s)
    # velocity at the mesh point
    dξdx = ForwardDiff.jacobian(x->body.map(x,t), ξ)
    dξdt = -ForwardDiff.derivative(t->body.map(x,t), t)
    # if the mesh is not a boundary, we need to adjust the distance
    !body.boundary && (d = abs(d)-body.half_thk)
    return (d,dξdx\n/body.scale,dξdx\dξdt)
end

using LinearAlgebra: cross
@fastmath @inline outside(x::SVector,origin,width) = !(all(origin .≤ x) && all(x .≤ origin+width))
@fastmath @inline d²_fast(x::SVector,tri::SMatrix) = sum(abs2,x-center(tri))
@fastmath @inline normal(tri::SMatrix) = hat(SVector(cross(tri[:,2]-tri[:,1],tri[:,3]-tri[:,1])))
@fastmath @inline hat(v) = v/(√(v'*v)+eps(eltype(v)))
@fastmath @inline center(tri::SMatrix) = SVector(sum(tri,dims=2)/3)

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

function locate(x::SVector{T},tri::SMatrix{T}) where T
    # unpack the triangle vertices
    a,b,c = tri[:,1],tri[:,2],tri[:,3]
    ab = b.-a
    ac = c.-a
    ap = x.-a
    d1 = sum(ab.*ap)
    d2 = sum(ac.*ap)
    # is point `a` closest?
    if ((d1 ≤ 0) && (d2 ≤ 0))
        return a
    end
    # is point `b` closest?
    bp = x.-b
    d3 = sum(ab.*bp)
    d4 = sum(ac.*bp)
    if ((d3 ≥ 0) && (d4 ≤ d3))
        return b
    end
    # is point `c` closest?
    cp = x.-c
    d5 = sum(ab.*cp)
    d6 = sum(ac.*cp)
    if ((d6 ≥ 0) && (d5 ≤ d6))
        return c
    end
    # is segment 'ab' closest?
    vc = d1*d4 - d3*d2
    if ((vc ≤ 0) && (d1 ≥ 0) && (d3 ≤ 0))
        p =  a .+ ab.*d1 ./ (d1 - d3)
        return p
    end
    #  is segment 'ac' closest?
    vb = d5*d2 - d1*d6
    if ((vb ≤ 0) && (d2 ≥ 0) && (d6 ≤ 0))
        p =  a .+ ac.*d2 ./ (d2 - d6)
        return p
    end
    # is segment 'bc' closest?
    va = d3*d6 - d5*d4
    if ((va ≤ 0) && (d4 ≥ d3) && (d5 ≥ d6))
        p =  b .+ (c .- b) .* (d4 - d3) ./ ((d4 - d3) + (d5 - d6))
        return p
    end
    # closest is interior to `abc`
    denom = one(T) / (va + vb + vc)
    v= vb*denom
    w = vc*denom
    p = a .+ ab .* v .+ ac .* w
    return p
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

function expand_bvh_points(bvh)
    points = Float32[]
    for (C,W) in Array(bvh)
        o = C; u = C+W
        push!(points, o[1], o[2], o[3])
        push!(points, u[1], o[2], o[3])
        push!(points, o[1], u[2], o[3])
        push!(points, u[1], u[2], o[3])
        push!(points, o[1], o[2], u[3])
        push!(points, u[1], o[2], u[3])
        push!(points, o[1], u[2], u[3])
        push!(points, u[1], u[2], u[3])
    end
    return reshape(points, 3, length(points)÷3)
end

# save each levelof a b vhm esh to a vtk file
save!(bvh, fname="bvh") = for l in 1:Int(floor(log2(length(bvh)+1)))
    points = expand_bvh_points(bvh[2^(l-1):2^l-1])
    cells = [MeshCell(VTKCellTypes.VTK_VOXEL, collect(8i+1:8i+8)) for i in 0:length(bvh[2^(l-1):2^l-1])-1]
    vtk = vtk_grid(fname*"_$l", points, cells)
    vtk["ID"] = collect(2^(l-1):2^l-1)
    vtk_save(vtk)
end

end # module