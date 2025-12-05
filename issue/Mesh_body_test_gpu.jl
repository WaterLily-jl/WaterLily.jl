using WaterLily,StaticArrays
using GeometryBasics,CUDA,WriteVTK
using Plots; gr()


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


function make_cube(L;U=1,mem=CuArray,T=Float32,level=3)
    α = π/10.f0 # rotation angle
    function map(x,t)
        Rx = SA[1 0 0; 0 cos(α) -sin(α); 0 sin(α) cos(α)]
        Ry = SA[cos(α) 0 sin(α); 0 1 0; -sin(α) 0 cos(α)]
        Rz = SA[cos(α) -sin(α) 0; sin(α) cos(α) 0; 0 0 1]
        Rx*Ry*Rz*(x.-L/2.f0).+0.5f0
    end
    # make body
    body = MeshBody("/home/marin/Workspace/WaterLilyPreCICE/meshes/sphere.stl";
                    scale=L/2,level=level,map,mem)
    # make sim
    Simulation((L,L,L),(U,0,0),L;body,mem,T)
end
function make_aorta(L=32;Re=250,U=1,T=Float32,mem=Array)
    # make the body from the stl mesh
    body = MeshBody("/home/marin/Workspace/WaterLilyPreCICE/meshes/aorta/aorta.inp";
                    scale=L/2, map=(x,t)->x-SA[L/2.f0,L/2.f0,L/4.f0],
                    boundary=false,half_thk=1.f0,mem)
    # generate sim
    Simulation((L,L,L÷2), (0,0,0), L; ν=U*L/Re, body, mem, T)
end
function make_GeomBasics(L=32;Re=250,U=1,T=Float32,mem=Array)
    rect = Rect(Vec(-L/4,-L/4,-L/4), Vec(L/2,L/2,L/2))
    rect_positions = decompose(Point{3, T}, rect)
    rect_faces = decompose(TriangleFace{Int}, rect)
    body = MeshBody(Mesh(rect_positions, rect_faces),map=(x,t)->x.-L/2.f0,mem=mem)
    # generate sim
    Simulation((L,L,L), (U,0,0), L; ν=U*L/Re, body, mem, T)
end
# make a writer with some attributes to output to the file
vtk_velocity(a::AbstractSimulation) = a.flow.u |> Array;
vtk_pressure(a::AbstractSimulation) = a.flow.p |> Array;
vtk_body(a::AbstractSimulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); a.flow.σ |> Array;)
vtk_mu0(a::AbstractSimulation) = a.flow.μ₀ |> Array;
vtk_normal(a::AbstractSimulation) = (WaterLily.@loop a.flow.f[I,:] .= measure(a.body,loc(0,I),0)[2] over I in CartesianIndices(a.flow.p);
                                     a.flow.f |> Array;)
custom_attrib = Dict("u"=>vtk_velocity, "p"=>vtk_pressure, "d"=>vtk_body, "μ₀"=>vtk_mu0, "n"=>vtk_normal)
using LinearAlgebra: cross
@fastmath @inline normal(tri::SMatrix) = hat(SVector(cross(tri[:,2]-tri[:,1],tri[:,3]-tri[:,1])))
@fastmath @inline hat(v) = v/(√(v'*v)+eps(eltype(v)))
vtu_normal(a) = [normal(tri) for tri in Array(sim.body.mesh)]
L = 32
MEMORY = Array
# sim = make_aorta(L;mem=MEMORY)
@time sim = make_cube(L;mem=MEMORY);
# # sim = make_GeomBasics(L;mem=MEMORY)

@time measure!(sim)

# # make the paraview writer
# wr = vtkWriter("MeshBody";attrib=custom_attrib)
# wr_mesh = vtkWriter("MeshBody_mesh";attrib=Dict("n"=>vtu_normal))
# save!(wr, sim); save!(wr_mesh, sim.body)
# # sim_step!(sim)
# push!(sim.flow.Δt, 1.0)
# save!(wr, sim); save!(wr_mesh, sim.body)
# close(wr); close(wr_mesh)

# grab a random box
# C,R = sim.body.bvh[5]

# split the mesh to submeshed and fit the box tight to it
function split_fit!(I,box_list,mesh,δ=1)
    (O,W) = box_list[I]
    @show I,O,W
    T = eltype(O)
    sub_mesh = Int[] # populate with dummy
    vmax = SVector{3,T}(typemin(T),typemin(T),typemin(T))
    vmin = SVector{3,T}(typemax(T),typemax(T),typemax(T))
    for (i,el) in enumerate(mesh)
        # if any of the vertices is inside, the element is inside
        # if any(all(O .<= el .<= O+W, dims=2))
        if all(O.-δ .<= el .<= O+W.+2δ)
            # remember it
            push!(sub_mesh,i)
            # new box, minimum of coordinates sets bounding box
            vmin = min.(first(minimum(el,dims=1)), vmin)
            vmax = max.(first(maximum(el,dims=1)), vmax)
        end
    end
    @show vmin,vmax
    o = vmin .- T(δ)
    r = (vmax - vmin)/2 .+ T(2δ) # make it a bit bigger
    # box_list[I] = (o,r) # update the box
    return sub_mesh
end


lvl = 5
# sim = make_cube(L;mem=MEMORY);
sim = make_aorta(L;mem=MEMORY);
# @show sim.body.bvh
sub_meshes_id = Vector[]
for I in 2^(lvl-1):2^lvl-1 # only on leafs
    subMesh = split_fit!(I,sim.body.bvh,sim.body.mesh,0.0)
    push!(sub_meshes_id,subMesh)
    @show I, length(subMesh)
end
println()

using Plots
function plot_box(box, ax, k, c)
    lo,up = box[1], box[1]+box[2]
    lo = lo .+ 0.01k; up = up .- 0.01k
    lines = [
        [lo[1],lo[1],up[1],up[1],lo[1],lo[1],lo[1],up[1],up[1],lo[1]],
        [lo[2],up[2],up[2],lo[2],lo[2],lo[2],up[2],up[2],lo[2],lo[2]],
        [lo[3],lo[3],lo[3],lo[3],lo[3],up[3],up[3],up[3],up[3],up[3]]
    ]
    plot!(ax, lines...;color=c[k],label="box $k|",lw=2)
    plot!(ax, [up[1],up[1]],[lo[2],lo[2]],[up[3],lo[3]];color=c[k],label=:none,lw=2);
    plot!(ax, [up[1],up[1]],[up[2],up[2]],[up[3],lo[3]];color=c[k],label=:none,lw=2)
    plot!(ax, [lo[1],lo[1]],[up[2],up[2]],[up[3],lo[3]];color=c[k],label=:none,lw=2)
end

ax = plot(size=(800,800),dpi=1200)
colors = distinguishable_colors(length(sim.body.bvh))
for (k,b) in enumerate(sim.body.bvh)
    plot_box(b,ax,k,colors)
end
for (k,sub) in enumerate(sub_meshes_id)
    subm = sim.body.mesh[sub]
    scatter!([p[1] for p in subm], [p[2] for p in subm], [p[3] for p in subm];
             markersize=8, color=colors[k+2^(lvl-1)-1], label=:none)
end
ax

# make sub meshed
sub_mesh = Vector[]
for id in sub_meshes_id
    push!(sub_mesh, sim.body.mesh[id])
end

@fastmath @inline sibling(current::Int) = current%2==0 ? current+1 : current-1  #0 alloc
@fastmath @inline parent(current::Int)= fld(current,2)  # 0 alloc

@fastmath function traverse_fsm(x::SVector{3,T},sub_meshes::AbstractVector,
                                bvh::AbstractVector,sub_meshes_id)::Tuple{Int,T} where T

    N = fld(length(bvh),2)
    u,a=(0,T(64)) # this one doesn't exist, and we use a square distance
    state = :fromParent
    current=1
    off = length(bvh) - length(sub_meshes)  # offset to access subsets
    while true
        if state==:fromSibling
            hit = inside(x,bvh[current]...)
            if !hit
                break
            elseif current > N
                v,b = closest(subset(current-off,sub_meshes),x)
                abs(b)<abs(a) && (a=b; u=subset(current-off,sub_meshes_id)[v])
                break
            else
                current = 2current
                state = :fromParent
            end
        elseif state==:fromParent
            hit = inside(x,bvh[current]...)
            if !hit && current ==1
                break
            elseif !hit
                current = sibling(current)
                state = :fromSibling
            elseif current > N
                v,b = closest(subset(current-off,sub_meshes),x)
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

@inline subset(I,mesh) = mesh[I]

using BenchmarkTools
@fastmath @inline d²_fast(tri::SMatrix,x::SVector) = sum(abs2,x-SVector(sum(tri,dims=2)/3))
@inline function closest(mesh,x::SVector{D,T};kwargs...)::Tuple{Int,T} where {D,T}
    u=1; a=b=d²_fast(@views(mesh[1]),x) # fast method
    for I in 2:length(mesh)
        b = d²_fast(@views(mesh[I]),x)
        b<a && (a=b; u=I) # Replace current best
    end
    return u,a
end

using ForwardDiff
function measure_bvh(body,x::SVector{D,T},submesh,bvh,sub_meshes_id;t=0) where {D,T}
    # map to correct location
    ξ = body.map(x,t)
    # we don't need to worry if the geom is a boundary or not
    !(all(body.origin .≤ ξ) && all(ξ .≤ body.origin+body.width)) && return (max(8,2body.half_thk),zeros(SVector{D,T}),zeros(SVector{D,T}))
    # locate the point on the mesh
    u,a = traverse_fsm(ξ,submesh,bvh,sub_meshes_id)
    # check that we have found something
    u == 0 && return (max(8,2body.half_thk),zeros(SVector{D,T}),zeros(SVector{D,T}))
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

body = sim.body;
x = body.mesh[112][:,1] .+ SVector(0.1f0,0.1f0,1f0)
mesh = sim.body.mesh;
bvh = sim.body.bvh;
σ = sim.flow.σ;

# simple tests
closest(mesh,x)
traverse_fsm(x,sub_mesh,bvh,sub_meshes_id)

# test a single function
@btime r = closest($mesh,$x)
# @code_warntype closest(mesh,x)

@btime s = traverse_fsm($x,$sub_mesh,$bvh,$sub_meshes_id)
# @which traverse_fsm(x,sub_mesh,bvh)
# @code_warntype traverse_fsm(x,sub_mesh,bvh)

m1(σ,body) = @inside σ[I] = sdf(body,loc(0,I),0.0)
m2(σ,sub_mesh,bvh,sub_meshes_id) = @inside σ[I] = traverse_fsm(loc(0,I),sub_mesh,bvh,sub_meshes_id)[2]
# # measures
@btime m1($σ,$body)           # 38.265 ms (173 allocations: 12.50 KiB)
@btime m2($σ,$sub_mesh,$bvh,$sub_meshes_id)  # 0.160 ms (1921 allocations: 89.44 KiB)
@inside σ[I] = sdf(body,loc(0,I),0.0)
@inside σ[I] = measure_bvh(body,loc(0,I),sub_mesh,bvh,sub_meshes_id)[1]
flood(σ[:,:,9])
flood(σ[:,9,:])

function expand_bvh_points(bvh)
    points = Float32[]
    for (C,W) in bvh
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

expand_bvh_points(sim.body.bvh)

points = expand_bvh_points(body.bvh)
cells = [MeshCell(VTKCellTypes.VTK_VOXEL, collect(8i+1:8i+8)) for i in 0:length(body.bvh)-1]
vtk = vtk_grid("test_bvh", points, cells)
vtk["ID"] = 1:length(cells)
vtk_save(vtk)
