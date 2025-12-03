using WaterLily,StaticArrays
using GeometryBasics,CUDA,WriteVTK
using Plots; gr()

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

# make the paraview writer
wr = vtkWriter("MeshBody";attrib=custom_attrib)
wr_mesh = vtkWriter("MeshBody_mesh";attrib=Dict("n"=>vtu_normal))
save!(wr, sim); save!(wr_mesh, sim.body)
# sim_step!(sim)
push!(sim.flow.Δt, 1.0)
save!(wr, sim); save!(wr_mesh, sim.body)
close(wr); close(wr_mesh)

# grab a random box
C,R = sim.body.bvh[5]

# split the mesh to submeshed and fit the box tight to it
function split_fit!(I,box_list,mesh,δ=0)
    (O,W) = box_list[I]
    @show I,O,W
    T = eltype(O)
    sub_mesh = [] # populate with dummy
    vmax = SVector{3,T}(typemin(T),typemin(T),typemin(T))
    vmin = SVector{3,T}(typemax(T),typemax(T),typemax(T))
    for (i,el) in enumerate(mesh)
        # if any of the vertices is inside, the element is inside
        # if any(all(O .<= el .<= O+W, dims=2))
        if all(O.-1 .<= el .<= O+W.+2)
            # remember it
            push!(sub_mesh,i)
            # new box, minimum of coordinates sets bounding box
            vmin = min.(first(minimum(el,dims=1)), vmin)
            vmax = max.(first(maximum(el,dims=1)), vmax)
        end
    end
    @show vmin,vmax
    o = vmin .- T(δ/2)
    r = (vmax - vmin)/2 .+ T(δ) # make it a bit bigger
    # box_list[I] = (o,r) # update the box
    return sub_mesh
end
println("\ntesting split")
lvl = 3
sim = make_cube(L;mem=MEMORY);
# @show sim.body.bvh
sub_meshes = []
for I in 2^(lvl-1):2^lvl-1 # only on leafs
    subMesh = split_fit!(I,sim.body.bvh,sim.body.mesh,0.0)
    push!(sub_meshes,subMesh)
    @show I, length(subMesh)
end
println()
# @show sim.body.bvh


using Plots
function plot_box(box, ax, k, c)
    lo,up = box[1], box[1]+box[2]
    lo = lo .+ 0.1k; up = up .- 0.1k
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
for (k,sub) in enumerate(sub_meshes)
    subm = sim.body.mesh[sub]
    scatter!([p[1] for p in subm], [p[2] for p in subm], [p[3] for p in subm];
             markersize=8, color=colors[k+2^(lvl-1)-1], label=:none)
end
ax


@fastmath @inline sibling(current::Int) = current%2==0 ? current+1 : current-1  #0 alloc
@fastmath @inline parent(current::Int)= fld(current,2)  # 0 alloc

@fastmath function traverse_fsm(x::SVector{3,T},mesh,bvh,subsets) where T
    # fromParent = 1
    # fromSibling = 2
    # fromChild = 3

    N = fld(length(bvh),2)
    sol=(0,T(64)) # this one doesn't exist, and we use a square distance
    state = :fromParent
    current=1
    off = length(bvh) - length(subsets)  # offset to access subsets
    while true
        # @show current,state,sol
        # if state==:fromChild
        #     if current==1
        #         break
        #     elseif current==2parent(current)
        #         current = sibling(current)
        #         state = :fromSibling
        #     else
        #         current = parent(current)
        #         state = :fromChild
        #     end
        if state==:fromSibling
            hit = inside(x,bvh[current]...)
            if !hit
                break
                # current = parent(current)
                # state = :fromChild
            elseif current > N #isLeaf(current)
                sol_leaf = closest(@views(mesh[subsets[current-off]]),x)
                abs(sol_leaf[2])<abs(sol[2]) && (sol=sol_leaf)
                break
                # current = parent(current)
                # state = :fromChild
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
            elseif current > N # isLeaf(current)
                sol_leaf = closest(@views(mesh[subsets[current-off]]),x)
                abs(sol_leaf[2])<abs(sol[2]) && (sol=sol_leaf)
                current = sibling(current)
                state = :fromSibling
            else
                current = 2current
                state = :fromParent
            end
        end
    end
    return sol
end
@fastmath @inline d²_fast(tri::SMatrix,x::SVector) = sum(abs2,x-SVector(sum(tri,dims=2)/3))
@inline function closest(mesh,x::SVector{T};kwargs...) where T
    u=1; a=b=d²_fast(mesh[1],x) # fast method
    for I in 2:length(mesh)
        b = d²_fast(mesh[I],x)
        b<a && (a=b; u=I) # Replace current best
    end
    return u,a
end

x = body.mesh[12][:,1] .+ SVector(0.1f0,0.1f0,0.1f0)
@time s = traverse_fsm(x,sim.body.mesh,sim.body.bvh,sub_meshes)

@time closest(sim.body.mesh,x)