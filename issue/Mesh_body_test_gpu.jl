using WaterLily,StaticArrays
using GeometryBasics,CUDA,WriteVTK
using Plots; gr()

function make_sphere(L;U=1,mem=CuArray,T=Float32,level=3,use_bvh=true)
    α = π/10.f0 # rotation angle
    function map(x,t)
        Rx = SA[1 0 0; 0 cos(α) -sin(α); 0 sin(α) cos(α)]
        Ry = SA[cos(α) 0 sin(α); 0 1 0; -sin(α) 0 cos(α)]
        Rz = SA[cos(α) -sin(α) 0; sin(α) cos(α) 0; 0 0 1]
        Rx*Ry*Rz*(x.-L/2.f0).+0.5f0
    end
    # make body
    body = MeshBody("/home/marin/Workspace/WaterLilyPreCICE/meshes/sphere.stl";
                    scale=L/2,level=level,map,use_bvh,mem)
    # make sim
    Simulation((L,L,L),(U,0,0),L;body,mem,T)
end
function make_aorta(L=32;Re=250,U=1,T=Float32,mem=Array,level=3,use_bvh=true)
    # make the body from the stl mesh
    body = MeshBody("/home/marin/Workspace/WaterLilyPreCICE/meshes/aorta/aorta.inp";
                    scale=L/2, map=(x,t)->x-SA[L/2.f0,L/2.f0,L/4.f0],
                    boundary=false,half_thk=1.f0,level,use_bvh,mem)
    # generate sim
    Simulation((L,L,L÷2), (0,0,0), L; ν=U*L/Re, body, mem, T)
end
function make_GeomBasics(L=32;Re=250,U=1,T=Float32,mem=Array,level=3,use_bvh=true)
    rect = Rect(Vec(-L/4,-L/4,-L/4), Vec(L/2,L/2,L/2))
    rect_positions = decompose(Point{3, T}, rect)
    rect_faces = decompose(TriangleFace{Int}, rect)
    body = MeshBody(Mesh(rect_positions, rect_faces),map=(x,t)->x.-L/2.f0,level,use_bvh,mem)
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
normal = Base.get_extension(WaterLily, :WaterLilyGeometryBasicsExt).normal
vtu_normal(a) = [normal(tri) for tri in Array(sim.body.mesh)]
L = 256
MEMORY = Array
@time sim = make_aorta(L;mem=MEMORY,level=5,use_bvh=true);
# @time sim = make_sphere(L;mem=MEMORY,level=5,use_bvh=false);
# # sim = make_GeomBasics(L;mem=MEMORY,level=5,use_bvh=false)

# measure the sime one more
@time measure!(sim)

# make the paraview writer
wr = vtkWriter("MeshBody";attrib=custom_attrib)
wr_mesh = vtkWriter("MeshBody_mesh";attrib=Dict("n"=>vtu_normal))
@time save!(wr, sim);
@time save!(wr_mesh, sim.body)
close(wr)
close(wr_mesh)
save!(sim.body.bvh, "bvh")