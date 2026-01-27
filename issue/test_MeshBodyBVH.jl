using WaterLily,StaticArrays
using GeometryBasics,CUDA,WriteVTK

# dims
L = 256
MEMORY = CuArray
T = Float32

# different meshes
body = MeshBody("/home/marin/Downloads/xyzrgb_dragon.obj",scale=L/215,boundary=true,mem=MEMORY,T=Float32)
# body = MeshBody("//home/marin/Workspace/WaterLilyPreCICE/meshes/sphere.stl",scale=L/2,boundary=true,mem=MEMORY,T=Float32)
# body = MeshBody("/home/marin/Workspace/WaterLilyPreCICE/meshes/cube.stl",scale=L/2,boundary=true,mem=MEMORY,T=Float32)

# # test
sim = Simulation((L,L,L),(1,0,0),L;body,mem=MEMORY,T);
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
custom_attrib = Dict("d"=>vtk_body)
vtu_normal(a) = [normal(tri) for tri in Array(a.mesh)]

# make the paraview writer
wr = vtkWriter("BVHBody";attrib=custom_attrib)
wr_mesh = vtkWriter("BVHBody_mesh";attrib=Dict("n"=>vtu_normal))
@time save!(wr, sim);
@time save!(wr_mesh, sim.body)
close(wr)
close(wr_mesh)
