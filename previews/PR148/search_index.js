var documenterSearchIndex = {"docs":
[{"location":"#WaterLily","page":"WaterLily","title":"WaterLily","text":"","category":"section"},{"location":"#Introduction-and-Quickstart","page":"WaterLily","title":"Introduction and Quickstart","text":"","category":"section"},{"location":"","page":"WaterLily","title":"WaterLily","text":"WaterLily","category":"page"},{"location":"#WaterLily","page":"WaterLily","title":"WaterLily","text":"WaterLily.jl\n\n(Image: Dev) (Image: CI) (Image: codecov)\n\n<p align=\"center\">   <img src=\"examples/julia2024.gif\" width=\"400\"/> </p>\n\nOverview\n\nWaterLily.jl is a simple and fast fluid simulator written in pure Julia. This project is supported by awesome libraries developed within the Julia scientific community, and it aims to accelerate and enhance fluid simulations. Watch the JuliaCon2024 talk here:\n\n<p align=\"center\">   <a href=\"https://www.youtube.com/live/qru5G5Yp77E?t=29074s\"><img src=\"examples/JuliaCon2024.png\" width=\"800\"/></a> </p>\n\nIf you have used WaterLily for research, please cite us! The following manuscript describes the main features of the solver and provides benchmarking, validation, and profiling results\n\n@misc{WeymouthFont2024,\n    title     = {WaterLily.jl: A differentiable and backend-agnostic Julia solver to simulate incompressible fluid flow and dynamic bodies},\n    author    = {Gabriel D. Weymouth and Bernat Font},\n    DOI       = {},\n    publisher = {arXiv},\n    year      = {2024}\n}\n\nMethod/capabilities\n\nWaterLily solves the unsteady incompressible 2D or 3D Navier-Stokes equations on a Cartesian grid. The pressure Poisson equation is solved with a geometric multigrid method. Solid boundaries are modelled using the Boundary Data Immersion Method. The solver can run on serial CPU, multi-threaded CPU, or GPU backends.\n\nExamples\n\nThe user can set the boundary conditions, the initial velocity field, the fluid viscosity (which determines the Reynolds number), and immerse solid obstacles using a signed distance function. These examples and others are found in the examples directory.\n\nFlow over a circle\n\nWe define the size of the simulation domain as ntimesm cells. The circle has radius m/8 and is centered at (m/2,m/2). The flow boundary conditions are (U,0), where we set U=1, and the Reynolds number is Re=U*radius/ν where ν (Greek \"nu\" U+03BD, not Latin lowercase \"v\") is the kinematic viscosity of the fluid.\n\nusing WaterLily\nfunction circle(n,m;Re=250,U=1)\n    radius, center = m/8, m/2\n    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)\n    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body)\nend\n\nThe second to last line defines the circle geometry using a signed distance function. The AutoBody function uses automatic differentiation to infer the other geometric parameter automatically. Replace the circle's distance function with any other, and now you have the flow around something else... such as a donut or the Julia logo. Finally, the last line defines the Simulation by passing in parameters we've defined.\n\nNow we can create a simulation (first line) and run it forward in time (third line)\n\ncirc = circle(3*2^6,2^7)\nt_end = 10\nsim_step!(circ,t_end)\n\nNote we've set n,m to be multiples of powers of 2, which is important when using the (very fast) geometric multi-grid solver. We can now access and plot whatever variables we like. For example, we could print the velocity at I::CartesianIndex using println(circ.flow.u[I]) or plot the whole pressure field using\n\nusing Plots\ncontour(circ.flow.p')\n\nA set of flow metric functions have been implemented and the examples use these to make gifs such as the one above.\n\n3D Taylor Green Vortex\n\nThe three-dimensional Taylor Green Vortex demonstrates many of the other available simulation options. First, you can simulate a nontrivial initial velocity field by passing in a vector function uλ(i,xyz) where i ∈ (1,2,3) indicates the velocity component uᵢ and xyz=[x,y,z] is the position vector.\n\nfunction TGV(; pow=6, Re=1e5, T=Float64, mem=Array)\n    # Define vortex size, velocity, viscosity\n    L = 2^pow; U = 1; ν = U*L/Re\n    # Taylor-Green-Vortex initial velocity field\n    function uλ(i,xyz)\n        x,y,z = @. (xyz-1.5)*π/L               # scaled coordinates\n        i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x\n        i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y\n        return 0.                              # u_z\n    end\n    # Initialize simulation\n    return Simulation((L, L, L), (0, 0, 0), L; U, uλ, ν, T, mem)\nend\n\nThis example also demonstrates the floating point type (T=Float64) and array memory type (mem=Array) options. For example, to run on an NVIDIA GPU we only need to import the CUDA.jl library and initialize the Simulation memory on that device.\n\nimport CUDA\n@assert CUDA.functional()\nvortex = TGV(T=Float32,mem=CUDA.CuArray)\nsim_step!(vortex,1)\n\nFor an AMD GPU, use import AMDGPU and mem=AMDGPU.ROCArray. Note that Julia 1.9 is required for AMD GPUs.\n\nMoving bodies\n\n<p align=\"center\">   <img src=\"examples/hover.gif\"/> </p>\n\nYou can simulate moving bodies in WaterLily by passing a coordinate map to AutoBody in addition to the sdf.\n\nusing StaticArrays\nfunction hover(L=2^5;Re=250,U=1,amp=π/4,ϵ=0.5,thk=2ϵ+√2)\n    # Line segment SDF\n    function sdf(x,t)\n        y = x .- SA[0,clamp(x[2],-L/2,L/2)]\n        √sum(abs2,y)-thk/2\n    end\n    # Oscillating motion and rotation\n    function map(x,t)\n        α = amp*cos(t*U/L); R = SA[cos(α) sin(α); -sin(α) cos(α)]\n        R * (x - SA[3L-L*sin(t*U/L),4L])\n    end\n    Simulation((6L,6L),(0,0),L;U,ν=U*L/Re,body=AutoBody(sdf,map),ϵ)\nend\n\nIn this example, the sdf function defines a line segment from -L/2 ≤ x[2] ≤ L/2 with a thickness thk. To make the line segment move, we define a coordinate transformation function map(x,t). In this example, the coordinate x is shifted by (3L,4L) at time t=0, which moves the center of the segment to this point. However, the horizontal shift varies harmonically in time, sweeping the segment left and right during the simulation. The example also rotates the segment using the rotation matrix R = [cos(α) sin(α); -sin(α) cos(α)] where the angle α is also varied harmonically. The combined result is a thin flapping line, similar to a cross-section of a hovering insect wing.\n\nOne important thing to note here is the use of StaticArrays to define the sdf and map. This speeds up the simulation since it eliminates allocations at every grid cell and time step.\n\nCircle inside an oscillating flow\n\n<p align=\"center\">   <img src=\"examples/oscillating.gif\"/> </p>\n\nThis example demonstrates a 2D oscillating periodic flow over a circle.\n\nfunction circle(n,m;Re=250,U=1)\n    # define a circle at the domain center\n    radius = m/8\n    body = AutoBody((x,t)->√sum(abs2, x .- (n/2,m/2)) - radius)\n\n    # define time-varying body force `g` and periodic direction `perdir`\n    accelScale, timeScale = U^2/2radius, radius/U\n    g(i,t) = i==1 ? -2accelScale*sin(t/timeScale) : 0\n    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, g, perdir=(1,))\nend\n\nThe g argument accepts a function with direction (i) and time (t) arguments. This allows you to create a spatially uniform body force with variations over time. In this example, the function adds a sinusoidal force in the \"x\" direction i=1, and nothing to the other directions.\n\nThe perdir argument is a tuple that specifies the directions to which periodic boundary conditions should be applied. Any number of directions may be defined as periodic, but in this example only the i=1 direction is used allowing the flow to accelerate freely in this direction.\n\nAccelerating reference frame\n\nWaterLily gives the possibility to set up a Simulation using time-varying boundary conditions for the velocity field, as demonstrated in this example. This can be used to simulate a flow in an accelerating reference frame. The following example demonstrates how to set up a Simulation with a time-varying velocity field.\n\nusing WaterLily\n# define time-varying velocity boundary conditions\nUt(i,t::T;a0=0.5) where T = i==1 ? convert(T, a0*t) : zero(T)\n# pass that to the function that creates the simulation\nsim = Simulation((256,256), Ut, 32)\n\nThe Ut function is used to define the time-varying velocity field. In this example, the velocity in the \"x\" direction is set to a0*t where a0 is the acceleration of the reference frame. The Simulation function is then called with the Ut function as the second argument. The simulation will then run with the time-varying velocity field.\n\nPeriodic and convective boundary conditions\n\nIn addition to the standard free-slip (or reflective) boundary conditions, WaterLily also supports periodic boundary conditions, as demonstrated in this example. For instance, to set up a Simulation with periodic boundary conditions in the \"y\" direction the perdir=(2,) keyword argument should be passed\n\nusing WaterLily,StaticArrays\n\n# sdf an map for a moving circle in y-direction\nfunction sdf(x,t)\n    norm2(SA[x[1]-192,mod(x[2]-384,384)-192])-32\nend\nfunction map(x,t)\n    x.-SA[0.,t/2]\nend\n\n# make a body\nbody = AutoBody(sdf, map)\n\n# y-periodic boundary conditions\nSimulation((512,384), (1,0), 32; body, perdir=(2,))\n\nAdditionally, the flag exitBC=true can be passed to the Simulation function to enable convective boundary conditions. This will apply a 1D convective exit in the x direction (currently, only the x direction is supported for the convective outlet BC). The exitBC flag is set to false by default. In this case, the boundary condition is set to the corresponding value of the u_BC vector specified when constructing the Simulation.\n\nusing WaterLily\n\n# make a body\nbody = AutoBody(sdf, map)\n\n# y-periodic boundary conditions\nSimulation((512,384), u_BC=(1,0), L=32; body, exitBC=true)\n\nWriting to a VTK file\n\nThe following example demonstrates how to write simulation data to a .pvd file using the WriteVTK package and the WaterLily vtkwriter function. The simplest writer can be instantiated with\n\nusing WaterLily,WriteVTK\n\n# make a sim\nsim = make_sim(...)\n\n# make a writer\nwriter = vtkwriter(\"simple_writer\")\n\n# write the data\nwrite!(writer,sim)\n\n# don't forget to close the file\nclose(writer)\n\nThis would write the velocity and pressure fields to a file named simmple_writer.pvd. The vtkwriter function can also take a dictionary of custom attributes to write to the file. For example, the following code can be run to write the body signed-distance function and λ₂ fields to the file\n\nusing WaterLily,WriteVTK\n\n# make a writer with some attributes, need to output to CPU array to save file (|> Array)\nvelocity(a::Simulation) = a.flow.u |> Array;\npressure(a::Simulation) = a.flow.p |> Array;\n_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a));\n                                     a.flow.σ |> Array;)\nlamda(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u);\n                        a.flow.σ |> Array;)\n\n# map field names to values in the file\ncustom_attrib = Dict(\n    \"Velocity\" => velocity,\n    \"Pressure\" => pressure,\n    \"Body\" => _body,\n    \"Lambda\" => lamda\n)\n\n# make the writer\nwriter = vtkWriter(\"advanced_writer\"; attrib=custom_attrib)\n...\nclose(writer)\n\nThe functions that are passed to the attrib (custom attributes) must follow the same structure as what is shown in this example, that is, given a Simulation, return an N-dimensional (scalar or vector) field. The vtkwriter function will automatically write the data to a .pvd file, which can be read by ParaView. The prototype for the vtkwriter function is:\n\n# prototype vtk writer function\ncustom_vtk_function(a::Simulation) = ... |> Array\n\nwhere ... should be replaced with the code that generates the field you want to write to the file. The piping to a (CPU) Array is necessary to ensure that the data is written to the CPU before being written to the file for GPU simulations.\n\nRestarting from a VTK file\n\nThe restart of a simulation from a VTK file is demonstrated in this example. The ReadVTK package is used to read simulation data from a .pvd file. This .pvd must have been written with the vtkwriter function and must contain at least the velocity and pressure fields. The following example demonstrates how to restart a simulation from a .pvd file using the ReadVTK package and the WaterLily vtkreader function\n\nusing WaterLily,ReadVTK\nsim = make_sim(...)\n# restart the simulation\nwriter = restart_sim!(sim; fname=\"file_restart.pvd\")\n\n# append sim data to the file used for restart\nwrite!(writer, sim)\n\n# don't forget to close the file\nclose(writer)\n\nInternally, this function reads the last file in the .pvd file and use that to set the velocity and pressure fields in the simulation. The sim_time is also set to the last value saved in the .pvd file. The function also returns a vtkwriter that will append the new data to the file used to restart the simulation. Note that the simulation object sim that will be filled must be identical to the one saved to the file for this restart to work, that is, the same size, same body, etc.\n\nMulti-threading and GPU backends\n\nWaterLily uses KernelAbstractions.jl to multi-thread on CPU and run on GPU backends. The implementation method and speed-up are documented in our preprint. In summary, a single macro WaterLily.@loop is used for nearly every loop in the code base, and this uses KernelAbstractactions to generate optimized code for each back-end. The speed-up with respect to a serial (single thread) execution is more pronounce for large simulations, and we have measure up to x8 speedups when multi-threading on an Intel Xeon Platinum 8460Y @ 2.3GHz backend, and up to 200x speedup on an NVIDIA Hopper H100 64GB HBM2 GPU. When maximizing the GPU load, a cost of 1.44 nano-seconds has been measured per degree of freedom and time step.\n\nNote that multi-threading requires starting Julia with the  --threads argument, see the multi-threading section of the manual. If you are running Julia with multiple threads, KernelAbstractions will detect this and multi-thread the loops automatically. As in the Taylor-Green-Vortex examples above, running on a GPU requires initializing the Simulation memory on the GPU, and care needs to be taken to move the data back to the CPU for visualization. See jelly fish for another non-trivial example.\n\nFinally, KernelAbstractions does incur some CPU allocations for every loop, but other than this sim_step! is completely non-allocating. This is one reason why the speed-up improves as the size of the simulation increases.\n\nDevelopment goals\n\nImmerse obstacles defined by 3D meshes using GeometryBasics.\nMulti-CPU/GPU simulations.\nFree-surface physics with Volume-of-Fluid or Level-Set.\nExternal potential-flow domain boundary conditions.\n\nIf you have other suggestions or want to help, please raise an issue.\n\n\n\n\n\n","category":"module"},{"location":"#Types-Methods-and-Functions","page":"WaterLily","title":"Types Methods and Functions","text":"","category":"section"},{"location":"","page":"WaterLily","title":"WaterLily","text":"CurrentModule = WaterLily","category":"page"},{"location":"","page":"WaterLily","title":"WaterLily","text":"","category":"page"},{"location":"","page":"WaterLily","title":"WaterLily","text":"Modules = [WaterLily]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"#WaterLily.AbstractBody","page":"WaterLily","title":"WaterLily.AbstractBody","text":"AbstractBody\n\nImmersed body Abstract Type. Any AbstractBody subtype must implement\n\nd = sdf(body::AbstractBody, x, t=0)\n\nand\n\nd,n,V = measure(body::AbstractBody, x, t=0)\n\nwhere d is the signed distance from x to the body at time t, and n & V are the normal and velocity vectors implied at x.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.AbstractPoisson","page":"WaterLily","title":"WaterLily.AbstractPoisson","text":"Poisson{N,M}\n\nComposite type for conservative variable coefficient Poisson equations:\n\n∮ds β ∂x/∂n = σ\n\nThe resulting linear system is\n\nAx = [L+D+L']x = z\n\nwhere A is symmetric, block-tridiagonal and extremely sparse. Moreover,  D[I]=-∑ᵢ(L[I,i]+L'[I,i]). This means matrix storage, multiplication, ect can be easily implemented and optimized without external libraries.\n\nTo help iteratively solve the system above, the Poisson structure holds helper arrays for inv(D), the error ϵ, and residual r=z-Ax. An iterative solution method then estimates the error ϵ=̃A⁻¹r and increments x+=ϵ, r-=Aϵ.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.AutoBody","page":"WaterLily","title":"WaterLily.AutoBody","text":"AutoBody(sdf,map=(x,t)->x; compose=true) <: AbstractBody\n\nsdf(x::AbstractVector,t::Real)::Real: signed distance function\nmap(x::AbstractVector,t::Real)::AbstractVector: coordinate mapping function\ncompose::Bool=true: Flag for composing sdf=sdf∘map\n\nImplicitly define a geometry by its sdf and optional coordinate map. Note: the map is composed automatically if compose=true, i.e. sdf(x,t) = sdf(map(x,t),t). Both parameters remain independent otherwise. It can be particularly heplful to set compose=false when adding mulitple bodies together to create a more complex one.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.Bodies","page":"WaterLily","title":"WaterLily.Bodies","text":"Bodies(bodies, ops::AbstractVector)\n\nbodies::Vector{AutoBody}: Vector of AutoBody\nops::Vector{Function}: Vector of operators for the superposition of multiple AutoBodys\n\nSuperposes multiple body::AutoBody objects together according to the operators ops. While this can be manually performed by the operators implemented for AutoBody, adding too many bodies can yield a recursion problem of the sdf and map functions not fitting in the stack. This type implements the superposition of bodies by iteration instead of recursion, and the reduction of the sdf and map functions is done on the mesure function, and not before. The operators vector ops specifies the operation to call between two consecutive AutoBodys in the bodies vector. Note that + (or the alias ∪) is the only operation supported between Bodies.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.Flow","page":"WaterLily","title":"WaterLily.Flow","text":"Flow{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}, Tf<:AbstractArray{T,D+2}}\n\nComposite type for a multidimensional immersed boundary flow simulation.\n\nFlow solves the unsteady incompressible Navier-Stokes equations on a Cartesian grid. Solid boundaries are modelled using the Boundary Data Immersion Method. The primary variables are the scalar pressure p (an array of dimension D) and the velocity vector field u (an array of dimension D+1).\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.MultiLevelPoisson","page":"WaterLily","title":"WaterLily.MultiLevelPoisson","text":"MultiLevelPoisson{N,M}\n\nComposite type used to solve the pressure Poisson equation with a geometric multigrid method. The only variable is levels, a vector of nested Poisson systems.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.NoBody","page":"WaterLily","title":"WaterLily.NoBody","text":"NoBody\n\nUse for a simulation without a body.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.Simulation","page":"WaterLily","title":"WaterLily.Simulation","text":"Simulation(dims::NTuple, u_BC::Union{NTuple,Function}, L::Number;\n           U=norm2(u_BC), Δt=0.25, ν=0., ϵ=1, perdir=()\n           uλ::nothing, g=nothing, exitBC=false,\n           body::AbstractBody=NoBody(),\n           T=Float32, mem=Array)\n\nConstructor for a WaterLily.jl simulation:\n\ndims: Simulation domain dimensions.\nu_BC: Simulation domain velocity boundary conditions, either a         tuple u_BC[i]=uᵢ, i=eachindex(dims), or a time-varying function f(i,t)\nL: Simulation length scale.\nU: Simulation velocity scale.\nΔt: Initial time step.\nν: Scaled viscosity (Re=UL/ν).\ng: Domain acceleration, g(i,t)=duᵢ/dt\nϵ: BDIM kernel width.\nperdir: Domain periodic boundary condition in the (i,) direction.\nexitBC: Convective exit boundary condition in the i=1 direction.\nuλ: Function to generate the initial velocity field.\nbody: Immersed geometry.\nT: Array element type.\nmem: memory location. Array, CuArray, ROCm to run on CPU, NVIDIA, or AMD devices, respectively.\n\nSee files in examples folder for examples.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.BC!","page":"WaterLily","title":"WaterLily.BC!","text":"BC!(a,A)\n\nApply boundary conditions to the ghost cells of a vector field. A Dirichlet condition a[I,i]=A[i] is applied to the vector component normal to the domain boundary. For example aₓ(x)=Aₓ ∀ x ∈ minmax(X). A zero Neumann condition is applied to the tangential components.\n\n\n\n\n\n","category":"function"},{"location":"#WaterLily.BCTuple","page":"WaterLily","title":"WaterLily.BCTuple","text":"BCTuple(U,dt,N)\n\nReturn BC tuple U(i∈1:N, t=sum(dt)).\n\n\n\n\n\n","category":"function"},{"location":"#WaterLily.CIj-Union{Tuple{d}, Tuple{Any, CartesianIndex{d}, Any}} where d","page":"WaterLily","title":"WaterLily.CIj","text":"CIj(j,I,jj)\n\nReplace jᵗʰ component of CartesianIndex with k\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.Jacobi!-Tuple{Any}","page":"WaterLily","title":"WaterLily.Jacobi!","text":"Jacobi!(p::Poisson; it=1)\n\nJacobi smoother run it times.  Note: This runs for general backends, but is very slow to converge.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.L₂-Tuple{Any}","page":"WaterLily","title":"WaterLily.L₂","text":"L₂(a)\n\nL₂ norm of array a excluding ghosts.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.accelerate!","page":"WaterLily","title":"WaterLily.accelerate!","text":"accelerate!(r,dt,g)\n\nAdd a uniform acceleration gᵢ+dUᵢ/dt at time t=sum(dt) to field r.\n\n\n\n\n\n","category":"function"},{"location":"#WaterLily.apply!-Tuple{Any, Any}","page":"WaterLily","title":"WaterLily.apply!","text":"apply!(f, c)\n\nApply a vector function f(i,x) to the faces of a uniform staggered array c or a function f(x) to the center of a uniform array c.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.check_nthreads-Tuple{Val{1}}","page":"WaterLily","title":"WaterLily.check_nthreads","text":"check_nthreads(::Val{1})\n\nCheck the number of threads available for the Julia session that loads WaterLily. A warning is shown when running in serial (JULIANUMTHREADS=1).\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.curl-Tuple{Any, Any, Any}","page":"WaterLily","title":"WaterLily.curl","text":"curl(i,I,u)\n\nCompute component i of 𝐮 at the edge of cell I. For example curl(3,CartesianIndex(2,2,2),u) will compute ω₃(x=1.5,y=1.5,z=2) as this edge produces the highest accuracy for this mix of cross derivatives on a staggered grid.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.curvature-Tuple{AbstractMatrix}","page":"WaterLily","title":"WaterLily.curvature","text":"curvature(A::AbstractMatrix)\n\nReturn H,K the mean and Gaussian curvature from A=hessian(sdf). K=tr(minor(A)) in 3D and K=0 in 2D.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.exitBC!-NTuple{4, Any}","page":"WaterLily","title":"WaterLily.exitBC!","text":"exitBC!(u,u⁰,U,Δt)\n\nApply a 1D convection scheme to fill the ghost cell on the exit of the domain.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.inside-Tuple{AbstractArray}","page":"WaterLily","title":"WaterLily.inside","text":"inside(a)\n\nReturn CartesianIndices range excluding a single layer of cells on all boundaries.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.inside_u-Union{Tuple{N}, Tuple{Tuple{Vararg{T, N}} where T, Any}} where N","page":"WaterLily","title":"WaterLily.inside_u","text":"inside_u(dims,j)\n\nReturn CartesianIndices range excluding the ghost-cells on the boundaries of a vector array on face j with size dims.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.interp-Union{Tuple{T}, Tuple{D}, Tuple{StaticArraysCore.SVector{D}, AbstractArray{T, D}}} where {D, T}","page":"WaterLily","title":"WaterLily.interp","text":"interp(x::SVector, arr::AbstractArray)\n\nLinear interpolation from array `arr` at index-coordinate `x`.\nNote: This routine works for any number of dimensions.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.ke-Union{Tuple{m}, Tuple{CartesianIndex{m}, Any}, Tuple{CartesianIndex{m}, Any, Any}} where m","page":"WaterLily","title":"WaterLily.ke","text":"ke(I::CartesianIndex,u,U=0)\n\nCompute ½𝐮-𝐔² at center of cell I where U can be used to subtract a background flow (by default, U=0).\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.loc-Union{Tuple{N}, Tuple{Any, CartesianIndex{N}}} where N","page":"WaterLily","title":"WaterLily.loc","text":"loc(i,I) = loc(Ii)\n\nLocation in space of the cell at CartesianIndex I at face i. Using i=0 returns the cell center s.t. loc = I.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.measure!","page":"WaterLily","title":"WaterLily.measure!","text":"measure!(sim::Simulation,t=timeNext(sim))\n\nMeasure a dynamic body to update the flow and pois coefficients.\n\n\n\n\n\n","category":"function"},{"location":"#WaterLily.measure!-Union{Tuple{T}, Tuple{N}, Tuple{Flow{N, T, Sf, Vf, Tf} where {Sf<:(AbstractArray{T}), Vf<:(AbstractArray{T}), Tf<:(AbstractArray{T})}, AbstractBody}} where {N, T}","page":"WaterLily","title":"WaterLily.measure!","text":"measure!(flow::Flow, body::AbstractBody; t=0, ϵ=1)\n\nQueries the body geometry to fill the arrays:\n\nflow.μ₀, Zeroth kernel moment\nflow.μ₁, First kernel moment scaled by the body normal\nflow.V,  Body velocity\n\nat time t using an immersion kernel of size ϵ.\n\nSee Maertens & Weymouth, doi:10.1016/j.cma.2014.09.007.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.measure-Tuple{AutoBody, Any, Any}","page":"WaterLily","title":"WaterLily.measure","text":"d,n,V = measure(body::AutoBody,x,t;fast=false)\nd,n,V = measure(body::Bodies,x,t;fast=false)\n\nDetermine the implicit geometric properties from the sdf and map. The gradient of d=sdf(map(x,t)) is used to improve d for pseudo-sdfs. The velocity is determined solely from the optional map function. Using fast=true skips the n,V calculation when |d|>1.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.measure_sdf!","page":"WaterLily","title":"WaterLily.measure_sdf!","text":"measure_sdf!(a::AbstractArray, body::AbstractBody, t=0)\n\nUses sdf(body,x,t) to fill a.\n\n\n\n\n\n","category":"function"},{"location":"#WaterLily.mom_step!-Union{Tuple{N}, Tuple{Flow{N, T} where T, AbstractPoisson}} where N","page":"WaterLily","title":"WaterLily.mom_step!","text":"mom_step!(a::Flow,b::AbstractPoisson)\n\nIntegrate the Flow one time step using the Boundary Data Immersion Method and the AbstractPoisson pressure solver to project the velocity onto an incompressible flow.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.mult!-Tuple{Poisson, Any}","page":"WaterLily","title":"WaterLily.mult!","text":"mult!(p::Poisson,x)\n\nEfficient function for Poisson matrix-vector multiplication.  Fills p.z = p.A x with 0 in the ghost cells.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.nds-Tuple{Any, Any, Any}","page":"WaterLily","title":"WaterLily.nds","text":"nds(body,x,t)\n\nBDIM-masked surface normal.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.pcg!-Union{Tuple{Poisson{T, S, V} where {S<:(AbstractArray{T}), V<:(AbstractArray{T})}}, Tuple{T}} where T","page":"WaterLily","title":"WaterLily.pcg!","text":"pcg!(p::Poisson; it=6)\n\nConjugate-Gradient smoother with Jacobi preditioning. Runs at most it iterations,  but will exit early if the Gram-Schmidt update parameter |α| < 1% or |r D⁻¹ r| < 1e-8. Note: This runs for general backends and is the default smoother.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.perBC!-Tuple{Any, Tuple{}}","page":"WaterLily","title":"WaterLily.perBC!","text":"perBC!(a,perdir)\n\nApply periodic conditions to the ghost cells of a scalar field.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.pressure_force-Tuple{Any}","page":"WaterLily","title":"WaterLily.pressure_force","text":"pressure_force(sim::Simulation)\n\nCompute the pressure force on an immersed body.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.pressure_moment-Tuple{Any, Any}","page":"WaterLily","title":"WaterLily.pressure_moment","text":"pressure_moment(x₀,sim::Simulation)\n\nComputes the pressure moment on an immersed body relative to point x₀.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.reduce_sdf_map-NTuple{7, Any}","page":"WaterLily","title":"WaterLily.reduce_sdf_map","text":"reduce_sdf_map(sdf_a,map_a,d_a,sdf_b,map_b,d_b,op,x,t)\n\nReduces two different sdf and map functions, and d value.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.residual!-Tuple{Poisson}","page":"WaterLily","title":"WaterLily.residual!","text":"residual!(p::Poisson)\n\nComputes the resiual r = z-Ax and corrects it such that r = 0 if iD==0 which ensures local satisfiability     and  sum(r) = 0 which ensures global satisfiability.\n\nThe global correction is done by adjusting all points uniformly,  minimizing the local effect. Other approaches are possible.\n\nNote: These corrections mean x is not strictly solving Ax=z, but without the corrections, no solution exists.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.sdf-Tuple{AutoBody, Any, Any}","page":"WaterLily","title":"WaterLily.sdf","text":"d = sdf(body::AutoBody,x,t) = body.sdf(x,t)\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.sdf-Tuple{Bodies, Any, Any}","page":"WaterLily","title":"WaterLily.sdf","text":"d = sdf(a::Bodies,x,t)\n\nComputes distance for Bodies type.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.sdf_map_d-NTuple{4, Any}","page":"WaterLily","title":"WaterLily.sdf_map_d","text":"sdf_map_d(ab::Bodies,x,t)\n\nReturns the sdf and map functions, and the distance d (d=sdf(x,t)) for the Bodies type.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.sim_step!-Tuple{Simulation, Any}","page":"WaterLily","title":"WaterLily.sim_step!","text":"sim_step!(sim::Simulation,t_end=sim(time)+Δt;max_steps=typemax(Int),remeasure=true,verbose=false)\n\nIntegrate the simulation sim up to dimensionless time t_end. If remeasure=true, the body is remeasured at every time step. Can be set to false for static geometries to speed up simulation.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.sim_time-Tuple{Simulation}","page":"WaterLily","title":"WaterLily.sim_time","text":"sim_time(sim::Simulation)\n\nReturn the current dimensionless time of the simulation tU/L where t=sum(Δt), and U,L are the simulation velocity and length scales.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.slice-Union{Tuple{N}, Tuple{Tuple{Vararg{T, N}} where T, Any, Any}, Tuple{Tuple{Vararg{T, N}} where T, Any, Any, Any}} where N","page":"WaterLily","title":"WaterLily.slice","text":"slice(dims,i,j,low=1)\n\nReturn CartesianIndices range slicing through an array of size dims in dimension j at index i. low optionally sets the lower extent of the range in the other dimensions.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.solver!-Tuple{Poisson}","page":"WaterLily","title":"WaterLily.solver!","text":"solver!(A::Poisson;log,tol,itmx)\n\nApproximate iterative solver for the Poisson matrix equation Ax=b.\n\nA: Poisson matrix with working arrays.\nA.x: Solution vector. Can start with an initial guess.\nA.z: Right-Hand-Side vector. Will be overwritten!\nA.n[end]: stores the number of iterations performed.\nlog: If true, this function returns a vector holding the L₂-norm of the residual at each iteration.\ntol: Convergence tolerance on the L₂-norm residual.\nitmx: Maximum number of iterations.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.time-Tuple{Flow}","page":"WaterLily","title":"WaterLily.time","text":"time(a::Flow)\n\nCurrent flow time.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.total_force-Tuple{Any}","page":"WaterLily","title":"WaterLily.total_force","text":"total_force(sim::Simulation)\n\nCompute the total force on an immersed body.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.viscous_force-Tuple{Any}","page":"WaterLily","title":"WaterLily.viscous_force","text":"viscous_force(sim::Simulation)\n\nCompute the viscous force on an immersed body.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.δ-Union{Tuple{N}, Tuple{Any, Val{N}}} where N","page":"WaterLily","title":"WaterLily.δ","text":"δ(i,N::Int)\nδ(i,I::CartesianIndex{N}) where {N}\n\nReturn a CartesianIndex of dimension N which is one at index i and zero elsewhere.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.λ₂-Tuple{CartesianIndex{3}, Any}","page":"WaterLily","title":"WaterLily.λ₂","text":"λ₂(I::CartesianIndex{3},u)\n\nλ₂ is a deformation tensor metric to identify vortex cores. See https://en.wikipedia.org/wiki/Lambda2_method and Jeong, J., & Hussain, F., doi:10.1017/S0022112095000462\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.ω-Tuple{CartesianIndex{3}, Any}","page":"WaterLily","title":"WaterLily.ω","text":"ω(I::CartesianIndex{3},u)\n\nCompute 3-vector 𝛚=𝐮 at the center of cell I.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.ω_mag-Tuple{CartesianIndex{3}, Any}","page":"WaterLily","title":"WaterLily.ω_mag","text":"ω_mag(I::CartesianIndex{3},u)\n\nCompute 𝛚 at the center of cell I.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.ω_θ-Tuple{CartesianIndex{3}, Any, Any, Any}","page":"WaterLily","title":"WaterLily.ω_θ","text":"ω_θ(I::CartesianIndex{3},z,center,u)\n\nCompute 𝛚𝛉 at the center of cell I where 𝛉 is the azimuth direction around vector z passing through center.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.∂-NTuple{4, Any}","page":"WaterLily","title":"WaterLily.∂","text":"∂(i,j,I,u)\n\nCompute uᵢxⱼ at center of cell I. Cross terms are computed less accurately than inline terms because of the staggered grid.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.∇²u-Tuple{CartesianIndex{2}, Any}","page":"WaterLily","title":"WaterLily.∇²u","text":"∇²u(I::CartesianIndex,u)\n\nRate-of-strain tensor.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.@inside-Tuple{Any}","page":"WaterLily","title":"WaterLily.@inside","text":"@inside <expr>\n\nSimple macro to automate efficient loops over cells excluding ghosts. For example,\n\n@inside p[I] = sum(loc(0,I))\n\nbecomes\n\n@loop p[I] = sum(loc(0,I)) over I ∈ inside(p)\n\nSee @loop.\n\n\n\n\n\n","category":"macro"},{"location":"#WaterLily.@loop-Tuple","page":"WaterLily","title":"WaterLily.@loop","text":"@loop <expr> over <I ∈ R>\n\nMacro to automate fast loops using @simd when running in serial, or KernelAbstractions when running multi-threaded CPU or GPU.\n\nFor example\n\n@loop a[I,i] += sum(loc(i,I)) over I ∈ R\n\nbecomes\n\n@simd for I ∈ R\n    @fastmath @inbounds a[I,i] += sum(loc(i,I))\nend\n\non serial execution, or\n\n@kernel function kern(a,i,@Const(I0))\n    I ∈ @index(Global,Cartesian)+I0\n    @fastmath @inbounds a[I,i] += sum(loc(i,I))\nend\nkern(get_backend(a),64)(a,i,R[1]-oneunit(R[1]),ndrange=size(R))\n\nwhen multi-threading on CPU or using CuArrays. Note that get_backend is used on the first variable in expr (a in this example).\n\n\n\n\n\n","category":"macro"}]
}
