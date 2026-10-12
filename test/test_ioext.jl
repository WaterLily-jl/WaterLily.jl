using ReadVTK, WriteVTK, JLD2

function sphere_sim(radius = 8; D=2, mem=Array, exitBC=false)
    body = AutoBody((x,t)-> √sum(abs2,x .- (2radius+1.5)) - radius)
    D==2 && return Simulation(radius.*(6,4),(1,0),radius; body, ν=radius/250, T=Float32, mem, exitBC)
    Simulation(radius.*(6,4,1),(1,0,0),radius; body, ν=radius/250, T=Float32, mem, exitBC)
end
@testset "VTKExt.jl" begin
    for D ∈ [2,3], mem ∈ arrays
        # make a simulation
        sim = sphere_sim(;D,mem);
        # make a vtk writer, with the curl at the cell corners in 2D
        curl3(sim) = (a = sim.flow.σ; @inside a[I] = WaterLily.curl(3,I,sim.flow.u); Array(a))
        attrib = D==2 ? merge(default_attrib(), Dict("curl"=>curl3=>:corner)) : default_attrib()
        wr = vtkWriter("test_vtk_reader_$D";dir="TEST_DIR",attrib)
        sim_step!(sim,1); save!(wr, sim); close(wr)
        # fields are cell data, and point data padded to N+1 points at the corners
        vtk = VTKFile("TEST_DIR/test_vtk_reader_$(D)_000000.vti")
        @test issubset(("Pressure","Velocity"), keys(get_cell_data(vtk)))
        D==2 && @test WaterLily.squeeze(get_data_reshaped(get_point_data(vtk)["curl"]))[1:end-1,1:end-1] == curl3(sim)
        @test_throws ArgumentError save!(vtkWriter("test_vtk_bad";dir="TEST_DIR",attrib=Dict("u"=>curl3=>(:face,1))), sim)

        # re start the sim from a paraview file
        restart = sphere_sim(;D,mem);
        load!(restart; fname="test_vtk_reader_$D.pvd")

        # check that the restart is the same as the original
        @test all(sim.flow.p .== restart.flow.p)
        @test all(sim.flow.u .== restart.flow.u)
        @test all(sim.flow.μ₀ .== restart.flow.μ₀)
        @test sim.flow.Δt[end] == restart.flow.Δt[end]
        @test abs(sim_time(sim)-sim_time(restart))<1e-3

        # re start from the point data of older versions
        vtk = vtk_grid("TEST_DIR/test_vtk_old_$D", [1:n for n in size(sim.flow.p)]...)
        vtk["Pressure"] = Array(sim.flow.p); vtk["Velocity"] = permutedims(Array(sim.flow.u),[D+1,1:D...])
        pvd = paraview_collection("test_vtk_old_$D"); pvd[sim_time(sim)] = vtk; vtk_save(pvd)
        restart = sphere_sim(;D,mem); load!(restart; fname="test_vtk_old_$D.pvd")
        @test all(sim.flow.p .== restart.flow.p) && all(sim.flow.u .== restart.flow.u)

        # clean-up
        @test_nowarn rm("TEST_DIR",recursive=true)
        @test_nowarn rm("test_vtk_reader_$D.pvd"); @test_nowarn rm("test_vtk_old_$D.pvd")
    end
end

@testset "WaterLilyJLD2Ext.jl" begin
    test_dir = "TEST_DIR"; mkpath(test_dir)
    for D ∈ [2,3], mem ∈ arrays
        sim1 = sphere_sim(;D,mem)
        sim_step!(sim1, 1)
        save!("sim1_sphere.jld2", sim1; dir=test_dir)

        sim2 = sphere_sim(;D,mem)
        load!(sim2; fname="sim1_sphere.jld2", dir=test_dir)

        @test all(sim1.flow.p .== sim2.flow.p)
        @test all(sim1.flow.u .== sim2.flow.u)
        @test all(sim1.flow.Δt .== sim2.flow.Δt)

        # temporal averages
        sim = make_bl_flow(; T=Float32, mem)
        meanflow1 = MeanFlow(sim.flow; uu_stats=true)
        for t in range(0,10;step=0.1)
            sim_step!(sim, t)
            update!(meanflow1, sim.flow)
        end
        save!("meanflow.jld2", meanflow1; dir=test_dir)
        meanflow2 = MeanFlow(sim.flow; uu_stats=true)
        WaterLily.reset!(meanflow2)
        load!(meanflow2; fname="meanflow.jld2", dir=test_dir)
        @test all(meanflow1.U .== meanflow2.U)
        @test all(meanflow1.P .== meanflow2.P)
        @test all(meanflow1.UU .== meanflow2.UU)
        @test all(meanflow1.t .== meanflow2.t)
    end
    @test_nowarn rm(test_dir, recursive=true)
end
