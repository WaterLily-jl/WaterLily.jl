using ReadVTK, WriteVTK, JLD2

function sphere_sim(radius = 8; D=2, mem=Array, exitBC=false)
    body = AutoBody((x,t)-> √sum(abs2,x .- (2radius+1.5)) - radius)
    Simulation(radius.*(6,4,1)[1:D],(1,0,0)[1:D],radius; body, ν=radius/250, T=Float32, mem, exitBC)
end
@testset "VTKExt.jl" begin
    for D ∈ [2,3], mem ∈ arrays
        # make a simulation
        sim = sphere_sim(;D,mem);
        # make a vtk writer, also writing the pressure at the cell corners to check the layout
        pc(sim) = Array(sim.flow.p)
        wr = vtkWriter("test_vtk_reader_$D";dir="TEST_DIR",attrib=merge(default_attrib(),Dict("pc"=>pc=>:corner)))
        sim_step!(sim,1); save!(wr, sim); close(wr)
        # fields are cell data, and point data padded to N+1 points at the corners
        vtk = VTKFile("TEST_DIR/test_vtk_reader_$(D)_000000.vti")
        @test issubset(("Pressure","Velocity"), keys(get_cell_data(vtk)))
        @test WaterLily.squeeze(get_data_reshaped(get_point_data(vtk)["pc"]))[CartesianIndices(sim.flow.p)] == pc(sim)
        @test_throws ArgumentError save!(vtkWriter("test_vtk_bad";dir="TEST_DIR",attrib=Dict("u"=>pc=>(:face,1))), sim)

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
            WaterLily.update!(meanflow1, sim.flow) # CUDA 6.5 also exports an update!
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

lin(y, c) = sum(Tuple(c); init=1f0) + sum(y .* (0.5f0, 0.25f0, 0.125f0)[1:length(y)])
offset(c, N) = SVector{N,Float32}(count(==(d), Tuple(c))/2max(1, length(c)) for d in 1:N)
function fill_lin!(a, N)
    h = Array(a)
    for c in CartesianIndices(size(h)[N+1:end]), I in CartesianIndices(size(h)[1:N])
        h[I, c] = lin(WaterLily.loc(0, I, Float32) - offset(c, N), c)
    end
    copyto!(a, h)
end
function regridded(a, n, scale, N)
    h = similar(Array(a))
    for c in CartesianIndices(size(h)[N+1:end]), I in CartesianIndices(size(h)[1:N])
        s = offset(c, N)
        h[I, c] = lin(clamp.((WaterLily.loc(0, I, Float32) - s) / scale + s, 0, SVector(n) .- 2) - s, c)
    end
    h
end

@testset "WaterLilyJLD2Ext.jl regrid" begin
    test_dir = "TEST_DIR"; mkpath(test_dir)
    for D ∈ [2,3], mem ∈ arrays
        m, scale = (16,8,8)[1:D], 2
        n = m .+ 2
        sim1 = Simulation(m, ntuple(i -> i==1, D), 4; T=Float32, mem)
        sim2 = Simulation(scale .* m .+ (16,8,8)[1:D], ntuple(i -> i==1, D), 4scale; T=Float32, mem)
        fill_lin!(sim1.flow.p, D); fill_lin!(sim1.flow.u, D)
        empty!(sim1.flow.Δt); push!(sim1.flow.Δt, 0.3f0, 0.2f0, 0.25f0)
        save!("lin.jld2", sim1; dir=test_dir)
        load!(sim2.flow; fname="lin.jld2", dir=test_dir, scale)
        @test Array(sim2.flow.p) ≈ regridded(sim2.flow.p, n, scale, D)
        @test Array(sim2.flow.u) ≈ regridded(sim2.flow.u, n, scale, D)
        @test sim_time(sim2) ≈ sim_time(sim1)
        @test sim2.flow.Δt[end] == WaterLily.CFL(sim2.flow)

        mean1 = MeanFlow(sim1.flow; uu_stats=true)
        fill_lin!(mean1.P, D); fill_lin!(mean1.U, D); fill_lin!(mean1.UU, D)
        push!(mean1.t, 1f0)
        save!("linmean.jld2", mean1; dir=test_dir)
        mean2 = MeanFlow(sim2.flow; uu_stats=true)
        load!(mean2; fname="linmean.jld2", dir=test_dir, scale)
        @test Array(mean2.P) ≈ regridded(mean2.P, n, scale, D)
        @test Array(mean2.U) ≈ regridded(mean2.U, n, scale, D)
        @test Array(mean2.UU) ≈ regridded(mean2.UU, n, scale, D)
        @test mean2.t == scale .* mean1.t
    end
    @test_nowarn rm(test_dir, recursive=true)
end
