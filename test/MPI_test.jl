# test/MPI_test.jl
using MPI
using Test

@testset "WaterLilyMPIExt.jl" begin
    n = 2  # number of processes
    run(`$(mpiexec()) -n $n $(Base.julia_cmd()) [...]/01-hello.jl`)
    # alternatively:
    # p = run(ignorestatus(`$(mpiexec()) ...`))
    # @test success(p)

    """Flow around a circle"""
    function circle(n,m,center,radius;Re=250,U=1,psolver=Poisson)
        body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
        Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, psolver=psolver)
    end

    # local grid size
    nx = 2^8
    ny = 2^7

    # init the MPI grid and the simulation
    r = init_mpi((nx,ny))
    sim = circle(nx,ny,SA[ny,ny],nx/8)

    (me()==0) && println("nx=$nx, ny=$ny")

    # check global coordinates
    x1 = loc(0,CartesianIndex(3,3))
    x2 = loc(CartesianIndex(3,3,1))
    println("I am rank $r, at global coordinate $x1 and $x2")

    # first we check simple rank matrix
    sim.flow.σ .= NaN
    sim.flow.σ[inside(sim.flow.σ)] .= me() #reshape(collect(1:length(inside(sim.flow.σ))),size(inside(sim.flow.σ)))
    save("sigma_1_$(me()).jld2", "C", sim.flow.σ)
    # updating halos
    WaterLily.perBC!(sim.flow.σ,())
    save("sigma_2_$(me()).jld2", "C", sim.flow.σ)

    # test global sdf
    global_loc_function(i,x) = x[i]
    sim.flow.σ .= NaN
    apply!(global_loc_function,sim.flow.μ₀)
    # check that the measure uses the correct loc function
    measure_sdf!(sim.flow.σ,sim.body,0.0)
    save("sdf_3_$(me()).jld2", "C", sim.flow.σ)
    # updating the halos here
    WaterLily.perBC!(sim.flow.σ,())
    save("sdf_4_$(me()).jld2", "C", sim.flow.σ)

    # test on vector field
    measure!(sim,0.0)
    save("mu0_1_$(me()).jld2", "C", sim.flow.μ₀[:,:,1])
    save("mu0_2_$(me()).jld2", "C", sim.flow.μ₀[:,:,2])

    #try a momentum step
    sim = circle(nx,ny,SA[ny,ny],nx/8)
    mom_step!(sim.flow,sim.pois)
    me()==0 && println("mom_step! with $(sim.pois.n) MG iters $(typeof(sim.pois))")
    save("mom_step_$(me())_p.jld2","C",sim.flow.p)
    save("mom_step_$(me())_u1.jld2","C",sim.flow.u[:,:,1])
    save("mom_step_$(me())_u2.jld2","C",sim.flow.u[:,:,2])

    # test norm functions
    sim.pois.r .= 0.0
    me() == 2 && (sim.pois.r[32,32] = 123.456789) # make this the only non-zero element
    println("L∞(pois) ($(me())): $(WaterLily.L∞(sim.pois)) true : 123.456789")
    println("L₂(pois) ($(me())) : $(WaterLily.L₂(sim.pois)) true : $(123.456789^2)")
end