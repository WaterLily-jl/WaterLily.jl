@testset "parallel.jl" begin
    # Serial defaults of the AbstractParMode dispatch hooks (the MPI extension's methods are
    # exercised by test/mpitests.jl)
    @test WaterLily.par_mode[] isa WaterLily.Serial
    @test mpi_rank() == 0 && mpi_comm() === nothing && mpi_nprocs() == 1
    @test global_allreduce(3) === 3 && global_sum([1,2,3]) == 6 && global_dot([1,2],[3,4]) == 11
    @test global_min(1,2) == 1 && global_max(5) == 5 && global_length(1:4) == 4
    @test WaterLily.phys_left(1) && WaterLily.phys_right(2)
    @test !WaterLily.decomposed(1) && WaterLily.effective_perdir((1,2)) == (1,2)
    @test global_offset(Val(2)) == zeros(SVector{2,Float32}) && WaterLily._loop_offset(Float64) === nothing
    @test mg_maxlevels((64,64)) == 10
    @test scalar_halo!(zeros(4,4)) === nothing && velocity_halo!(zeros(4,4,2)) === nothing

    # @distributed rewrite: `(local,_,_) = init_waterlily_mpi(dims; perdir=…); f(local, …)` for ANY
    # constructor call `f(dims, …)` (dims = first positional argument), with `perdir` found in the
    # `; perdir=…`, `; perdir` and `, perdir=…` keyword forms.
    function rewrite(ex)
        r = Base.remove_linenums!(WaterLily._rewrite_distributed_call(ex))
        init, call = r.args                       # (local,_,_) = init…(dims; perdir=…) ; f(local,…)
        initcall = init.args[2]
        (dims = initcall.args[3], perdir = initcall.args[2].args[1].args[2],
         local_sym = init.args[1].args[1], call = call)
    end
    r = rewrite(:(Simulation((8,4), (1,0), 2; ν=0.1, perdir=(1,))))
    @test r.dims == :((8,4)) && r.perdir == :((1,)) && r.call.args[1] == :Simulation
    @test r.call.args[3] === r.local_sym && r.call.args[4:5] == [:((1,0)), 2]
    @test r.call.args[2] == Expr(:parameters, Expr(:kw, :ν, 0.1), Expr(:kw, :perdir, :((1,))))  # kwargs kept
    r = rewrite(:(BiotSimulation((8,4), (1,0), 2, perdir=(2,))))                          # comma form
    @test r.dims == :((8,4)) && r.perdir == :((2,)) && r.call.args[1] == :BiotSimulation
    @test r.call.args[2] === r.local_sym && r.call.args[end] == Expr(:kw, :perdir, :((2,)))
    r = rewrite(:(WaterLily.Simulation(dims, U, L; perdir)))                                # shorthand
    @test r.dims == :dims && r.perdir == :perdir && r.call.args[1] == :(WaterLily.Simulation)
    r = rewrite(:(Foo(dims)))                                                              # no perdir
    @test r.dims == :dims && r.perdir == :(()) && r.call == Expr(:call, :Foo, r.local_sym)
    @test_throws ErrorException WaterLily._rewrite_distributed_call(:(sim.flow))
    @test_throws ErrorException WaterLily._rewrite_distributed_call(:(Simulation()))
    @test_throws ErrorException WaterLily._rewrite_distributed_call(:(Simulation(; perdir=(1,))))
    # both placements of the assignment
    ex = Base.remove_linenums!(@macroexpand @distributed sim = Simulation((8,4), (1,0), 2))
    @test ex.head == :(=) && ex.args[1] == :sim && ex.args[2].head == :block
    ex = Base.remove_linenums!(@macroexpand @distributed Simulation((8,4), (1,0), 2))
    @test ex.head == :block && ex.args[1].args[2].args[1] == :init_waterlily_mpi
end
