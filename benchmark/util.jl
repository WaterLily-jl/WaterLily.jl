function parse_cla(args; log2n=(2,3,4), t_end=1.0, max_steps=10, dtype=Float32, backend=Array, samples=1)
    iarg(arg) = occursin.(arg, args) |> findfirst
    parse_tuple(T, s) = Tuple(parse.(T, split(strip(s, ['(', ')', ' ']), ','; keepempty=false)))
    arg_value(arg) = split(args[iarg(arg)], "=")[end]

    log2n = !isnothing(iarg("log2n")) ? arg_value("log2n") |> x -> parse_tuple(Int, x) : log2n
    t_end = !isnothing(iarg("t_end")) ? arg_value("t_end") |> x -> parse(Float64, x) : t_end
    max_steps = !isnothing(iarg("max_steps")) ? arg_value("max_steps") |> x -> parse(Int, x) : max_steps
    dtype = !isnothing(iarg("dtype")) ? arg_value("dtype") |> x -> eval(Symbol(x)) : dtype
    backend = !isnothing(iarg("backend")) ? arg_value("backend") |> x -> eval(Symbol(x)) : backend
    samples = !isnothing(iarg("sampels")) ? arg_value("samples") |> x -> parse(Int, x) : samples
    return log2n, t_end, max_steps, dtype, backend, samples
end

macro add_benchmark(args...)
    ex, b, suite, label = args
    return quote
        $suite[$label] = @benchmarkable begin
            $ex
            synchronize($b)
        end
    end |> esc
end

backend_str = Dict(Array => "CPUx$(Threads.nthreads())", CuArray => "GPU")

function add_to_suite!(suite, sim_function; log2n=(3,4,5), t_end=t_end, max_steps=max_steps, dtype=Float32, backend=Array)
    bstr = backend_str[backend]
    suite[bstr] = BenchmarkGroup([bstr])
    for n in log2n
        sim = sim_function(n, backend; T=dtype)
        suite[bstr][repr(n)] = BenchmarkGroup([repr(n)])
        @add_benchmark sim_step!($sim, $t_end; max_steps=$max_steps, verbose=true, remeasure=false) $(get_backend(sim.flow.p)) suite[bstr][repr(n)] "sim_step!"
    end
end

git_hash() = read(`git rev-parse --short HEAD`, String) |> x -> strip(x, '\n')

