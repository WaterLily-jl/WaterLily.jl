function parse_cla(args; log2p=(2,3,4), max_steps=10, ftype=Float32, backend=Array)
    iarg(arg) = occursin.(arg, args) |> findfirst
    parse_tuple(T, s) = Tuple(parse.(T, split(strip(s, ['(', ')', ' ']), ','; keepempty=false)))
    arg_value(arg) = split(args[iarg(arg)], "=")[end]

    log2p = !isnothing(iarg("log2p")) ? arg_value("log2p") |> x -> parse_tuple(Int, x) : log2p
    max_steps = !isnothing(iarg("max_steps")) ? arg_value("max_steps") |> x -> parse(Int, x) : max_steps
    ftype = !isnothing(iarg("ftype")) ? arg_value("ftype") |> x -> eval(Symbol(x)) : ftype
    backend = !isnothing(iarg("backend")) ? arg_value("backend") |> x -> eval(Symbol(x)) : backend
    return log2p, max_steps, ftype, backend
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

function add_to_suite!(suite, sim_function; log2p=(3,4,5), max_steps=max_steps, ftype=Float32, backend=Array)
    bstr = backend_str[backend]
    suite[bstr] = BenchmarkGroup([bstr])
    for n in log2p
        sim = sim_function(n, backend; T=ftype)
        suite[bstr][repr(n)] = BenchmarkGroup([repr(n)])
        @add_benchmark sim_step!($sim, $typemax(ftype); max_steps=$max_steps, verbose=false, remeasure=false) $(get_backend(sim.flow.p)) suite[bstr][repr(n)] "sim_step!"
    end
end

git_hash = read(`git rev-parse --short HEAD`, String) |> x -> strip(x, '\n')

