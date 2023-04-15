using WaterLily
using BenchmarkTools
using CUDA: CuArray,@sync

using StaticArrays
function get_flow(N,f)
    a = Flow((N,N),(1.,0.);f,T=Float32)
    sdf(x,t) = √sum(abs2,x.-N÷2)-N÷4
    map(x,t) = x.-SVector(t,0)
    body = AutoBody(sdf,map)
    return a,body
end

# SERIAL BASELINE: 14.901 ms (0 allocations: 0 bytes)
a,body = get_flow(2^10,Array);
@btime @sync measure!($a,$body) 
# 4.093 ms (5998 allocations: 490.53 KiB) 4x speed-up
a,body = get_flow(2^10,CuArray);
@btime @sync measure!($a,$body) 
# 1.469 ms (2770 allocations: 120.28 KiB) 10x speed-up