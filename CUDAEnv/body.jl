using WaterLily
using BenchmarkTools
using CUDA: CuArray

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
@btime measure!($a,$body) 
# 4.490 ms (5998 allocations: 490.53 KiB) 3x speed-up
a,body = get_flow(2^10,CuArray);
@btime measure!($a,$body) 
# 167.300 μs (2721 allocations: 116.97 KiB) 90x speed-up