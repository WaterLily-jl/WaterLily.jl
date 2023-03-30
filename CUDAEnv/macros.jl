using KernelAbstractions,CUDA,CUDA.CUDAKernels,Adapt,OffsetArrays
using OffsetArrays: Origin
CUDA.allowscalar(false)

N = (4+2,3+2); D = length(N)
Nd = (N...,D); Od = (zeros(Int,D)...,1)
u = adapt(CuArray,Origin(Od)(rand(Float32,Nd)));
div = adapt(CuArray,Origin(0)(zeros(Float32,N)));
div2 = copy(div);
 
@inline δ(i,I::CartesianIndex{m}) where{m} = CartesianIndex(ntuple(j -> j==i ? 1 : 0, m))
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@fastmath @inline function div_operator(I::CartesianIndex{m},u) where {m} 
    init=zero(eltype(u))
    for i in 1:m
     init += @inbounds ∂(i,I,u)
    end
    return init
end

# write kernel by hand and test
@kernel function div_kernel(div, u)
    I = @index(Global, Cartesian)
    div[I] = div_operator(I,u)
end
div_hand(div,u) = div_kernel(CUDABackend(),64)(div,u,ndrange=size(div).-2)
div_hand(div,u) # works

# Automate writing the kernel from the expression
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args; sym = []
    grab!(sym,ex)
    setdiff!(sym,[I]) # don't want to pass I as an argument
    @gensym kernel
    return quote
        @kernel function $kernel($(sym...))
            $I = @index(Global, Cartesian)
            $ex
        end;
        $kernel(CUDABackend(),64)($(sym...),ndrange=$R)
    end |> esc
end
# Grab all the arguments from an expression
function grab!(sym,ex::Expr)
    start = ex.head==:(call) ? 2 : 1
    for args in ex.args[start:end]
        grab!(sym,args)
    end
end
@inline grab!(sym,ex::Symbol) = union!(sym,[ex])
@inline grab!(sym,ex) = nothing

# Test
@macroexpand1 @loop a[I,i] = coeff[I,i]*grad(i,I,p) over I ∈ size(p).-2

# Automate the range for assignment loops
macro inside(ex)
    # Make sure its a single assignment
    @assert ex.head == :(=) && ex.args[1].head == :(ref)
    a,I = ex.args[1].args[1:2]
    return quote # loop over the size of the reference
        @loop $ex over $I ∈ size($a).-2
    end |> esc
end

# Test against the hand-written kernel
@macroexpand1 @inside div2[I] = div_operator(I,u)
@inside div2[I] = div_operator(I,u)
div.parent ≈ div2.parent

# Benchmark against the hand-written kernel
using BenchmarkTools
N = (2^10+2,2^10+2); D = length(N)
Nd = (N...,D); Od = (zeros(Int,D)...,1)
u = adapt(CuArray,Origin(Od)(rand(Float32,Nd)));
div = adapt(CuArray,Origin(0)(zeros(Float32,N)));

@btime div_hand($div,$u) # 2.389 μs (52 allocations: 2.80 KiB)
# div_macro also creates the function - so it's not a fair comparison
div_macro(div,u) = @inside div[I] = div_operator(I,u)
@btime div_macro($div,$u) # 2.700 μs (62 allocations: 2.94 KiB) 
# still close