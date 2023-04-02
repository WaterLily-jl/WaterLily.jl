using KernelAbstractions,CUDA,CUDA.CUDAKernels,Adapt,OffsetArrays
using OffsetArrays: Origin
CUDA.allowscalar(false)

struct Flow{Scal,Vect}
    u :: Vect
    σ :: Scal
    function Flow(N::NTuple{D}; f=identity, T = Float32) where D
        Nd = (N...,D); Od = (zeros(Int,D)...,1)
        u = rand(T,Nd)|>Origin(Od)|>f
        σ = zeros(T,N)|>Origin(0)|>f
        new{typeof(σ),typeof(u)}(u,σ)
    end
end
Base.size(a::Flow) = size(a.σ).-2

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
div_hand!(flow::Flow) = div_kernel(KernelAbstractions.get_backend(flow.σ),64)(flow.σ,flow.u,ndrange=size(flow.σ).-2)

flow = Flow((6,6));
div_hand!(flow) # works

# Automate writing the kernel from the expression
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args; sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    return quote
        @kernel function f($(rep.(sym)...)) # replace composite arguments
            $I = @index(Global, Cartesian)
            $ex
        end
        f(KernelAbstractions.get_backend($(sym[1])),64)($(sym...),ndrange=$R)
        return nothing
    end |> esc
end
function grab!(sym,ex::Expr)
    ex.head == :. && return union!(sym,[ex])    # keep composited names without recursion
    start = ex.head==:(call) ? 2 : 1            # don't grab function names
    foreach(a->grab!(sym,a),ex.args[start:end]) # recurse
    ex.args .= rep.(ex.args)                    # replace composite names with value
end
grab!(sym,ex::Symbol) = union!(sym,[ex])        # keep symbol names
grab!(sym,ex) = nothing
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex

# Automate the range for assignment loops
macro inside(ex)
    # Make sure its a single assignment
    @assert ex.head == :(=) && ex.args[1].head == :(ref)
    a,I = ex.args[1].args[1:2]
    return quote # loop over the size of the reference
        @loop $ex over $I ∈ size($a).-2
    end |> esc
end
@macroexpand @inside flow.σ[I] = div_operator(I,flow.u)
divergence!(flow) = @inside flow.σ[I] = div_operator(I,flow.u)

# Test against the hand-written kernel
div = copy(flow.σ); fill!(flow.σ,0); 
divergence!(flow) # works
flow.σ == div # true

# Benchmark against the hand-written kernel
using BenchmarkTools
flowCPU = Flow((2^10+2,2^10+2));
flowGPU = Flow((2^10+2,2^10+2),f=a->adapt(CuArray,a));

@btime div_hand!($flowCPU) # 111.000 μs (200 allocations: 17.94 KiB)
@btime div_hand!($flowGPU) # 2.178 μs (51 allocations: 2.64 KiB)
@btime divergence!($flowCPU); # 122.200 μs (209 allocations: 18.22 KiB) 
@btime divergence!($flowGPU); # 2.700 μs (62 allocations: 2.94 KiB) 