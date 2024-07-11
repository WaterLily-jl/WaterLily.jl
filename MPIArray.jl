

struct MPIArray{T,N,V<:AbstractArray{T,N},W<:AbstractVector{T}} <: AbstractArray{T,N}
    A :: V
    send :: W
    recv :: W
    function MPIArray(::Type{T}, dims::NTuple{N, Integer}) where {T,N}
        A = Array{T,N}(undef, dims); fill!(A, zero(T))
        send, recv = zeros(T,maximum(dims)), zeros(T,maximum(dims))
        new{T,N,typeof(A),typeof(send)}(A,send,recv)
    end
    MPIArray(A::AbstractArray{T}) where T = MPIArray(T, size(A))
end
for fname in (:size, :length, :ndims, :eltype) # how to write 4 lines of code in 5...
    @eval begin
        Base.$fname(A::MPIArray) = Base.$fname(A.A)
    end
end
Base.getindex(A::MPIArray, i::Int...) = Base.getindex(A.A, i...)
Base.setindex!(A::MPIArray, v, i...)  = Base.setindex!(A.A, v, i...)


function write_h2h!(A::MPIArray, N=size(A))
    @inbounds copyto!(view(A.send[:]), view(A.A[halos(N,j)]))
end

function read_h2h!(A::MPIArray, N=size(A))
    @inbounds copyto!(view(A.A[buff()]), view(A.send[1,:]))
end

# initialize array
Nd = (10,10,2)
u = Array{Float64}(undef, Nd...) |> MPIArray
p = zeros(Base.front(Nd)...) |> MPIArray
