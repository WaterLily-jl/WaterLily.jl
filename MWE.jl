using CUDA
N=128
ω_cu = ntuple(i->(m=N÷2^(i-1);CUDA.rand(m,m,m)),5);
u_cu = CUDA.zeros(N,N,N);
ω = ntuple(i->(m=N÷2^(i-1);rand(m,m,m)),5);
u = Array(u_cu);

function sumover(A,I,s=zero(eltype(first(A))))
    for Al in A
        R = CartesianIndices(Al)
        for I in (I-3oneunit(I)):(I+3oneunit(I))
            I ∈ R && (s += Al[I])
        end
    end; s
end

R = CartesianIndices((1:N,2:2,1:N));
u_cu[R] .= sumover.(Ref(ω_cu),R)

using BenchmarkTools
@benchmark CUDA.@sync u_cu[R] .= sumover.(Ref(ω_cu),R)
@benchmark u[R] .= sumover.(Ref(ω),R)
