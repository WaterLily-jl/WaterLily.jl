using CUDA,StaticArrays
N=128
ω_cu = ntuple(i->(m=N÷2^(i-1);CUDA.rand(m,m,m)),5);
u_cu = CUDA.zeros(N,N,N);
ω = ntuple(i->(m=N÷2^(i-1);rand(Float32,m,m,m)),5);
u = Array(u_cu);

function u_ω(J,ω)
    s = 0f0
    for ωᵢ in ω
        R = CartesianIndices(ωᵢ)
        for I ∈ (J-3oneunit(J)):(J+3oneunit(J))
            I ∈ R && (r = loc(J)-loc(I)-SA_F32[0.5,0,0]; s += ωᵢ[I]*r[1]/√sum(abs2,r)^3)
        end
    end
    return s/Float32(4π)
end
loc(I::CartesianIndex) = SA_F32[I.I...];

R = CartesianIndices((1:N,1:1,1:N));
u_cu[R] .= u_ω.(R,Ref(ω_cu));

using KernelAbstractions
@kernel function kern(u,@Const(ω))
    J = @index(Global,Cartesian)
    u[J]=u_ω(J,ω)
end
u_ω_kern(u,ω,R) = kern(get_backend(u),64)(u,ω,ndrange=size(R))
u_ω_kern(u_cu,ω_cu,R);

using BenchmarkTools
@benchmark CUDA.@sync $u_cu[$R] .= u_ω.($R,Ref($ω_cu))
@benchmark $u[$R] .= u_ω.($R,Ref($ω))
@benchmark CUDA.@sync u_ω_kern($u_cu,$ω_cu,$R)
@benchmark u_ω_kern($u,$ω,$R)