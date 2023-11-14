using CUDA,StaticArrays
N=128
ω_cu = ntuple(i->(m=N÷2^(i-1);CUDA.rand(m,m,m)),5);
u_cu = CUDA.zeros(N,N,N);
ω = ntuple(i->(m=N÷2^(i-1);rand(Float32,m,m,m)),5);
u = Array(u_cu);

import WaterLily
function u_ω(j,J,ω)
    x = WaterLily.loc(j,J,Float32)
    s = 0f0
    R = CartesianIndices(ω[end])
    for l ∈ lastindex(ω):-1:2
        dx = 2f0^(l-1)
        Rclose = inR(x/dx .- 1,R):inR(x/dx .+ 1,R)
        for I ∈ R
            !(I ∈ Rclose) && (rI = r(x,I,dx); s += ω[l][I]*rI[j%3+1]/√sum(abs2,rI)^3)
        end
        R = (2first(Rclose)-oneunit(J)):2last(Rclose)
    end
    for I ∈ R
        rI = r(x,I); s += ω[1][I]*rI[j%3+1]/√sum(abs2,rI)^3
    end
return s/Float32(4π)
end
r(x,I::CartesianIndex,dx=1) = x-dx*(SA_F32[I.I...] .- 1.5f0)#WaterLily.loc(0,I,Float32)
inR(x,R) = clamp(CartesianIndex(round.(Int,x .+ 1.5f0)...),R)
Base.clamp(I::CartesianIndex,R::CartesianIndices) = CartesianIndex(clamp.(I.I,first(R).I,last(R).I))

R = CartesianIndices((1:N,1:1,1:N));
u_ω(2,CartesianIndex(96,1,64),ω)
u[R] .= u_ω.(Ref(2),R,Ref(ω));

using KernelAbstractions
@kernel function kern(u,@Const(j),@Const(ω))
    J = @index(Global,Cartesian)
    u[J]=u_ω(j,J,ω)
end
u_ω_kern(u,j,ω,R) = kern(get_backend(u),64)(u,j,ω,ndrange=size(R))
u_ω_kern(u_cu,2,ω_cu,R);

using BenchmarkTools
@btime $u[$R] .= u_ω.(Ref(2),$R,Ref($ω));
@btime u_ω_kern($u,2,$ω,$R);
@btime CUDA.@sync u_ω_kern($u_cu,2,$ω_cu,$R); 