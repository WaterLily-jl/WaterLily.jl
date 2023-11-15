using CUDA,StaticArrays
N=128
ω_cu = ntuple(j->ntuple(i->(m=N÷2^(i-1);CUDA.rand(m,m,m)),5),3);
u_cu = CUDA.zeros(N,N,N);
ω = ntuple(j->ntuple(i->(m=N÷2^(i-1);rand(Float32,m,m,m)),5),3);
u = Array(u_cu);

using WaterLily
function _u_ω(x,l,R,biotsavart,s=0f0)
    while l>1
        dx = 2f0^(l-1)
        Rclose = inR(x/dx .- 1,R):inR(x/dx .+ 1,R)
        for I ∈ R
            !(I ∈ Rclose) && (s += biotsavart(r(x,I,dx),I,l))
        end
        l -= 1
        R = (2first(Rclose)-oneunit(first(Rclose))):2last(Rclose)
    end
    for I ∈ R
        s += biotsavart(r(x,I),I,l)
    end
    return s/Float32(4π)
end
u_ω(i,I::CartesianIndex{3},ω) = _u_ω(loc(i,I,Float32),lastindex(ω[1]),CartesianIndices(ω[1][end]),
    @inline (r,I,l) -> WaterLily.permute((j,k)->@inbounds(ω[k][l][I]*r[j]),i)/√sum(abs2,r)^3)
r(x,I::CartesianIndex,dx=1) = x-dx*(SA_F32[I.I...] .- 1.5f0)#WaterLily.loc(0,I,Float32)
inR(x,R) = clamp(CartesianIndex(round.(Int,x .+ 1.5f0)...),R)
Base.clamp(I::CartesianIndex,R::CartesianIndices) = CartesianIndex(clamp.(I.I,first(R).I,last(R).I))

R = CartesianIndices((1:N,1:1,1:N));
u_ω(2,CartesianIndex(96,1,64),ω)
u[R] .= u_ω.(Ref(2),R,Ref(ω));
u_ω_loop(u,i,ω,R) = WaterLily.@loop u[I]=u_ω(i,I,ω) over I ∈ R

using BenchmarkTools
@btime $u[$R] .= u_ω.(Ref(2),$R,Ref($ω));
@btime u_ω_loop($u,2,$ω,$R);
@btime CUDA.@sync u_ω_loop($u_cu,2,$ω_cu,$R);