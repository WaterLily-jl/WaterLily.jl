using CUDA,StaticArrays
N=128
ω_cu = ntuple(j->ntuple(i->(m=N÷2^(i-1)+2;CUDA.rand(m,m,m)),5),3);
u_cu = CUDA.zeros(N+2,N+2,N+2);
ω = ntuple(j->ntuple(i->(m=N÷2^(i-1)+2;rand(Float32,m,m,m)),5),3);
u = Array(u_cu);

using WaterLily
import WaterLily: up,@loop,permute
function _u_ω(x,dis,l,R,biotsavart,s=0f0)
    while l>1
        # find Region close to x
        dx = 2f0^(l-1)
        Rclose = inR(x/dx .- dis,R):inR(x/dx .+ dis,R)

        # get contributions outside Rclose
        for I ∈ R
            !(I ∈ Rclose) && (s += biotsavart(r(x,I,dx),I,l))
        end

        # move "up" one level within Rclose
        l -= 1
        R = first(up(first(Rclose))):last(up(last(Rclose)))
    end

    # top level contribution
    for I ∈ R
        s += biotsavart(r(x,I),I,l)
    end
    return s
end
u_ω(i,I::CartesianIndex{3},ω) = _u_ω(loc(i,I,Float32),1,lastindex(ω[1]),inside(ω[1][end]),
    @inline (r,I,l) -> permute((j,k)->@inbounds(ω[k][l][I]*r[j]),i)/√sum(abs2,r)^3)/Float32(4π)
r(x,I::CartesianIndex,dx=1) = x-dx*(SA_F32[I.I...] .- 1.5f0) # faster than loc(0,I,Float32)
inR(x,R) = clamp(CartesianIndex(round.(Int,x .+ 1.5f0)...),R)
Base.clamp(I::CartesianIndex,R::CartesianIndices) = CartesianIndex(clamp.(I.I,first(R).I,last(R).I))

R = CartesianIndices((1:N+2,1:N+2,2:2))
u_ω(2,CartesianIndex(96,1,64),ω)
u[R] .= u_ω.(Ref(2),R,Ref(ω));
u_ω_loop(u,i,ω,R) = @loop u[I]=u_ω(i,I,ω) over I ∈ R

using BenchmarkTools
@btime $u[$R] .= u_ω.(Ref(2),$R,Ref($ω));
@btime u_ω_loop($u,2,$ω,$R);
@btime CUDA.@sync u_ω_loop($u_cu,3,$ω_cu,$R);
