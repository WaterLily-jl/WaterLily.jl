using LinearAlgebra
"""
Biot-Savart integral
"""
function BiotSavart!(u,ω)
    N,n = WaterLily.size_u(u)
    u .= 0.0
    for i ∈ 1:n
        j=i%2+1 # the only component not zero in the vorticity
        for I ∈ WaterLily.inside_u(N,i)
            @WaterLily.loop u[I,i] += K(i,I,j,J)*ω[J] over J ∈ WaterLily.inside_u(N,i)
        end
    end
end

"""
Biot-Savart integral
"""
function BiotSavart!(Is,u,ω)
    N,n = WaterLily.size_u(u)
    for i ∈ 1:n
        j=i%2+1 # the only component not zero in the vorticity
        for I ∈ Is
            u[I,i] = 0.0
            @WaterLily.loop u[I,i] += K(i,I,j,J)*ω[J] over J ∈ WaterLily.inside_u(N,i)
        end
    end
end

"""
Biot-Savart kernel
"""
function K(i,I,j,J,ϵ=1e-6)
    # face centered velocity at I due to vorticity at cell edge J
    r = loc(i,I) .- loc(0,J) .+ 0.5 # cell edge vorticity
    rⁿ = norm(r)^2
    return sign(i-j)*r[j]/(2π*rⁿ+ϵ^2)
end

function error_at_slice(ue,u,dn,up)
    N,n = WaterLily.size_u(u)
    error = 0.0
    for i ∈ 1:n, s ∈ [up,dn]
        for I ∈ WaterLily.slice(N,s,2)
            error += abs.(u[I,i] - ue[I,i])
        end
    end
    return error / (2N[1])
end

function velocity_at_slice(ue,dn,up)
    res = zeros(eltype(ue),size(u))
    N,n = WaterLily.size_u(u)
    for i ∈ 1:n, s ∈ [up,dn]
        for I ∈ WaterLily.slice(N,s,2)
            res[I,i] = ue[I,i]
        end
    end
    return res
end

function ∮(a::AbstractArray{T},j) where T
    N,n = WaterLily.size_u(a)
    return ∮(a,N,N[j],j)
end

function ∮(a,N,s,j) 
    sm = 0.0
    for I ∈ WaterLily.slice(N.-1,s,j,2) # remove ghosts
        sm += a[I,j] 
    end
    return sm
end
