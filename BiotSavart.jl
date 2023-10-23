using LinearAlgebra
"""
Biot-Savart integral

Compute the biot-savart for all the components of the velocity at every point in the domain
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

Compute the biot-savart for the `i`-th component of the velocity at the `I`-th cell
"""
function BiotSavart!(i,I,u,ω)
    N,n = WaterLily.size_u(u)
    j=i%2+1 # the only component not zero in the vorticity
    u[I,i] = 0.0
    @WaterLily.loop u[I,i] += K(i,I,j,J)*ω[J] over J ∈ WaterLily.inside_u(N,i)
end

"""
Biot-Savart kernel only, must be dotted with the vorticity to give velocity `i`
at `I`
"""
function BiotSavartKernel!(i,I,κ)
    N = size(κ)
    j=i%2+1 # the only component not zero in the vorticity
    @WaterLily.loop κ[J] = K(i,I,j,J) over J ∈ WaterLily.inside_u(N,i)
end

"""
Biot-Savart kernel at `x` due to vorticity at cell edge `a[J,:]`
"""
function BiotSavart!(i,x,a::AbstractArray,b::AbstractArray,Is::CartesianIndices,ϵ=1e-6)
    j = i%2+1
    for J ∈ Is
        r = x .- b[J,:] .+ 0.5 # cell edge vorticity
        rⁿ = norm(r)^2
        a[J] = sign(i-j)*r[j]/(2π*rⁿ+ϵ^2)
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