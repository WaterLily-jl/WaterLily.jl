using FileIO, MeshIO
using GeometryBasics
using WaterLily: AbstractBody,@loop,measure
using StaticArrays

struct MeshBody{T} <: AbstractBody
    mesh  :: GeometryBasics.Mesh
    map   :: Function
    scale :: T
    bbox  :: Rect
    function MeshBody(fname;map=(x,t)->x,scale=1.0,T=Float32)
        tmp = load(fname) 
        points = GeometryBasics.Point.(tmp.position*scale) # can we specify types?
        mesh = GeometryBasics.Mesh(points,GeometryBasics.faces(tmp))
        bbox = Rect(mesh.position)
        bbox = Rect(bbox.origin.-4,bbox.widths.+8)
        new{T}(mesh,map,scale,bbox)
    end
end
"""
    locate(p,tri)

Find the closest point `x` on the triangle `tri` to the point `p`.
"""
function locate(tri::GeometryBasics.Ngon{3},p::SVector{T}) where T #5.327 ns (0 allocations: 0 bytes)
    # is point `a` closest?
    a, b, c = tri.points
    ab = b.-a
    ac = c.-a
    ap = p.-a
    d1 = sum(ab.*ap)
    d2 = sum(ac.*ap)
    # is point `a` closest?
    if ((d1 ≤ 0) && (d2 ≤ 0))
        return a
    end
    # is point `b` closest?
    bp = p.-b
    d3 = sum(ab.*bp)
    d4 = sum(ac.*bp)
    if ((d3 ≥ 0) && (d4 ≤ d3))
        return b
    end
    # is point `c` closest?
    cp = p.-c
    d5 = sum(ab.*cp)
    d6 = sum(ac.*cp)
    if ((d6 ≥ 0) && (d5 ≤ d6))
        return c
    end
    # is segment 'ab' closest?
    vc = d1*d4 - d3*d2
    if ((vc ≤ 0) && (d1 ≥ 0) && (d3 ≤ 0))
        x =  a .+ ab.*d1 ./ (d1 - d3)
        return x
    end
    #  is segment 'ac' closest?
    vb = d5*d2 - d1*d6
    if ((vb ≤ 0) && (d2 ≥ 0) && (d6 ≤ 0))
        x =  a .+ ac.*d2 ./ (d2 - d6)
        return x
    end
    # is segment 'bc' closest?
    va = d3*d6 - d5*d4
    if ((va ≤ 0) && (d4 ≥ d3) && (d5 ≥ d6))
        x =  b .+ (c .- b) .* (d4 - d3) ./ ((d4 - d3) + (d5 - d6))
        return x
    end
    # closest is interior to `abc`
    denom = one(T) / (va + vb + vc)
    v= vb*denom
    w = vc*denom
    x = a .+ ab .* v .+ ac .* w
    return x
end

# inside bbox or not
inside(bbox::Rect,x::SVector) = all(bbox.origin .≤ x) && all(x .≤ bbox.origin+bbox.widths)
rect = Rect(0,0,0,1,1,1) # origin and widths
# @assert !inside(rect,SA[0.5,1,2.5]) && inside(rect,SA[0.5,0.5,0.5])
# 
# distance to box center
dist(bbox::Rect, x::SVector) = √sum(abs2,x.-bbox.origin-0.5bbox.widths)
rect = Rect(0,0,0,1,1,1) # origin and widths
# @assert dist(rect,SA[1.0,1.0,1.0]) == √0.75
# @assert dist(rect,SA[1.5,1.0,1.0]) == √1.5
# @assert dist(rect,SA[1.5,1.5,1.0]) == √2.25
# @assert dist(rect,SA[1.5,1.5,1.5]) == √3.0

sdf(body::MeshBody,x,t;kwargs...) = measure(body.mesh,body.map(x,t),t;kwargs...)[1]

function measure(body::MeshBody,x,t;kwargs...)
    # eval d=map(x,t)-x, and n̂
    ξ = body.map(x,t) # if we are outside of the bouding box, we can measure approx
    # !inside(body.bbox,ξ) && return (dist(body.bbox,ξ),zero(x),zero(x))
    d,n = measure(body.mesh,ξ,t)

    # The velocity depends on the material change of ξ=m(x,t):
    #   Dm/Dt=0 → ṁ + (dm/dx)ẋ = 0 ∴  ẋ =-(dm/dx)\ṁ
    J = ForwardDiff.jacobian(x->body.map(x,t), x)
    dot = ForwardDiff.derivative(t->body.map(x,t), t)
    return (d,n,-J\dot)
end
using LinearAlgebra: cross
"""
    normal(tri::GeometryBasics.Ngon{3})

Return the normal vector to the triangle `tri`.
"""
function normal(tri::GeometryBasics.Ngon{3})
    n = cross(SVector(tri.points[2]-tri.points[1]),SVector(tri.points[3]-tri.points[1]))
    n/√sum(abs2,n)
end
d²(tri::GeometryBasics.Ngon{3},x) = sum(abs2,x-locate(tri,x))
center(tri::GeometryBasics.Ngon{3}) = SVector(sum(tri.points;dims=1)/3.f0...)
"""
    measure(mesh::GeometryBasics.Mesh,x,t;kwargs...)

Measure the distance `d` and normal `n` to the mesh at point `x` and time `t`.
"""
function measure(mesh::GeometryBasics.Mesh,x::SVector{T},t;kwargs...) where T
    tmp = [d²(mesh[I],x) for I in CartesianIndices(mesh)] # can we awoid this?
    idx = argmin(tmp); n = WaterLily.normal(mesh[idx])
    d = sum((x-locate(mesh[idx],x)).*n)
    d,n
end