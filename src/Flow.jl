function conv_diff!(r,u,Φ,λ::F;ν=0.1,perdir=()) where {F}
    r .= zero(eltype(r))
    N,n = size_u(u)
    for i ∈ 1:n, j ∈ 1:n
        # if it is periodic direction
        tagper = (j in perdir)
        # treatment for bottom boundary with BCs
        lowerBoundary!(r,u,Φ,ν,i,j,N,λ,Val{tagper}())
        # inner cells
        @loop (Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) - ν*∂(j,CI(I,i),u);
               r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundary!(r,u,Φ,ν,i,j,N,λ,Val{tagper}())
    end
end

"""
    accelerate!(r,t,g,U)

Accounts for applied and reference-frame acceleration using `rᵢ += g(i,x,t)+dU(i,x,t)/dt`
"""
accelerate!(r,t,::Nothing,::Union{Nothing,Tuple}) = nothing
accelerate!(r,t,f::Function) = @loop r[Ii] += f(last(Ii),loc(Ii,eltype(r)),t) over Ii ∈ CartesianIndices(r)
accelerate!(r,t,g::Function,::Union{Nothing,Tuple}) = accelerate!(r,t,g)
accelerate!(r,t,::Nothing,U::Function) = accelerate!(r,t,(i,x,t)->ForwardDiff.derivative(τ->U(i,x,τ),t))
accelerate!(r,t,g::Function,U::Function) = accelerate!(r,t,(i,x,t)->g(i,x,t)+ForwardDiff.derivative(τ->U(i,x,τ),t))

"""
    Flow{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}, Tf<:AbstractArray{T,D+2}}

Composite type for a multidimensional immersed boundary flow simulation.

Flow solves the unsteady incompressible [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) on a Cartesian grid.
Solid boundaries are modelled using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/).
The primary variables are the scalar pressure `p` (an array of dimension `D`)
and the velocity vector field `u` (an array of dimension `D+1`).
"""
struct Flow{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}} <: AbstractFlow{D,T,Sf,Vf}
    u :: Vf # velocity vector field
    u⁰:: Vf # previous velocity
    f :: Vf # force vector
    p :: Sf # pressure scalar field
    σ :: Sf # divergence scalar
    Δt:: Vector{T} # time step (stored in CPU memory)
    ν :: T # kinematic viscosity
    g :: Union{Function,Nothing} # acceleration field funciton
    function Flow(N::NTuple{D}, bc::AbstractBC; Δt=0.25, ν=0., g=nothing, uλ=nothing, mem=Array, T=Float32) where D
        Ng = N .+ 2
        Nd = (Ng..., D)
        isnothing(uλ) && (uλ = ic_function(bc.uBC))
        u = Array{T}(undef, Nd...) |> mem
        isa(uλ, Function) ? apply!(uλ, u) : apply!((i,x)->uλ[i], u)
        BC!(u, bc)
        exitBC!(u,u,0.)
        f, p, σ = zeros(T, Nd) |> mem, zeros(T, Ng) |> mem, zeros(T, Ng) |> mem
        new{D,T,typeof(p),typeof(u)}(u,copy(u),f,p,σ,T[Δt],T(ν),g)
    end
end

"""
    time(a::Flow)

Current flow time.
"""
time(a::Flow) = sum(@view(a.Δt[1:end-1]))

function project!(a::Flow{n},b::AbstractPoisson,bc::AbstractBC,t; w=1) where n
    BDIM!(a,bc)
    scale_u!(a,w)
    BC!(a.u,bc,t)
    bc.exitBC && w > 0.5 && exitBC!(a.u,a.u⁰,a.Δt[end]) # convective exit
    dt = w*a.Δt[end]
    @inside b.z[I] = div(I,a.u); b.x .*= dt # set source term & solution IC
    solver!(b)
    for i ∈ 1:n  # apply solution and unscale to recover pressure
        @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
    end
    b.x ./= dt
    BC!(a.u,bc,t)
end

"""
    mom_step!(a::Flow,b::AbstractPoisson;λ=quick,udf=nothing,kwargs...)

Integrate the `Flow` one time step using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/)
and the `AbstractPoisson` pressure solver to project the velocity onto an incompressible flow.
"""
@fastmath function mom_step!(a::Flow{N},b::AbstractPoisson,bc::AbstractBC; λ=quick,udf=nothing,kwargs...) where N
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    @log "p"
    conv_diff!(a.f,a.u⁰,a.σ,λ;ν=a.ν,perdir=bc.perdir)
    udf!(a,udf,t₀; kwargs...)
    accelerate!(a.f,t₀,a.g,bc.uBC)
    project!(a,b,bc,t₁)
    # corrector u → u¹
    @log "c"
    conv_diff!(a.f,a.u,a.σ,λ;ν=a.ν,perdir=bc.perdir)
    udf!(a,udf,t₁; kwargs...)
    accelerate!(a.f,t₁,a.g,bc.uBC)
    project!(a,b,bc,t₁;w=0.5)
    push!(a.Δt,CFL(a))
end
scale_u!(a,scale) = @loop a.u[Ii] *= scale over Ii ∈ inside_u(size(a.p))

function CFL(a::Flow;Δt_max=10)
    @inside a.σ[I] = flux_out(I,a.u)
    min(Δt_max,inv(maximum(a.σ)+5a.ν))
end
@fastmath @inline function flux_out(I::CartesianIndex{d},u) where {d}
    s = zero(eltype(u))
    for i in 1:d
        s += @inbounds(max(0.,u[I+δ(i,I),i])+max(0.,-u[I,i]))
    end
    return s
end

"""
    udf!(flow::Flow,udf::Function,t)

User defined function using `udf::Function` to operate on `flow::Flow` during the predictor and corrector step, in sync with time `t`.
Keyword arguments must be passed to `sim_step!` for them to be carried over the actual function call.
"""
udf!(flow,::Nothing,t; kwargs...) = nothing
udf!(flow,force!::Function,t; kwargs...) = force!(flow,t; kwargs...)
