"""
WaterLilyMPIExt — Julia package extension
==========================================
Activated automatically when ImplicitGlobalGrid and MPI are loaded alongside
WaterLily.  Provides MPI-aware overrides for every function that reads arrays
with a finite-difference stencil (via δ/∂) so that ghost cells at MPI-subdomain
boundaries contain the correct neighbour values before the stencil is evaluated.

Stencil-accessing functions and required halo arrays
------------------------------------------------------
  conv_diff!(r, u⁰, ...)      → u⁰  (QUICK reads u[I±δ], u[I±2δ])
  BDIM!(a, bc)                → bc.μ₁  (μddn reads μ₁[I±δ])
  div(I, u)                   → u   (reads u[I±δ])
  project! velocity correction→ b.x (reads b.x[I±δ])
  mult(I, L, D, x)            → x   (reads x[I±δ])
  residual!(p)                → p.x (before mult())
  increment!(p)               → p.ϵ (before mult())
  pcg!(p)                     → p.ϵ (each CG iteration, before mult())
  flux_out(I, u) in CFL       → u   (reads u[I±δ])

Global reductions (must be consistent across all ranks)
---------------------------------------------------------
  L₂(p)     — global r⋅r so all ranks converge together
  CFL        — global min Δt so all ranks use the same time step
  residual!  — global mean-removal so null space is eliminated globally
  pcg!       — global dot products so CG direction is consistent
"""
module WaterLilyMPIExt

# Disable precompilation: WaterLily functions are overridden here
__precompile__(false)

using WaterLily
import WaterLily: @loop, CFL, flux_out, measure!
using ImplicitGlobalGrid
using MPI
using StaticArrays
using LinearAlgebra: ⋅

# ── Communicator ──────────────────────────────────────────────────────────────

const _comm = Ref{MPI.Comm}(MPI.COMM_WORLD)
WaterLily.set_comm!(comm::MPI.Comm) = (_comm[] = comm)

# ── Global coordinate offset ──────────────────────────────────────────────────

"""
    global_offset(Val(N), T=Float32) → SVector{N,T}

Rank-local origin in global WaterLily index space.
  offset[d] = coords[d] * (nxyz[d] - overlaps[d])  =  coords[d] * nx_loc
"""
function WaterLily.global_offset(::Val{N}, ::Type{T}=Float32) where {N,T}
    g = ImplicitGlobalGrid.global_grid()
    SVector{N,T}(ntuple(d -> T(g.coords[d] * (g.nxyz[d] - g.overlaps[d])), N))
end
WaterLily.global_offset(N::Int, T::Type=Float32) = WaterLily.global_offset(Val(N), T)

# ── Dimension helpers ─────────────────────────────────────────────────────────

# Number of active spatial dimensions (nxyz > 1) in the IGG grid.
_ndims_active() = sum(ImplicitGlobalGrid.global_grid().nxyz .> 1)

# ── Scalar halo exchange (fine grid — via IGG) ───────────────────────────────

function _scalar_halo_igg!(arr::AbstractArray)
    nd = _ndims_active()
    if ndims(arr) < 3
        arr3d = reshape(arr, size(arr)..., ntuple(_->1, 3-ndims(arr))...)
        update_halo!(arr3d; dims=ntuple(identity, nd))
    else
        update_halo!(arr; dims=ntuple(identity, nd))
    end
end

# ── Direct MPI halo exchange (any array size) ────────────────────────────────
#
# IGG pre-allocates MPI send/recv buffers sized for the registered fine grid.
# Calling update_halo! on coarse multigrid arrays produces garbage.  This
# function performs a direct MPI halo exchange using Sendrecv! with freshly
# allocated buffers sized for the actual array.  It exchanges 1-cell-wide
# slabs in each active spatial dimension:
#   • send interior cells adjacent to each face  (index 2 or N[d]-1)
#   • receive into ghost cells                   (index 1 or N[d])
# Physical boundaries (no MPI neighbour) are left untouched.

function _scalar_halo_mpi!(arr::AbstractArray{T}) where T
    g   = ImplicitGlobalGrid.global_grid()
    nd  = _ndims_active()
    N   = size(arr)
    comm = _comm[]
    for dim in 1:nd
        nleft  = g.neighbors[1, dim]   # MPI_PROC_NULL = -1 if no neighbour
        nright = g.neighbors[2, dim]
        # Slab shape: two cells thick in dimension `dim` (halowidth=2)
        slab_shape = ntuple(i -> i == dim ? 2 : N[i], ndims(arr))
        slab_CI    = CartesianIndices(slab_shape)
        send_left  = zeros(T, slab_shape)
        recv_left  = zeros(T, slab_shape)
        send_right = zeros(T, slab_shape)
        recv_right = zeros(T, slab_shape)
        # Fill send buffers: indices 3,4 (left interior) and N[dim]-3,N[dim]-2 (right interior)
        for I in slab_CI
            k = I[dim]  # 1 or 2
            J_left  = CartesianIndex(ntuple(i -> i == dim ? 2+k          : I[i], ndims(arr)))
            J_right = CartesianIndex(ntuple(i -> i == dim ? N[dim]-4+k   : I[i], ndims(arr)))
            send_left[I]  = arr[J_left]
            send_right[I] = arr[J_right]
        end
        reqs = MPI.Request[]
        # Send right interior → right neighbour; recv from left neighbour into left ghost
        if nright >= 0
            push!(reqs, MPI.Isend(send_right, comm; dest=nright, tag=dim * 10))
        end
        if nleft >= 0
            push!(reqs, MPI.Irecv!(recv_left, comm; source=nleft, tag=dim * 10))
        end
        # Send left interior → left neighbour; recv from right neighbour into right ghost
        if nleft >= 0
            push!(reqs, MPI.Isend(send_left, comm; dest=nleft, tag=dim * 10 + 1))
        end
        if nright >= 0
            push!(reqs, MPI.Irecv!(recv_right, comm; source=nright, tag=dim * 10 + 1))
        end
        MPI.Waitall(reqs)
        # Copy received data into ghost cells (indices 1,2 and N-1,N)
        if nleft >= 0
            for I in slab_CI
                k = I[dim]
                J = CartesianIndex(ntuple(i -> i == dim ? k : I[i], ndims(arr)))
                arr[J] = recv_left[I]
            end
        end
        if nright >= 0
            for I in slab_CI
                k = I[dim]
                J = CartesianIndex(ntuple(i -> i == dim ? N[dim]-2+k : I[i], ndims(arr)))
                arr[J] = recv_right[I]
            end
        end
    end
end

# ── Unified scalar halo exchange ─────────────────────────────────────────────
#
# Use IGG for fine-grid arrays (fast, pre-allocated buffers) and direct MPI
# for coarse multigrid arrays (correct for any size).

function _is_fine(arr::AbstractArray)
    g  = ImplicitGlobalGrid.global_grid()
    nd = _ndims_active()
    size(arr)[1:nd] == Tuple(g.nxyz[1:nd])
end
_is_fine(p::WaterLily.Poisson) = _is_fine(p.x)

function _scalar_halo!(arr::AbstractArray)
    if _is_fine(arr)
        _scalar_halo_igg!(arr)
    else
        _scalar_halo_mpi!(arr)
    end
end

# ── Vector (velocity-shaped) halo exchange ────────────────────────────────────
#
# update_halo! requires a single contiguous Array per call; passing SubArray
# views silently skips the exchange.  For each velocity component d, copy the
# slice to a contiguous temporary, exchange via _scalar_halo!, then write back.
#
# Works for any field with shape (spatial..., D): velocity u[N...,D],
# body mask μ₀[N...,D], body velocity V[N...,D].

function _velocity_halo!(u::AbstractArray{T,N}) where {T,N}
    D  = size(u, N)            # number of components (last dim)
    sp = ntuple(_ -> :, N-1)   # all spatial dims as Colons
    for d in 1:D
        tmp = u[sp..., d]      # contiguous copy of component d
        _scalar_halo!(tmp)     # in-place exchange via reshape view
        u[sp..., d] .= tmp
    end
end

# ── MPI-aware measure! ────────────────────────────────────────────────────────
#
# Serial measure! fills BDIM arrays on inside(σ), then BC! zeros the boundary
# faces (index 3 for normal component with 2 ghost cells per side).
# At MPI-internal left boundaries, index 3 is interior to the global domain.
# Fix: restore index 3 by re-querying the body geometry, then halo exchange.

# NoBody + ParallelBC: no geometry to measure, but the BC constructor zeroes μ₀
# at boundary faces (index 3, N-1, N for normal component).  At MPI-internal
# left boundaries, index 3 is interior and μ₀ must be 1.  Right-boundary ghost
# cells (N-1, N) are fixed by the subsequent halo exchange.
function WaterLily.measure!(::WaterLily.NoBody, bc::WaterLily.ParallelBC; kwargs...)
    g  = ImplicitGlobalGrid.global_grid()
    nd = _ndims_active()
    sz = size(bc.σ)
    for dim in 1:nd
        g.neighbors[1, dim] < 0 && continue   # physical left boundary — μ₀=0 is correct
        for I in WaterLily.slice(sz, 3, dim)
            bc.μ₀[I, dim] = one(eltype(bc.μ₀))
        end
    end
    _velocity_halo!(bc.μ₀)
end

# Restore BDIM cells destroyed by BC! at MPI-internal left boundaries.
# Called after invoke(measure!) and before halo exchange.
function _restore_mpi_bdim!(body, bc; t=zero(eltype(bc.σ)), ϵ=1)
    T  = eltype(bc.σ)
    g  = ImplicitGlobalGrid.global_grid()
    nd = _ndims_active()
    sz = size(bc.σ)
    d² = T((2 + ϵ)^2)
    for dim in 1:nd
        # Skip: no left MPI neighbour → this face is a physical boundary.
        # BC! correctly zeroed it and update_halo! will leave it untouched.
        g.neighbors[1, dim] < 0 && continue
        # This rank has a left MPI neighbour in `dim`.  BC! set
        #   bc.μ₀[slice(sz,3,dim), dim] = 0   (normal component, boundary face)
        #   bc.V[slice(sz,3,dim),   dim] = 0
        # but those cells are interior to the global domain.  Recompute.
        for I in WaterLily.slice(sz, 3, dim)
            dᵢ, _, Vᵢ = measure(body, loc(dim, I, T), t; fastd²=d²)
            bc.μ₀[I, dim] = WaterLily.μ₀(dᵢ, ϵ)
            bc.V[I, dim]   = Vᵢ[dim]
        end
    end
end

function WaterLily.measure!(body::WaterLily.AbstractBody, bc::WaterLily.ParallelBC;
                              t=zero(eltype(bc.σ)), ϵ=1, kwargs...)
    # Phase 1+2: serial BDIM fill + BC! on μ₀ and V
    invoke(WaterLily.measure!,
           Tuple{WaterLily.AbstractBody, WaterLily.AbstractBC},
           body, bc; t, ϵ, kwargs...)
    # Phase 3: repair the cells that BC! wrongly zeroed at MPI-internal left faces
    _restore_mpi_bdim!(body, bc; t, ϵ)
    # Phase 4: halo exchange
    _velocity_halo!(bc.μ₀)
    _velocity_halo!(bc.V)
    # μ₁ has shape (N..., D, D): flatten the last two dims into D² so
    # _velocity_halo! can loop over D² independent (N...) spatial slices.
    # For 2D: (Nx,Ny,2,2) → (Nx,Ny,4);  for 3D: (Nx,Ny,Nz,3,3) → (Nx,Ny,Nz,9).
    nd = _ndims_active()
    μ₁_flat = reshape(bc.μ₁, size(bc.μ₁)[1:nd]..., :)
    _velocity_halo!(μ₁_flat)
end

# ── MPI-aware mom_step! ───────────────────────────────────────────────────────
#
# Two MPI-specific halo exchanges are needed beyond what the serial code does:
#   1. Before u⁰ = u at the start: ghost cells may be stale from the previous
#      step's velocity correction (project! updates interior but not ghosts).
#   2. Between predictor and corrector: the predictor's velocity correction
#      updates interior u but leaves ghost cells stale.  The corrector's
#      conv_diff!(f, a.u, ...) reads u[I±δ] at MPI boundaries and needs
#      correct ghost cells.
#
# We cannot use invoke() because we need to intercept between predictor and
# corrector, so we replicate the serial structure with the two halo exchanges.

@fastmath function WaterLily.mom_step!(a::WaterLily.Flow{N}, b::WaterLily.AbstractPoisson,
                                        bc::WaterLily.ParallelBC;
                                        λ=WaterLily.quick, udf=nothing, kwargs...) where N
    _velocity_halo!(a.u)   # (1) correct ghost cells before u⁰ = u
    a.u⁰ .= a.u; WaterLily.scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,λ;ν=a.ν,perdir=bc.perdir)
    WaterLily.udf!(a,udf,t₀; kwargs...)
    WaterLily.accelerate!(a.f,t₀,a.g,bc.uBC)
    WaterLily.project!(a,b,bc,t₁)
    # (2) exchange ghost cells after predictor's velocity correction
    _velocity_halo!(a.u)
    # corrector u → u¹
    WaterLily.conv_diff!(a.f,a.u,a.σ,λ;ν=a.ν,perdir=bc.perdir)
    WaterLily.udf!(a,udf,t₁; kwargs...)
    WaterLily.accelerate!(a.f,t₁,a.g,bc.uBC)
    WaterLily.project!(a,b,bc,t₁;w=0.5)
    push!(a.Δt,WaterLily.CFL(a))
end

# ── Global CFL ────────────────────────────────────────────────────────────────
#
# CFL computes the maximum stable Δt from the local flux field σ.  Each rank
# sees only its own subdomain, so ranks would get different Δt without a global
# reduction, breaking the predictor/corrector scaling (dt = w*a.Δt[end]).
# Replace the local max with a global Allreduce min so all ranks agree.

function WaterLily.CFL(a::WaterLily.Flow; Δt_max=10)
    @inside a.σ[I] = WaterLily.flux_out(I, a.u)
    MPI.Allreduce(min(Δt_max, inv(maximum(a.σ) + 5a.ν)), MPI.MIN, _comm[])
end

# ── MPI-aware BC! ─────────────────────────────────────────────────────────────
#
# The serial BC! applies Dirichlet conditions at ghost cells (indices 1,2) AND
# the boundary face (index 3) for the velocity component normal to each face.
# At physical domain boundaries this is correct, but at MPI-internal left
# boundaries index 3 is interior to the global domain and must keep its value.
#
# Fix: save the normal velocity at index 3 before serial BC!, then restore it
# at faces where a left MPI neighbour exists.  Ghost cells (indices 1,2 and
# N-1,N) are overwritten by the subsequent halo exchange.

function WaterLily.BC!(u, bc::WaterLily.ParallelBC, t=0)
    g  = ImplicitGlobalGrid.global_grid()
    nd = _ndims_active()
    N  = size(u)[1:end-1]          # spatial dimensions (strip last = components)

    # Save normal velocity at index 3 for MPI-internal left boundaries
    saved = ntuple(nd) do dim
        if g.neighbors[1, dim] >= 0   # has left MPI neighbour
            eltype(u)[u[I, dim] for I in WaterLily.slice(N, 3, dim)]
        else
            nothing
        end
    end

    # Apply serial BCs (may corrupt index 3 at MPI-internal left faces)
    WaterLily.BC!(u, bc.uBC, bc.exitBC, bc.perdir, t)

    # Restore normal velocity at MPI-internal left boundaries
    for dim in 1:nd
        saved[dim] === nothing && continue
        for (k, I) in enumerate(WaterLily.slice(N, 3, dim))
            u[I, dim] = saved[dim][k]
        end
    end

    # Exchange ghost cells across MPI subdomain interfaces
    _velocity_halo!(u)
end

# ── Pressure BC ───────────────────────────────────────────────────────────────
#
# Called from project!() after solver!(b).  Exchanges pressure ghost cells so
# that the velocity-correction step reads the correct ∂p/∂x values at MPI
# subdomain interfaces:
#   @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)

function WaterLily.pressureBC!(x, bc::WaterLily.ParallelBC, b)
    _scalar_halo!(x)
end

# ── Residual with global null-space correction ────────────────────────────────
#
# mult(I, L, D, p.x) reads p.x[I±δ]: exchange p.x halos before computing r.
# The null-space correction s = sum(r)/N must be global (not per-rank) so that
# all ranks subtract the same constant and pressures remain continuous across
# subdomain interfaces.

function WaterLily.residual!(p::WaterLily.Poisson)
    WaterLily.perBC!(p.x, p.perdir)
    _scalar_halo!(p.x)
    @inside p.r[I] = ifelse(p.iD[I]==0, 0, p.z[I]-WaterLily.mult(I,p.L,p.D,p.x))
    s = MPI.Allreduce(sum(p.r),                      MPI.SUM, _comm[]) /
        MPI.Allreduce(length(WaterLily.inside(p.r)), MPI.SUM, _comm[])
    abs(s) <= 2eps(eltype(s)) && return
    @inside p.r[I] = p.r[I] - s
end

# ── Global convergence norm ───────────────────────────────────────────────────
#
# r⋅r must be summed across all ranks so the convergence criterion is global.
# Ghost cells of r are zero (residual! only writes inside cells), so the full
# dot product is equivalent to summing over inside(r) only.

WaterLily.L₂(p::WaterLily.Poisson) =
    MPI.Allreduce(p.r ⋅ p.r, MPI.SUM, _comm[])

# ── MPI-aware increment! ──────────────────────────────────────────────────────
#
# mult(I, L, D, p.ϵ) reads p.ϵ[I±δ]: exchange ϵ halos before the stencil.
# Called for both fine and coarse levels; _scalar_halo! works for any uniform array size.

function WaterLily.increment!(p::WaterLily.Poisson)
    WaterLily.perBC!(p.ϵ, p.perdir)
    _scalar_halo!(p.ϵ)
    @loop (p.r[I] = p.r[I]-WaterLily.mult(I,p.L,p.D,p.ϵ);
           p.x[I] = p.x[I]+p.ϵ[I]) over I ∈ WaterLily.inside(p.x)
end

# ── MPI-aware pcg! ────────────────────────────────────────────────────────────
#
# Changes from the serial version (applied at all multigrid levels):
#   1. Exchange ϵ halos before each mult() call (mult reads ϵ[I±δ]).
#   2. Replace local dot products with global MPI.Allreduce so all ranks follow
#      the same CG direction; otherwise each rank solves a different system and
#      pressures diverge across subdomain boundaries.
# _scalar_halo! works for any uniform array size, so coarse levels are handled too.

function WaterLily.pcg!(p::WaterLily.Poisson{T}; it=6) where T
    x, r, ϵ, z = p.x, p.r, p.ϵ, p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    rho = MPI.Allreduce(r⋅z, MPI.SUM, _comm[])
    abs(rho) < 10eps(T) && return
    for i in 1:it
        WaterLily.perBC!(ϵ, p.perdir)
        _scalar_halo!(ϵ)
        @inside z[I] = WaterLily.mult(I,p.L,p.D,ϵ)
        alpha = rho / MPI.Allreduce(z⋅ϵ, MPI.SUM, _comm[])
        (abs(alpha)<1e-2 || abs(alpha)>1e2) && return
        @loop (x[I] += alpha*ϵ[I];
               r[I] -= alpha*z[I]) over I ∈ WaterLily.inside(x)
        i == it && return
        @inside z[I] = r[I]*p.iD[I]
        rho2 = MPI.Allreduce(r⋅z, MPI.SUM, _comm[])
        abs(rho2) < 10eps(T) && return
        beta = rho2/rho
        @inside ϵ[I] = beta*ϵ[I]+z[I]
        rho = rho2
    end
end

# ── MPI-aware solver! (single-level Poisson) ─────────────────────────────────
#
# The serial solver! ends with perBC!(p.x) but no halo exchange.  Add the
# final _scalar_halo!(p.x) so that p.x ghost cells are valid if this solver
# is called directly (e.g. in tests).  In the normal simulation path, the
# halo is also exchanged by pressureBC! in project!().

function WaterLily.solver!(p::WaterLily.Poisson; tol=1e-4, itmx=1e3)
    WaterLily.residual!(p); r₂ = WaterLily.L₂(p)
    nᵖ = 0
    while nᵖ < itmx
        WaterLily.smooth!(p); r₂ = WaterLily.L₂(p); nᵖ += 1
        r₂ < tol && break
    end
    WaterLily.perBC!(p.x, p.perdir)
    _scalar_halo!(p.x)
    push!(p.n, nᵖ)
end

# ── MPI-aware divisible ───────────────────────────────────────────────────────
#
# With direct MPI halo exchange for coarse arrays, the serial coarsening
# threshold is sufficient.  Keep a slightly higher minimum (N>6) to ensure
# coarse arrays have at least a 2-cell interior for meaningful solves.

WaterLily.divisible(N::Integer) = mod(N,2)==0 && N>8

# ── MPI-aware update! (MultiLevelPoisson) ────────────────────────────────────
#
# Serial restrictL! (called from update! for every coarse level) applies
# WaterLily.BC! to the restricted conductivity, which zeroes the first interior
# cell L[slice(Na,2,dim), dim] on ALL ranks — including ranks where that face is
# interior to the global domain (left MPI neighbour exists).  This creates an
# artificial zero-conductivity wall that prevents the V-cycle coarse correction
# from propagating information across subdomain boundaries.
#
# Fix: after the serial update!, for each coarse level re-compute the restricted
# conductivity at MPI-internal left faces (by re-running WaterLily.restrictL),
# restore the Neumann ghost (index 1 ← index 2), then recompute D/iD for that
# level so the operator is correct.

function _restore_coarse_level!(L_fine, coarse::WaterLily.Poisson)
    # After restrictL!(coarse.L, L_fine, ...) → BC! zeroed coarse.L[slice(Na,3,dim),dim]
    # on ALL ranks including MPI-internal ones.  Re-compute the correct restricted
    # value for each dim where this rank has a left MPI neighbour.
    g  = ImplicitGlobalGrid.global_grid()
    nd = _ndims_active()
    Na = size(coarse.x)
    for dim in 1:nd
        g.neighbors[1, dim] < 0 && continue   # physical boundary — BC! was correct
        R = CartesianIndices(ntuple(k -> k==dim ? (3:3) : (3:Na[k]-2), nd))
        for I in R
            coarse.L[I, dim] = WaterLily.restrictL(I, dim, L_fine)
        end
    end
end

function WaterLily.update!(ml::WaterLily.MultiLevelPoisson)
    # Replicate serial update!(ml::MultiLevelPoisson) while inserting the MPI fix
    # after each restrictL! call.  We cannot use invoke() here because both the
    # serial method and this override share the same signature, causing recursion.
    WaterLily.update!(ml.levels[1])   # update!(p::Poisson) — no recursion
    for l in 2:length(ml.levels)
        WaterLily.restrictL!(ml.levels[l].L, ml.levels[l-1].L,
                             perdir=ml.levels[l-1].perdir)
        # BC! inside restrictL! zeroed MPI-internal left-boundary index-3 cells.
        # Restore the correct restricted L at those cells.
        _restore_coarse_level!(ml.levels[l-1].L, ml.levels[l])
        # Exchange L ghost cells so that MPI-internal right-boundary ghost cells
        # (and all other ghost cells) get the correct restricted values from
        # their neighbours.  Without this, L ghost cells remain zeroed by BC!,
        # making the coarse Poisson operator see artificial walls at every
        # subdomain interface.
        _velocity_halo!(ml.levels[l].L)
        WaterLily.update!(ml.levels[l])   # recompute D/iD from corrected L
    end
end

# ── MPI-aware Vcycle! ─────────────────────────────────────────────────────────
#
# With direct MPI halo exchange (_scalar_halo_mpi!) available for coarse-level
# arrays, the full V-cycle recursion is restored.  Every level (fine or coarse)
# gets proper halo exchange via _scalar_halo! which auto-selects IGG (fine) or
# direct MPI (coarse).  residual!, increment!, and pcg! all use _scalar_halo!
# unconditionally, so the coarse-level solves communicate correctly across
# subdomain boundaries.

function WaterLily.Vcycle!(ml::WaterLily.MultiLevelPoisson; l=1)
    fine, coarse = ml.levels[l], ml.levels[l+1]
    WaterLily.Jacobi!(fine)
    WaterLily.restrict!(coarse.r, fine.r)
    fill!(coarse.x, 0.)
    l+1 < length(ml.levels) && WaterLily.Vcycle!(ml; l=l+1)
    WaterLily.smooth!(coarse)
    WaterLily.prolongate!(fine.ϵ, coarse.x)
    WaterLily.increment!(fine)
end

# ── MPI-aware solver! (MultiLevelPoisson) ─────────────────────────────────────
#
# Same structure as the serial version; residual!, smooth!, L₂ already dispatch
# to our MPI overrides above.  Add the final _scalar_halo!(p.x) so that the
# velocity-correction step in project!() reads correct ∂p/∂x at MPI boundaries.
# (pressureBC! then re-exchanges, but that is harmless.)

function WaterLily.solver!(ml::WaterLily.MultiLevelPoisson; tol=1e-4, itmx=32)
    p = ml.levels[1]
    WaterLily.residual!(p); r₂ = WaterLily.L₂(p)
    nᵖ = 0
    while nᵖ < itmx
        WaterLily.Vcycle!(ml); WaterLily.smooth!(p); r₂ = WaterLily.L₂(p); nᵖ += 1
        r₂ < tol && break
    end
    WaterLily.perBC!(p.x, p.perdir)
    _scalar_halo!(p.x)
    push!(ml.n, nᵖ)
end

end # module WaterLilyMPIExt
