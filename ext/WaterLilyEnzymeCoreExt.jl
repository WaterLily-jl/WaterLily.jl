module WaterLilyEnzymeCoreExt

using WaterLily
using EnzymeCore
using EnzymeCore.EnzymeRules

# Discrete-adjoint rule for the multigrid Poisson solve. Forward solves Ax = z;
# reverse solves A λ = x̄ with the same multigrid (A is symmetric — face
# coefficients in L are shared by adjacent cells, and BDIM modulates μ₀ on
# faces preserving symmetry — so Aᵀ = A) and accumulates cotangents into the
# source-term and operator-coefficient shadows.
#
# The cost passed to autodiff must be invariant to constant pressure shifts
# (e.g. `sum(u)`, force integrals) — `sum(p)` has uniform cotangents that lie
# in nullspace(Aᵀ) and produces zero gradient. Tighten Poisson tolerance for
# accurate gradients (defaults stop at 1e-4; 1e-10 gives ~5 sig figs vs FD).

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Const{typeof(WaterLily.poisson_solve!)},
    ::Type{<:Const},
    p::Duplicated{<:WaterLily.MultiLevelPoisson},
)
    func.val(p.val)
    tape = copy(p.val.levels[1].x)
    return AugmentedReturn{Nothing,Nothing,typeof(tape)}(nothing, nothing, tape)
end

function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(WaterLily.poisson_solve!)},
    dret,
    tape,
    p::Duplicated{<:WaterLily.MultiLevelPoisson},
)
    pp, dpp = p.val, p.dval
    x_saved = tape
    fine, dfine = pp.levels[1], dpp.levels[1]

    fine.z .= dfine.x
    fill!(fine.x, 0)
    func.val(pp)
    λ = fine.x

    # x = A⁻¹ z, λ = A⁻ᵀ x̄. Accumulate dz̄ += λ; dA_{IJ} += -λ_I x_J.
    # A[I, I-δᵢ] = L[I, i] (off-diag), A[I, I] = D[I] (diagonal); other entries 0.
    dfine.z .+= λ
    d = ndims(x_saved)
    for i in 1:d
        WaterLily.@loop dfine.L[I, i] += -(λ[I]*x_saved[I-WaterLily.δ(i,I)] +
                                           λ[I-WaterLily.δ(i,I)]*x_saved[I]) over I in WaterLily.inside(x_saved)
    end
    WaterLily.@loop dfine.D[I] += -λ[I]*x_saved[I] over I in WaterLily.inside(x_saved)

    fine.x .= x_saved
    fill!(dfine.x, 0)
    return (nothing,)
end

end # module
