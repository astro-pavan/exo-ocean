# Prototype fix for the Float32 NaNs (docs/LLM/simulation_speedups.md §12.6): include after simulation_sweep.jl to replace advection_schemes.
# WENO-Z weights C(1 + (τ/(β + 1f-8))²) overflow Float32 when one stencil of the momentum divergence flux is flat (β = 0) and τ ≳ 2e11; the divergence
# smoothness δx(Ax u) + δy(Ay v) is in m³/s (~1e5–1e7), so β ~ 1e10–1e14. Scaling it by a constant leaves τ/β, hence the weights, unchanged.
using Oceananigans.Advection: FunctionStencil, OnlySelfUpwinding, divergence_smoothness

const DIVERGENCE_SCALE = 1e-4    # overflow then needs a divergence > 4e9 m³/s (Δw > 0.02 m/s at 4°); ϵ matters only below ~1 m³/s

struct ScaledDivergence{FT} <: Function    # the WENO stencil code dispatches function stencils on ::Function
    scale :: FT
end

@inline (s::ScaledDivergence)(i, j, k, grid, u, v) = s.scale * divergence_smoothness(i, j, k, grid, u, v)

# WENOVectorInvariant(order = 5) with the divergence smoothness rescaled; used only when building a Float32 model
function scaled_weno_vector_invariant(FT)
    stencil = FunctionStencil(ScaledDivergence(FT(DIVERGENCE_SCALE)))
    upwinding = OnlySelfUpwinding(cross_scheme = WENO(FT; order = 5), δU_stencil = stencil, δV_stencil = stencil)
    return WENOVectorInvariant(FT; order = 5, upwinding)
end

const WENO_FIX = Ref(:float32)    # :float32 (Float32 models only), :off (the unfixed scheme), :always (also Float64, to check it changes nothing)

function advection_schemes(advection)
    advection in ADVECTION || error("advection must be one of $(ADVECTION), got $(repr(advection))")
    FT = Oceananigans.defaults.FloatType
    if advection == :weno5
        fix = WENO_FIX[] == :always || (WENO_FIX[] == :float32 && FT == Float32)
        momentum = fix ? scaled_weno_vector_invariant(FT) : WENOVectorInvariant(order = 5)
        return momentum, WENO(order = 5)
    end
    upwind = UpwindBiased(order = 3)
    momentum = VectorInvariant(vorticity_scheme = upwind, vertical_advection_scheme = upwind, divergence_scheme = upwind)
    return momentum, advection == :upwind3 ? upwind : WENO(order = 5)
end
