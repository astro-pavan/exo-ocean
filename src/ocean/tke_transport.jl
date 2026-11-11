# Zero GM-Redi flux for a tracer whose κ_skew entry is `nothing`; ocean_closure gives CATKE's TKE `e` one when tke_gm_redi = false.
# These methods only extend Oceananigans 0.106.5's diffusive_flux_{x,y,z}; check their signatures in isopycnal_skew_symmetric_diffusivity.jl on an upgrade.
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, DiffusiveFormulation
import Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z

for id in 1:4, flux in (:diffusive_flux_x, :diffusive_flux_y, :diffusive_flux_z)
    K = Tuple{ntuple(_ -> Any, id - 1)..., Nothing, Vararg{Any}}
    @eval @inline $flux(i, j, k, grid, ::IsopycnalSkewSymmetricDiffusivity{<:Any, <:DiffusiveFormulation, <:NamedTuple{<:Any, <:$K}},
                        closure_fields, ::Val{$id}, c, clock, fields, buoyancy) = zero(grid)
end
