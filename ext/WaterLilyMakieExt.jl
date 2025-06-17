module WaterLilyMakieExt

using Makie, WaterLily
using Makie.GeometryBasics
import WaterLily: viz!, get_body, plot_body_obs!

"""
    update_body!(a_cpu::Array, sim)

Measure the body SDF and update the CPU buffer array.
"""
function update_body!(a_cpu::Array, sim)
    WaterLily.measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
    copyto!(a_cpu, sim.flow.σ[inside(sim.flow.σ)])
end

"""
    default_colormap_and_levels(minv, maxv, threshhold, levels)
"""
function default_colormap_and_levels(clims; threshhold=0.1, nlevels=20, colormap=:seismic, threshhold_color=RGB(1,1,1))
    colormap_upperrange(max_val, threshhold, nlevels) = range(threshhold, max_val, (nlevels - 1) ÷ 2)
    colormap_lowerrange(min_val, threshhold, nlevels) = range(min_val, threshhold, (nlevels - 1) ÷ 2)
    lowerrange = colormap_lowerrange(clims[0], threshhold, nlevels)
    upperrange = colormap_upperrange(clims[1], -threshhold, nlevels)
    levels = [lowerrange; upperrange]
    colors = palette(colormap, nlevels).colors.colors
    colors[[(nlevels - 1) ÷ 2 + 1, (nlevels - 1) ÷ 2 + 2]] .= threshhold_color
    colors, levels
end

"""
    get_body(sdf_array, ::Val{false})

Identity function that returns the same `sdf_array`. Required for compatibility with WaterLilyMeshingExt.
"""
get_body(sdf_array, ::Val{false}) = sdf_array

"""
    plot_body_obs!(ax, b::Observable{Array{T,2}} where T; color=:black)

Plot the 2D body SDF `b::Observable` at value 0 in a 2D contourf axis.
"""
plot_body_obs!(ax, b::Observable{Array{T,2}} where T; color=(:grey, 0.9)) = Makie.contourf!(ax, b;
    levels=[0], colormap=[color], extendlow=:auto
)

"""
    plot_body_obs!(ax, sdf_array::Observable{Array{T,3}} where T; color=:black, isorange=0.3)

Plot the 3D body SDF `sdf_array::Observable` at value 0 in a 3D volume axis.
"""
plot_body_obs!(ax, sdf_array::Observable{Array{T,3}} where T; color=(:grey, 0.9), isorange=0.3) = Makie.volume!(ax, sdf_array;
    algorithm=:iso, colormap=[color], isovalue=0, isorange, lowclip=color
)

"""
    plot_σ_obs!(ax, σ::Observable{Array{T,2}} where T; kwargs...)

Plot the 2D scalar `σ::Observable` in a 2D contour axis.
"""
plot_σ_obs!(ax, σ::Observable{Array{T,2}} where T; kwargs...) = Makie.contourf!(ax, σ; kwargs...)

"""
    plot_σ_obs!(ax, σ::Observable{Array{T,3}} where T; kwargs...)

Plot the 3D scalar `σ::Observable` in a 3D volume axis.
"""
plot_σ_obs!(ax, σ::Observable{Array{T,3}} where T; kwargs...) = Makie.volume!(ax, σ; kwargs...)

"""
    viz!(sim, f!::Function; t_end=nothing, remeasure=true, max_steps=typemax(Int), verbose=true,
        d=ndims(sim.flow.p), CIs=nothing, cut=nothing,
        body=!(typeof(sim.body)<:WaterLily.NoBody), body_color=:black, body2mesh=false,
        video=nothing, skipframes=1, hideaxis=false, elevation=π/8, azimuth=1.275π, framerate=30, compression=5,
        theme=nothing, fig_size=(1200,1200), fig_pad=40, kwargs...
    )

General visualization routine to simulate and render the flow field using Makie.
Works for both 2D and 3D simulations. For 3D simulations, the user can choose to render 3D volumetric scalar data, or a 2D slice.
Users must pass a function `f!` used to post-process the flow field data and copy the scalar field into a CPU buffer array.
The interface of `f!` must follow `f!(cpu_array::Array, sim::AbstractSimulation)`. For example, to visualize vorticity magnitude:
```
function f!(cpu_array, sim)
    a = sim.flow.σ
    WaterLily.@inside a[I] = WaterLily.ω_mag(I,sim.flow.u)
    copyto!(cpu_array, a[inside(a)]) # copy to CPU
end
```
Keyword arguments:
    - `t_end::Number`: Simulation end time.
    - `remeasure::Bool`: Update the body position.
    - `max_steps::Int`: Simulation end time.
    - `verbose::Bool`: Print simulation information.
    - `λ::Function`: Convective scheme function passed into `sim_step!`.
    - `meanflow::MeanFlow`: `MeanFlow` object passed into `sim_step!`.
    - `udf::Function`: User-defined function passed into `sim_step!`.
    - `udf_kwargs::Dict{Symbol}`: User-defined function keyword arguments passed into `sim_step!`. Needs to be a `Dict{Symbol}` or any
        `Pair{Symbol,Any}` iterator.
    - `d::Int`: Plot dimension. `d=2` produces a `Makie.contourf`, and `d=3` produces a `Makie.volume`.
        Defaults to simulation number of dimension.
    - `CIs::CartesianIndices`: Range of Cartesian indices to render.
    - `cut::Tuple{Int, Int, Int}`: For 3D simulation and `d=2`, `cut` provides the plane to render, and defaults to (0,0,N[3]/2).
        It needs to be defined as a Tuple of 0s with a single non-zero entry on the cutting plane.
    - `tidy_colormap::Bool`: Adjusts the colormap to have a fully transparent color near 0 values. Additional plotting options
        passed into `kwargs` (eg. colormap, levels) are preserved.
        Pass `threshhold::Number` to adjust the near-0 range`, `threshhold_color::RGBA` to set to a color different from white, and
        clims::Tuple{Number,Number} to adjust the colormap limits.
    - `body::Bool`: Plot the body.
    - `body2mesh::Bool`: The body is plotted by generating a GeometryBasics.mesh, otherwise just as a Makie.volume (faster).
        Note that Meshing and GeometryBasics packages must be loaded if `body2mesh=true`.
    - `body_color`: Body color, can also containt alpha value, eg (:black, 0.9)
    - `video::String`: Save the simulation as as video, instead of rendering. Defaults to `nothing` (not saving video).
    - `skipframes::Int`: Only render every `skipframes` time steps.
    - `hideaxis::Bool`: Figures without axis details.
    - `azimuth::Number`: Camera azimuth angle. Find a suitable angle interactively checking `ax.azimuth.val`
    - `elevation::Number`: Camera elevation angle. Find a suitable angle interactively checking `ax.elevation.val`.
    - `framerate::Int`: Video framerate.
    - `compression::Int`: Video compression.
    - `theme::Attributes`: Makie theme, eg. `theme_light()` or `theme_latexfonts()`
    - `fig_size::Tuple{Int, Int}`: Figure size.
    - `fig_pad::Int`: Figure padding.
    - `kwargs`: Additional keyword arguments passed to `plot_σ_obs!`.
"""
function viz!(sim, f!::Function; t_end=nothing, remeasure=true, max_steps=typemax(Int), verbose=true,
    λ=quick, udf=nothing, udf_kwargs=nothing, meanflow=nothing,
    d=ndims(sim.flow.p), CIs=nothing, cut=nothing, tidy_colormap=true,
    body=!(typeof(sim.body)<:WaterLily.NoBody), body_color=:grey, body2mesh=false,
    video=nothing, skipframes=1, hideaxis=false, elevation=π/8, azimuth=1.275π, framerate=30, compression=5,
    theme=nothing, fig_size=(1200,1200), fig_pad=40, kwargs...)

    function update_data()
        f!(dat, sim)
        σ[] = WaterLily.squeeze(dat[CIs])
        if body && remeasure
            update_body!(dat, sim)
            σb_obs[] = get_body(WaterLily.squeeze(dat[CIs]), Val{body2mesh}())
        end
    end

    d==2 && (@assert !(body2mesh) "body2mesh only allowed for 3D plots (d=3).")
    body2mesh && (@assert !isnothing(Base.get_extension(WaterLily, :WaterLilyMeshingExt)) "If body2mesh=true, Meshing and GeometryBasics must be loaded.")
    D = ndims(sim.flow.σ)
    @assert d <= D "Cannot do a 3D plot on a 2D simulation."
    !isnothing(udf) && !isnothing(udf_kwargs) && (@assert all(isa(kw, Pair{Symbol}) for kw in udf_kwargs) "udf_kwargs needs to contain Pair{Symbol,Any} elements, eg. Dict{Symbol,Any}.")
    isnothing(udf) && (udf_kwargs=[])

    isnothing(CIs) && (CIs = CartesianIndices(Tuple(1:n for n in size(inside(sim.flow.σ)))))
    dat = sim.flow.σ[inside(sim.flow.σ)] |> Array
    if d != D && all(>(1), length.(CIs.indices)) # Requesting 2D plot on 3D data, and CIs is not a slice
        isnothing(cut) && (cut = (0, 0, size(dat,3)÷2))
        @assert count(==(0), cut) == 2 "Requesting 2D plot on 3D data, but `cut` is not an slice, eg: (0,0,10)"
        cut_dim = findfirst(!=(0), cut)
        CIs = Tuple(i == cut_dim ? (cut[i]:cut[i]) : CIs.indices[i] for i in 1:D) |> CartesianIndices
    end
    limits = Tuple((1,n) for n in size(CIs) if n > 1)

    f!(dat, sim)
    σ = WaterLily.squeeze(dat[CIs]) |> Observable
    if body
        update_body!(dat, sim)
        σb_obs = get_body(WaterLily.squeeze(dat[CIs]), Val{body2mesh}()) |> Observable
    end

    !isnothing(theme) && set_theme!(theme)
    fig = Figure(size=fig_size, figure_padding=fig_pad)
    ax = d==2 ? Axis(fig[1, 1]; aspect=DataAspect(), limits) : Axis3(fig[1, 1]; limits, azimuth, elevation)
    if d == 2 && tidy_colormap
        clims = :clims in keys(clims) ? kwargs[:clims] : (-1,1)
        nlevels = :levels in keys(kwargs) && kwargs[:levels] isa Int ? kwargs[:levels] : 20
        colormap = :colormap in keys(kwargs) ? kwargs[:colormap] : :seismic
        threshhold = :threshhold in keys(kwargs) ? kwargs[:threshhold] : 0.1
        threshhold_color = :threshhold_color in keys(kwargs) ? kwargs[:threshhold_color] : RGB(1,1,1)
        tidy_colormap, tidy_levels = default_colormap_and_levels(clims; threshhold, nlevels, colormap, threshhold_color)
        kwargs = remove_kwargs(:levels, :colormap, :clims, :threshhold, :threshhold_color, :extendlow, :extendhigh; kwargs...)
        kwargs = add_kwarg(:colormap=>tidy_colormap, :levels=>tidy_levels, :extendlow=>:auto, :extendhigh=>:auto; kwargs...)
    end
    plot_σ_obs!(ax, σ; kwargs...)
    body && plot_body_obs!(ax, σb_obs; color=body_color)
    hideaxis && (hidedecorations!(ax); ax.xspinesvisible = false; ax.yspinesvisible = false; ax.zspinesvisible = false)

    if !isnothing(t_end) # time loop for animation
        steps₀ = length(sim.flow.Δt)
        if !isnothing(video)
            Makie.record(fig, video; framerate, compression) do frame
                while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
                    sim_step!(sim; remeasure, λ, udf, meanflow, udf_kwargs...)
                    verbose && sim_info(sim)
                    if mod(length(sim.flow.Δt), skipframes) == 0
                        update_data()
                        recordframe!(frame)
                    end
                end
            end
        else
            display(fig)
            while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
                sim_step!(sim; remeasure, λ, udf, meanflow, udf_kwargs...)
                verbose && sim_info(sim)
                if mod(length(sim.flow.Δt), skipframes) == 0
                    update_data()
                end
            end
        end
    end
    isnothing(video) && display(fig)
    return sim, fig, ax
end

function ω_viz!(cpu_array, sim)
    a = sim.flow.σ
    WaterLily.@inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    copyto!(cpu_array, a[inside(a)])
end

# Utils
add_kwarg(args...; kwargs...) = (; kwargs..., (p.first => p.second for p in args)...) |> pairs
remove_kwargs(args...; kwargs...) = (;(x.first=>x.second for x in kwargs if !in(x.first, args))...) |> pairs

end # module