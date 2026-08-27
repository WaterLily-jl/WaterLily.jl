module WaterLilyPathlinesExt

using WaterLily, Pathlines, Makie

"""
    _pathlines_setup(sim; N, life, mem, bgcolor, fadetau, colormap, colorrange,
                          figsize, resolution, kwargs...)

Called by `WaterLilyMakieExt.viz!` when Pathlines is loaded.  Creates a
`PathlineCanvas` and a `Particles` swarm, sets up a Makie figure with a single
`image!` layer backed by the canvas Observable, and returns
`(fig, ax, update_fn)` where `update_fn(sim)` fades the canvas, advances the
particles one integration step with the current flow field, draws the new
segments, and notifies the Observable.

Body visualization and the animation loop are handled entirely by
`WaterLilyMakieExt.viz!` — this function only sets up the canvas renderer.
"""
function _pathlines_setup(sim;
        N=10_000, life=UInt(255), mem=Array,
        bgcolor=:black, fadetau=0.2,
        colormap=:plasma, colorrange=(0, 3),
        figsize=nothing, resolution=nothing,
        kwargs...)   # absorb viz! kwargs irrelevant to pathlines

    σ = sim.flow.σ
    nx, ny = size(inside(σ))

    pc = Pathlines.PathlineCanvas(nx, ny; bgcolor, fadetau,
                                  colormap, colorrange, figsize, resolution)
    p  = Pathlines.Particles(N, σ; mem, life)
    pos  = Array(p.position)
    pos⁰ = Array(p.position⁰)

    fig = Makie.Figure(size=pc.figsize)
    ax  = Makie.Axis(fig[1, 1]; autolimitaspect=1, limits=(1, nx, 1, ny))
    Makie.hidedecorations!(ax)
    ocanvas = Makie.Observable(pc.canvas)
    Makie.image!(ax, (1, nx), (1, ny), ocanvas)

    function update_fn(sim)
        # Δt[end-1] is used by Particles.update! — skip if sim hasn't stepped yet
        length(sim.flow.Δt) < 2 && return
        Pathlines.update!(p, sim)
        copyto!(pos, p.position)
        copyto!(pos⁰, p.position⁰)
        dt = Float32(sim.flow.Δt[end-1])
        Pathlines.fade!(pc, dt*sim.U/sim.L)
        Pathlines.draw!(pc, pos, pos⁰, dt)
        Makie.notify(ocanvas)
    end

    return fig, ax, update_fn
end

function __init__()
    WaterLily._pathlines_viz_hook[] = _pathlines_setup
end

end # module
