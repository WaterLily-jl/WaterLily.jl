using NPZ
using WaterLily

"""
    run_simulation_export_data!(sim; kwargs...)

Run simulation and export data for Python visualization.
Saves vorticity fields, particle positions, body mask, and simulation parameters.
"""
function run_simulation_export_data!(
    sim::Simulation;
    t_i=0.01, 
    duration=2., 
    Δt=0.05, 
    N_particles=2^14, 
    life_particles=100,
    save_path="simulation_data.npz",
    verbose=true,
    mem=Array,
    R=nothing,
)
    N = Int(N_particles)
    life = UInt(life_particles)
    t_f = duration
    
    if R === nothing
        R = inside(sim.flow.p)
    end

    # Initialize particles
    p = Particles(N, sim.flow.σ; mem, life)
    
    # Calculate time points and storage
    time_points = collect(t_i:Δt:t_f)
    n_frames = length(time_points)
    
    # Pre-allocate storage arrays
    vorticity_data = Array{Float32}(undef, n_frames, size(sim.flow.σ[R])...)
    particle_x = Array{Float32}(undef, n_frames, N)
    particle_y = Array{Float32}(undef, n_frames, N)
    
    verbose && println("Running simulation with $n_frames frames")
    
    for (frame_idx, t) in enumerate(time_points)
        # Advance simulation
        while sim_time(sim) < t
            WaterLily.mom_step!(sim.flow, sim.pois)
            update!(p, sim)
        end
        
        # Store vorticity field
        @WaterLily.inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
        vorticity_data[frame_idx, :, :] = Array(sim.flow.σ[R])
        
        # Store particle positions
        positions = Array(p.position)
        for i in 1:N
            if i <= length(positions)
                particle_x[frame_idx, i] = positions[i][1]
                particle_y[frame_idx, i] = positions[i][2]
            else
                particle_x[frame_idx, i] = NaN32
                particle_y[frame_idx, i] = NaN32
            end
        end
        
        verbose && println("Frame $frame_idx/$n_frames: t=$(round(t, digits=3))")
    end
    
    # Prepare data for export
    body_mask = Array(sim.body.μ₀)
    
    data_dict = Dict{String, Any}(
        "vorticity" => vorticity_data,
        "particle_x" => particle_x,
        "particle_y" => particle_y,
        "body_mask" => body_mask,
        "time_points" => time_points,
        "sim_L" => sim.L,
        "sim_U" => sim.U,
        "domain_size" => [size(sim.flow.p)...]
    )
    
    npzwrite(save_path, data_dict)
    verbose && println("Data exported to: $save_path")
    
    return save_path
end
