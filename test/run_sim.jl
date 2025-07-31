using WaterLily
using Pathlines

"""
    run_simple_simulation(sim; kwargs...)

Run simulation and show progress with timestamps. No data saving or plotting.
"""
function run_simple_simulation(
    sim::Simulation;
    t_i=0.01, 
    duration=2., 
    Δt=0.05, 
    N_particles=2^14, 
    life_particles=100,
    scale=1.0,
    minsize=0.1,
    width=1,
    verbose=true,
    mem=Array,
)
    N = Int(N_particles)
    life = UInt(life_particles)
    t_f = duration

    # Initialize particles
    p = Particles(N, sim.flow.σ; mem, life)
    v = ParticleViz(p, Δt; scale=scale, minsize=minsize, width=width)
    
    # Calculate time points
    time_points = collect(t_i:Δt:t_f)
    n_frames = length(time_points)
    
    verbose && println("Running simulation for $n_frames frames (t=$t_i to $t_f, Δt=$Δt)")
    verbose && println("Particles: $N, Life: $life")
    
    # Run simulation
    start_time = time()
    @time for (frame_idx, t) in enumerate(time_points)
        # Advance simulation
        while sim_time(sim) < t
            WaterLily.mom_step!(sim.flow, sim.pois)
            update!(p, sim)
        end
        
        # Update particle visualization to get current state
        notify!(v, p, sim.flow.Δt[end-1])
        
        # Show progress every ith frame
        i_frame = 1
        if verbose && (frame_idx % i_frame == 0 || frame_idx == n_frames)
            elapsed = time() - start_time
            progress = frame_idx / n_frames * 100
            println("Frame $frame_idx/$n_frames ($(round(progress, digits=1))%) - tU/L = $(round(t, digits=4)) - Elapsed: $(round(elapsed, digits=1))s")
        end
    end
    
    total_time = time() - start_time
    verbose && println("✓ Simulation completed in $(round(total_time, digits=1)) seconds")
    
    return true
end

"""
    run_simulation_collect_data(sim; kwargs...)

Run simulation and collect particle data for post-processing visualization.
Returns arrays of particle positions, previous positions, and time steps.
"""
function run_simulation_collect_data(
    sim::Simulation;
    t_i=0.01, 
    duration=2., 
    Δt=0.05, 
    N_particles=2^14, 
    life_particles=100,
    verbose=true,
    mem=Array,
)
    N = Int(N_particles)
    life = UInt(life_particles)
    t_f = duration

    # Initialize particles
    p = Particles(N, sim.flow.σ; mem, life)
    
    # Calculate time points
    time_points = collect(t_i:Δt:t_f)
    n_frames = length(time_points)
    
    # Pre-allocate storage arrays
    positions = Vector{typeof(p.position)}(undef, n_frames)
    positions_prev = Vector{typeof(p.position⁰)}(undef, n_frames)
    delta_times = Vector{Float64}(undef, n_frames)
    
    verbose && println("Running simulation for $n_frames frames (t=$t_i to $t_f, Δt=$Δt)")
    verbose && println("Particles: $N, Life: $life")
    
    # Run simulation and collect data
    start_time = time()
    @time for (frame_idx, t) in enumerate(time_points)
        # Advance simulation
        while sim_time(sim) < t
            WaterLily.mom_step!(sim.flow, sim.pois)
            update!(p, sim)
        end
        
        # Store particle data for this frame
        positions[frame_idx] = copy(Array(p.position))
        positions_prev[frame_idx] = copy(Array(p.position⁰))
        delta_times[frame_idx] = sim.flow.Δt[end-1]
        
        # Show progress every ith frame
        i_frame = 1
        if verbose && (frame_idx % i_frame == 0 || frame_idx == n_frames)
            elapsed = time() - start_time
            progress = frame_idx / n_frames * 100
            println("Frame $frame_idx/$n_frames ($(round(progress, digits=1))%) - tU/L = $(round(t, digits=4)) - Elapsed: $(round(elapsed, digits=1))s")
        end
    end
    
    total_time = time() - start_time
    verbose && println("Simulation completed in $(round(total_time, digits=1)) seconds")
    
    # Return the collected data and body mask
    return (
        positions = positions,
        positions_prev = positions_prev, 
        delta_times = delta_times,
        body_mask = Array(sim.body.μ₀),
        time_points = time_points
    )
end
