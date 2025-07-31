using Plots
gr()

"""
    flood(f)

Plot a filled contour plot of the 2D array `f`. The keyword arguments are passed to `Plots.contourf`.
"""
function flood(f::AbstractArray; shift=(0.,0.), cfill=:RdBu_11, clims=(), levels=10, kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    Plots.contourf(axes(f,1).+shift[1], axes(f,2).+shift[2], f'|>Array,
                   linewidth=0, levels=levels, color=cfill, clims=clims,
                   aspect_ratio=:equal, 
                   axis=false, 
                   showaxis=false,
                   grid=false,
                   framestyle=:none,
                   margin=0Plots.mm,
                   left_margin=0Plots.mm,
                   right_margin=0Plots.mm,
                   top_margin=0Plots.mm,
                   bottom_margin=0Plots.mm,
                   size=(800,800),
                   dpi=150; kv...)
end

"""
    create_heatmap_gif_from_data!(data; save_path, time_points, verbose=true, kv...)

Create a heatmap GIF from 3D data array where data[x, y, time].
Uses the same visualization approach as WaterLily's sim_gif! function.
"""
function create_heatmap_gif_from_data!(
    data::AbstractArray{T,3};
    save_path="picture_sim_app/output/heatmap.gif",
    time_points=nothing,
    verbose=true,
    cfill=:RdBu_11,
    clims=(-5,5),
    levels=10,
    kv...
) where T
    
    n_frames = size(data, 3)
    verbose && println("Creating heatmap GIF from $n_frames frames of data")
    
    # Use provided time points or create frame indices
    if time_points === nothing
        time_values = 1:n_frames
        time_label = "Frame"
    else
        time_values = time_points
        time_label = "tU/L"
    end
    
    @time begin
        anim = @animate for frame_idx in 1:n_frames
            # Extract 2D slice for current frame
            field_slice = data[:, :, frame_idx]
            
            # Create flood plot
            flood(field_slice; cfill=cfill, clims=clims, levels=levels, kv...)
            
            # Add progress output
            if verbose
                time_val = time_values[frame_idx]
                println("$time_label=", round(time_val, digits=4))
            end
        end
        gif(anim, save_path)
    end
    
    verbose && println("Heatmap GIF saved to: $save_path")
end
