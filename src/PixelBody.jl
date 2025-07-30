using FileIO, Images, ImageDistances, ImageTransformations, Plots
using LinearAlgebra
try
    using CUDA
    CUDA.allowscalar(false)
catch e
    @warn "CUDA not available, running on CPU only." exception=e
end
import Statistics: mean

struct PixelBody{T,A<:AbstractArray{T,2}} <: AbstractBody
    μ₀::A # needs to be same size as sim scalar (p) and from 0..1
    # size(sim.flow.p)
    # sim.ϵ -> Gauss σ
    # extrema(μ₀) = (0,1)
end

"""
    PixelBody(mask::AbstractArray{Bool,2}; ϵ=1.0, mem=Array)

Simplified constructor that takes an existing boolean mask where:
- Input mask: true=fluid, false=solid

and returns
- μ₀_array: 1=fluid, 0=solid

If the logic of the provided mask is inverted, set 'invert_mask_logic' to true.
"""
function PixelBody(
    mask::AbstractArray{Bool,2}; ϵ=1.0, invert_mask_logic=false, mem=Array
)

    if invert_mask_logic
        # Invert mask so that 1=fluid, 0=solid for compatibility with padding logic
        mask = .!mask
    end

    # Pad mask with ghost cells (assumes all edges are fluid, i.e., 1)
    mask_padded = pad_to_pow2_with_ghost_cells(mask)

    # Compute signed distance field (SDF will be positive in fluid, negative in solid)
    sdf = Float32.(distance_transform(feature_transform(.!mask_padded)) .- distance_transform(feature_transform(mask_padded)))
    
    # Smooth volume fraction field using kernel (μ₀_array: 1=fluid, 0=solid)
    μ₀_array = mem(Float32.(μ₀.(sdf, Float32(ϵ))))

    # # TEMP Debug plot
    # display(heatmap(Array(μ₀_array)', color=:viridis, title="μ₀ Smoothed Mask", aspect_ratio=:equal))
    # println("Press Enter to continue...")
    # readline()

    return PixelBody(μ₀_array)
end

# Outer constructor for PixelBody from image path
function PixelBody(
    image_path::String;
    threshold=0.5,
    diff_threshold=nothing,
    ϵ=1.0, max_image_res=nothing,
    body_color="gray",
    manual_mode=false,
    invert_mask_logic=true,
    mem=Array,
)
    img = load(image_path)
    @show size(img)

    # Downsize image if max_image_res is provided
    if !isnothing(max_image_res)
        img = limit_resolution(img, max_image_res)
        println("Image resized to $(size(img))")
    end

    mask = create_fluid_solid_mask_using_image_recognition(img, body_color, threshold, diff_threshold, manual_mode, force_invert_mask)

    if invert_mask_logic
        # Sometimes image recognition returns inverse mask logic (depends on camera and light). Then a reversal is needed
        # such that 1=fluid, 0=solid for consistent logic
        mask = .!mask
    end

    # Pad mask with ghost cells (assumes all edges are fluid, i.e., 1)
    mask_padded = pad_to_pow2_with_ghost_cells(mask)

    # TODO: Attempt to compute signed distance field (see if this will work to create a gradient between the solid and fluid,
    # required to caclulate body forces).
    sdf = Float32.(distance_transform(feature_transform(.!mask_padded)) .- distance_transform(feature_transform(mask_padded)))

    # TODO: Smooth volume fraction field using kernel (Need to find out how to use properly to create the solid-fluid gradient)
    μ₀_array = mem(Float32.(μ₀.(sdf, Float32(ϵ))))

    # TODO: TEMP images for debugging
    # display(heatmap(Array(img), color=:coolwarm, title="Raw image", aspect_ratio=:equal))
    # display(heatmap(Array(mask)', color=:coolwarm, title="Threshold mask", aspect_ratio=:equal))
    # display(heatmap(Array(mask_padded)', color=:coolwarm, title="Threshold mask (padded)", aspect_ratio=:equal))
    # display(heatmap(Array(sdf)', color=:coolwarm, title="Signed Distance Field (sdf)", aspect_ratio=:equal))
    # display(heatmap(Array(sdf)', color=:coolwarm, title="Signed Distance Field (sdf between ϵ=-1 and ϵ=1)", aspect_ratio=:equal, clims=(-ϵ, ϵ)))
    # display(heatmap(Array(μ₀_array)', color=:viridis, title="μ₀ Smoothed Mask", aspect_ratio=:equal))

    return PixelBody(μ₀_array)
end

"""
    function create_fluid_solid_mask_using_image_recognition(img, body_color, threshold, diff_threshold, manual_mode, force_invert_mask)

Function to produce a boolean mask that distinguishes the solid from the fluid using image recognition logic. Both grayscale
and colored images are supported.

- 'threshold' (float):

Controls the minimum intensity required for a pixel to be considered as the "solid" color (e.g., red, green, blue, or gray).

For grayscale images, the 'threshold' value controls how restrictive the solid detection is. Higher threshold values
make detection more restrictive (detects less solid but decreases noise). The assumed contrast logic is that the solid is dominantly dark (black)
and the fluid is dominantly bright (white). Pixels with intensity below (1 - threshold) are considered solid.

For color images, a pixel's channel (e.g., R for red) must be greater than this value to be considered as part of the solid.
For colored images, lighting conditions, the color chosen for the solid, and even the camera used will influence the solid/fluid
image recognition logic. A "smart" logic was initially implemented to try to address these issues (see further documentation
in the comments below).

 - 'diff_threshold' (float):

Controls how much more the target color channel (e.g., R for red) must exceed the other channels for a pixel to be considered solid.
Helps distinguish the solid color from backgrounds or lighting variations.
For example, for red: a pixel is solid if R > threshold and R - G > diff_threshold and R - B > diff_threshold.
Suggested values:

- 'manual_mode' (bool) = False:

User selected threshold and diff_threshold are used if manual_mode is set to true (overwrites color detection logic).
Smart logic is still used to use color hierarchy to invert matrix

===Suggestions for threshold and diff threshold===

For well-lit, high-contrast images:
threshold ≈ 0.4-0.6
diff_threshold ≈ 0.1-0.3
For images with less contrast or more noise, try lowering threshold and/or diff_threshold slightly.

If the mask is too small (misses solid), lower the thresholds.
If the mask is too large (includes background), raise the thresholds.
"""
function create_fluid_solid_mask_using_image_recognition(img, body_color, threshold, diff_threshold, manual_mode=false, force_invert_mask=false)
     # Validate the body_color parameter
    valid_colors = ["gray", "red", "green", "blue"]
    if body_color ∉ valid_colors
        throw(ArgumentError("Unsupported solid color: $body_color. Supported colors are: $(join(valid_colors, ", "))."))
    end

    if body_color == "gray"
        img = Gray.(img)  # Convert to grayscale

        gray_img = Gray.(img)
        # Binary mask: 1 for solid, 0 for fluid
        # NOTE: Changed logic to match color behavior - higher threshold = more restrictive = less solid detected
        # Assumes solid is dark (low values) and fluid is bright (high values)
        mask = Float64.(gray_img) .< (1.0 - threshold)  # Invert threshold for consistent behavior

    else
        println("Colored figure selected. Smart body detection will be used.")

        # Convert to RGB to ensure image is in RBG format
        img_rgb = RGB.(img)
        # Extract channels
        R = channelview(img_rgb)[1, :, :]
        G = channelview(img_rgb)[2, :, :]
        B = channelview(img_rgb)[3, :, :]

        # NOTE: Different cameras and lighting conditions cause the boolean logic when trying to distinguish solid from fluid to
        # be inverted, depending on the final color hiearchy of the image. Since the padding function pad_to_pow2_with_ghost_cells
        # assume 1 is solid and 0 is fluid, the mask passed down to the padding function also needs to ahdere to this logic. This
        # means that depending on the color hierarchy, the mask logic changes based on the selected threshold values, and
        # the mask logic needs to be inverted. A smart body detection implemenation is used in the following lines to attempt
        # said automatic reversal.

        # Smart body detection uses channel hierarchy to determine masking logic between solid and fluid
        R_mean = mean(R)
        G_mean = mean(G)
        B_mean = mean(B)

        println("Channel means: R=$(round(R_mean, digits=3)), G=$(round(G_mean, digits=3)), B=$(round(B_mean, digits=3))")
        println("Manual mode: $manual_mode, Force invert mask: $force_invert_mask")

        # Analyze channel hierarchy and relative differences between color channels
        R_vs_G_diff = R_mean - G_mean
        R_vs_B_diff = R_mean - B_mean
        total_color_range = maximum([maximum(R), maximum(G), maximum(B)]) -
                           minimum([minimum(R), minimum(G), minimum(B)])

        # Logic to determine if mask inversion is needed based on color hierarchy
        needs_inversion = false

        if manual_mode
            # MANUAL MODE: Use provides thresholds but smart inversion logic based on color hierarchy is still used
            println("MANUAL MODE: Using provided threshold values")
            if R_mean > G_mean && R_mean > B_mean
                println("RED DOMINANT camera detected (mask inversion appplied)")
                needs_inversion = true
            elseif G_mean > R_mean && B_mean > R_mean
                println("RED SUPPRESSED camera detected (mask inversion not applied)")
                needs_inversion = false
            else
                println("BALANCED channels detected (mask inversion not applied)")
                needs_inversion = false
            end
        else
            # SMART MODE: Auto-adjust thresholds and determine inversion
            if R_mean > G_mean && R_mean > B_mean
                # Red is dominant channel
                println("RED DOMINANT camera detected")

                # Scale thresholds based on how much red dominates
                red_dominance = min(R_vs_G_diff, R_vs_B_diff) / total_color_range
                println("Red dominance factor: $(round(red_dominance, digits=3))")

                # Selects threshold and diff_threshold based on red dominance
                if red_dominance > 0.05  # Significant red dominance
                    threshold = 0.5 + red_dominance  # Higher threshold for red-heavy cameras
                    diff_threshold = 0.05 + red_dominance * 0.5  # Lower diff since red is already elevated
                else
                    # TODO: Need to calibrate better values
                    threshold = 0.45
                    diff_threshold = 0.15
                end
                needs_inversion = true  # RED DOMINANT requires inversion (e.g. macbook camera)

            elseif G_mean > R_mean && B_mean > R_mean
                # Red is lowest channel (e.g. logitech webcam)
                println("RED SUPPRESSED camera detected")

                # Scale thresholds based on how much red is suppressed
                red_suppression = max(G_mean - R_mean, B_mean - R_mean) / total_color_range
                println("Red suppression factor: $(round(red_suppression, digits=3))")

                threshold = 0.35 + red_suppression * 0.2  # Lower threshold for red-suppressed cameras
                diff_threshold = 0.15 + red_suppression * 0.3  # Higher diff needed to detect red
                needs_inversion = false  # RED SUPPRESSED doesn't need inversion (e.g. logitech webcam)

            else
                # Balanced channels - use adaptive thresholds based on overall range
                println("BALANCED channels detected")

                # Use the total dynamic range to scale thresholds
                if total_color_range > 0.5
                    threshold = 0.4
                    diff_threshold = 0.2
                else
                    # Lower dynamic range needs more sensitive detection
                    threshold = 0.3
                    diff_threshold = 0.1
                end
                needs_inversion = false  # Default behavior
            end

            # Ensure thresholds are within reasonable bounds
            threshold = clamp(threshold, 0.2, 0.7)
            diff_threshold = clamp(diff_threshold, 0.05, 0.4)
        end

        println("FINAL THRESHOLDS:")
        println("   threshold = $(round(threshold, digits=3))")
        println("   diff_threshold = $(round(diff_threshold, digits=3))")
        println("   needs_inversion = $needs_inversion")
        println("   manual_mode = $manual_mode")
        println("="^50)


        # TODO: The above logic was developed around the red color. However, other colors can be selected for the solid. For
        # now, the automated mask inversion logic is only used for the color 'red', but might be useful later.
        if body_color == "red"
            # Detect red pixels (flip logic for different hierarchies of color channels)
            red_detected = (R .> threshold) .& ((R .- G) .> diff_threshold) .& ((R .- B) .> diff_threshold)

            if needs_inversion
                # RED DOMINANT: red_detected=true means solid, so mask should be true for solid
                mask = red_detected  # 1 for solid (red), 0 for fluid
                println("Applied mask = red_detected (no inversion)")
            else
                # RED SUPPRESSED: red_detected=true means solid, but we need 0 for solid to match padding
                mask = .!red_detected  # 0 for solid (red), 1 for fluid
                println("Applied mask = .!red_detected (inverted)")
            end

        elseif body_color == "green"
            green_detected = (G .> threshold) .& ((G .- R) .> diff_threshold) .& ((G .- B) .> diff_threshold)
            # For now, use standard logic for green (can be enhanced later)
            mask = .!green_detected
            
        elseif body_color == "blue"
            blue_detected = (B .> threshold) .& ((B .- G) .> diff_threshold) .& ((B .- R) .> diff_threshold)
            # For now, use standard logic for blue (can be enhanced later)
            mask = .!blue_detected
        end

    end

    mask = reverse(mask, dims=1)' # Transpose to align matrix indices with physical x-y

    return mask
end

"""
    pad_to_pow2_with_ghost_cells(img)

Pads a binary mask to the next power-of-2 size in each dimension and adds 2 ghost cells per dimension.

**Mask logic required:**
    - Input mask must be 1 for solid, 0 for fluid (Bool or numeric).
    - Padding is always done with 0 (fluid) at the boundaries.
    - If mask is True/1 for fluid, False/0 for solid, invert it before calling this function.

IMPORTANT: Assumes all edges are fluid (0 in the mask), so will cause problems if there are solids in the boundary or if
there is an inversion in the pixel body mask due to camera scheme color and thresholds.
"""
function pad_to_pow2_with_ghost_cells(img)
    @show N, M = size(img)

    # Compute nearest powers of 2
    pow2_N = 2 ^ ceil(Int, log2(N))
    pow2_M = 2 ^ ceil(Int, log2(M))

    # Compute padding needed
    pad_N = pow2_N - N
    pad_M = pow2_M - M

    pad_top  = pad_N ÷ 2
    pad_bot  = pad_N - pad_top
    pad_left = pad_M ÷ 2
    pad_right = pad_M - pad_left

    padded_img = ones(eltype(img), N + pad_top + pad_bot + 2, M + pad_left + pad_right + 2) # All edges are fluid (1)
    padded_img[pad_top+1 : pad_top+N, pad_left+1 : pad_left+M] .= img
    @show size(padded_img)

    return padded_img
end

function measure!(a::Flow{2,T},body::PixelBody;t=zero(T),ϵ=1) where {T}
    a.V .= zero(T); a.μ₀ .= one(T); a.μ₁ .= zero(T)
    @assert size(a.p)==size(body.μ₀) # move to the constructor?
    μ₀ = body.μ₀
    apply!((i,x)->interp(x,μ₀), a.μ₀)
    BC!(a.μ₀,zeros(SVector{2,T}),false,a.perdir) # BC on μ₀, don't fill normal component yet
end

measure_sdf!(a::AbstractArray,body::PixelBody,t=0;kwargs...) = @warn "Can't do this yet"


function limit_resolution(img, max_grid::Int)
    h, w = size(img)
    max_dim = max(h, w)

    # Do not resize if image dimensions do not exceed set limit
    if max_dim <= max_grid
        return img
    end

    scale = max_grid / max_dim
    new_h = round(Int, h * scale)
    new_w = round(Int, w * scale)

    println("Resizing from $(h)x$(w) → $(new_h)x$(new_w)")
    return imresize(img, (new_h, new_w))
end


"""
    function estimate_characteristic_length(body::PixelBody; method="pca", plot_method=false)

Estimates the characteristic length of a PixelBody by either a bounding-box method (method='bbox')
or using Principal Component Analysis (method='pca').

Can enable 'plot_method=true' to see a plot of how each method estimated the characteristic length.
"""
function estimate_characteristic_length(body::PixelBody; method="pca", plot_method=false)
    body = body.μ₀

    if method == "pca"
        Lc, Θ_deg = characteristic_length_pca(body, plot_method=plot_method)
        return Lc, Θ_deg
    elseif method == "bbox"
        Lc = characteristic_length_bbox(body, plot_method=plot_method)
    else
        throw(ErrorException(
        "Invalid characteristic length estimation method selected. Current supported options are:
        'pca', 'bbox'."))
    end



end

"""
    characteristic_length_bbox(B::Matrix{Float64}; plot_method=false)

Method to estimate the characteristic length of a PixelBody by identifying a bounding box
around the object, and assuming the characteristic length is the longest diagonal in the box.
"""
function characteristic_length_bbox(B::AbstractArray{<:AbstractFloat,2}; plot_method=false)
    # Convert to CPU array if using CUDA array
    if isa(B, CuArray)
        B = Array(B)
    end
    # 0 Is solid, but due to image recognition artifacts there might be floating point errors
    coords = findall(abs.(B) .< 1e-6)
    if isempty(coords)
        throw(ErrorException(
            "No solid detected when attempting to calcualte characteristic length"))
    end

    xs = [c[1] for c in coords]
    ys = [c[2] for c in coords]

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)

    dx = xmax - xmin
    dy = ymax - ymin
    characteristic_length = sqrt(dx^2 + dy^2)

    if plot_method
        display(heatmap(Array(B)', c=:blues, aspect_ratio=1, xlabel="x", ylabel="y", legend=false))

        # Bounding box corners
        xcorners = [xmin, xmax, xmax, xmin, xmin]
        ycorners = [ymin, ymin, ymax, ymax, ymin]
        display(plot!(xcorners, ycorners, linecolor=:red, linewidth=2, label="Bounding Box"))

        # Diagonal line
        display(plot!([xmin, xmax], [ymin, ymax], linecolor=:orange, linewidth=2, linestyle=:dash, label="Diagonal"))

        display(scatter!(xs, ys, color=:black, markersize=2, alpha=0.4, label="Solid Pixels"))
    end

    return characteristic_length
end


"""
    characteristic_length_pca(B::Matrix{Float64}; plot_method=false)

Method to estimate the characteristic length of a PixelBody using the PCA (Principal Component Analysis).
In short, it finds the direction perpendicular to the principal component of the pixel distribution (the
direction in which points are most spread.)

Also estimates the angle of attack of the object based on the principal axis direction
"""
function characteristic_length_pca(B::AbstractArray{<:AbstractFloat,2}; plot_method=false)
    # Convert to CPU array if using CUDA array
    if isa(B, CuArray)
        B = Array(B)
    end
    # 0 Is solid, but due to image recognition artifacts there might be floating point errors
    coords = findall(abs.(B) .< 1e-6)
    if isempty(coords)
        throw(ErrorException(
            "No solid detected when attempting to calcualte characteristic length"))
    end

    xs = Float64[c[1] for c in coords]
    ys = Float64[c[2] for c in coords]

    pts = hcat(xs, ys)'  # 2 × N matrix
    μ = mean(pts, dims=2)
    X = pts .- μ
    U, _, _ = svd(X)

    # Project onto first principal axis
    p1 = U[:, 1]
    projections = p1' * X
    half_span = maximum(abs.(projections))

    characteristic_length = half_span * 2

    # Estimate angle of attack in degrees from x-axis (might not work properly for some objects)
    θ_deg = 180 - rad2deg(atan(p1[2], p1[1]))  

    if plot_method
        # Compute line endpoints
        min_proj = minimum(projections)
        max_proj = maximum(projections)
        p_start = μ + min_proj * p1
        p_end = μ + max_proj * p1

        display(scatter(xs, ys, markersize=2, label="Solid Pixels", aspect_ratio=1, c=:black))
        display(scatter!([μ[1]], [μ[2]], markershape=:cross, color=:red, label="Centroid"))

        display(plot!([p_start[1], p_end[1]], [p_start[2], p_end[2]],
            linewidth=2, color=:orange, label="Principal Axis"))

        display(scatter!([p_start[1], p_end[1]], [p_start[2], p_end[2]],
                markersize=6, markercolor=:orange, label="Extent"))
    end

    return characteristic_length, θ_deg
end