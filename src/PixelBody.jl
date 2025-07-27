using FileIO, Images, ImageDistances, ImageTransformations, Plots
using LinearAlgebra
using CUDA
import Statistics: mean
struct PixelBody{T,A<:AbstractArray{T,2}} <: AbstractBody
    μ₀::A # needs to be same size as sim scalar (p) and from 0..1
    # size(sim.flow.p)
    # sim.ϵ -> Gauss σ
    # extrema(μ₀) = (0,1)
end

# Outer constructor for PixelBody from image path
function PixelBody(image_path::String; threshold=0.5, diff_threshold=nothing, ϵ=1.0, max_image_res=nothing, body_color="gray", mem=Array)
    img = load(image_path)
    @show size(img)

    # Downsize image if max_image_res is provided
    if !isnothing(max_image_res)
        img = limit_resolution(img, max_image_res)
        println("Image resized to $(size(img))")
    end

    # Validate the body_color parameter
    valid_colors = ["gray", "red", "green", "blue"]
    if body_color ∉ valid_colors
        throw(ArgumentError("Unsupported solid color: $body_color. Supported colors are: $(join(valid_colors, ", "))."))
    end

    if body_color == "gray"
        img = Gray.(img)  # Convert to grayscale

        gray_img = Gray.(img)
        # Binary mask: 1 for solid, 0 for fluid
        mask = Float64.(gray_img) .< threshold

    else
        # Convert to RGB to ensure image is in RBG format
        img_rgb = RGB.(img)
        # Extract channels
        R = channelview(img_rgb)[1, :, :]
        G = channelview(img_rgb)[2, :, :]
        B = channelview(img_rgb)[3, :, :]

        if body_color == "red"
            # threshold = 0.5      # Minimum red value
            # diff_threshold = 0.2     # How much more red than green/blue 
            # Binary mask: 0 for solid, 1 for fluid 
            mask = .!((R .> threshold) .& ((R .- G) .> diff_threshold) .& ((R .- B) .> diff_threshold))

        elseif body_color == "green"
            mask = .!((G .> threshold) .& ((G .- R) .> diff_threshold) .& ((G .- B) .> diff_threshold))
            
        elseif body_color == "blue"
            mask = .!((B .> threshold) .& ((B .- G) .> diff_threshold) .& ((B .- R) .> diff_threshold))
        end

    end

    mask = reverse(mask, dims=1)' # Transpose to align matrix indices with physical x-y
    mask_padded = pad_to_pow2_with_ghost_cells(mask)
    @show size(mask_padded)

    # Compute signed distance field
    sdf = Float32.(distance_transform(feature_transform(mask_padded)) .- distance_transform(feature_transform(.!mask_padded)))

    @show size(sdf)
    # Smooth volume fraction field
    μ₀_array = mem(Float32.(μ₀.(sdf, Float32(ϵ))))
    @show size(μ₀_array)

    # TODO: TEMP images for debugging
    display(heatmap(Array(img), color=:coolwarm, title="Raw image", aspect_ratio=:equal))
    display(heatmap(Array(mask)', color=:coolwarm, title="Threshold mask", aspect_ratio=:equal))
    display(heatmap(Array(mask_padded)', color=:coolwarm, title="Threshold mask (padded)", aspect_ratio=:equal))
    display(heatmap(Array(sdf)', color=:coolwarm, title="Signed Distance Field (sdf)", aspect_ratio=:equal))
    display(heatmap(Array(sdf)', color=:coolwarm, title="Signed Distance Field (sdf between ϵ=-1 and ϵ=1)", aspect_ratio=:equal, clims=(-ϵ, ϵ)))
    display(heatmap(Array(μ₀_array)', color=:viridis, title="μ₀ Smoothed Mask", aspect_ratio=:equal))

    return PixelBody(μ₀_array)
end


"""
    pad_to_pow2_with_ghost_cells(img)

Domain size must be (2^n, 2^m) for the MultiLevelPoisson solver. This function pads an
image to ensure this size, and adds 2 cells per dimension to account for ghost cells.

IMPORTANT: Assumes all edges are fluid (0 in the mask), so will cause problems if there are solids in the boundary or if
there is an inversion in the pixel body mask due to camera scheme color and thresholds.
"""
function pad_to_pow2_with_ghost_cells(img)
    @show N, M = size(img)

    # Compute nearest powers of 2
    pow2_N = 2 ^ ceil(Int, log2(N))
    pow2_M = 2 ^ ceil(Int, log2(M))

    # Compute padding needed
    @show pad_N = pow2_N - N
    @show  pad_M = pow2_M - M

    pad_top  = pad_N ÷ 2
    pad_bot  = pad_N - pad_top
    pad_left = pad_M ÷ 2
    pad_right = pad_M - pad_left

    padded_img = zeros(eltype(img), N + pad_top + pad_bot + 2, M + pad_left + pad_right + 2) # TODO: Only works if all edges are
                                                                                             # fluid
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