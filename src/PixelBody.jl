using FileIO, Images, ImageDistances, ImageTransformations, Plots

struct PixelBody{T,A<:AbstractArray{T,2}} <: AbstractBody
    μ₀::A # needs to be same size as sim scalar (p) and from 0..1
    # size(sim.flow.p)
    # sim.ϵ -> Gauss σ
    # extrema(μ₀) = (0,1)
end

# Outer constructor for PixelBody from image path
function PixelBody(image_path::String; threshold=0.5, ϵ=1.0)
    img = load(image_path)
    gray_img = reverse(Gray.(img), dims=1)' # Transpose to allign matrix indices with physical x-y coordinates
    @show size(gray_img)
    gray_img_padded = pad_to_pow2_with_ghost_cells(gray_img)
    @show size(gray_img_padded)

    # Binary mask: true for solid, false for fluid
    mask = Float64.(gray_img_padded) .< threshold

    # Compute signed distance field
    sdf = distance_transform(feature_transform(mask)) .- distance_transform(feature_transform(.!mask))

    @show size(sdf)
    # Smooth volume fraction field
    μ₀_array = μ₀.(sdf, ϵ)
    @show size(μ₀_array)


    # TODO: TEMP images for debugging
    display(heatmap(img, color=:coolwarm, title="Raw image", aspect_ratio=:equal))
    display(heatmap(gray_img', color=:coolwarm, title="gray scale image", aspect_ratio=:equal))
    display(heatmap(gray_img_padded', color=:coolwarm, title="gray scale image (padded)", aspect_ratio=:equal))
    display(heatmap(mask', color=:coolwarm, title="Threshold mask", aspect_ratio=:equal))
    display(heatmap(sdf', color=:coolwarm, title="Signed Distance Field (sdf)", aspect_ratio=:equal, clims=(-ϵ, ϵ)))
    display(heatmap(μ₀_array', color=:viridis, title="μ₀ Smoothed Mask", aspect_ratio=:equal))

    return PixelBody(μ₀_array)
end


# Required: domain size must be 2^n × 2^m for the MultiLevelPoisson solver. This function pads an 
# image to ensure this size, and adds 2 cells per dimension to account for ghost cells
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

    padded_img = ones(eltype(img), N + pad_top + pad_bot + 2, M + pad_left + pad_right + 2)
    padded_img[pad_top+1 : pad_top+N, pad_left+1 : pad_left+M] .= img
    @show size(padded_img)

    return padded_img
end

# TODO: Move to PixelBody in src
function measure!(a::Flow{2,T},body::PixelBody;t=zero(T),ϵ=1) where {T}
    a.V .= zero(T); a.μ₀ .= one(T); a.μ₁ .= zero(T)
    @assert size(a.p)==size(body.μ₀) # move to the constructor?
    apply!((i,x)->interp(x,body.μ₀),a.μ₀)
    BC!(a.μ₀,zeros(SVector{2,T}),false,a.perdir) # BC on μ₀, don't fill normal component yet
end

measure_sdf!(a::AbstractArray,body::PixelBody,t=0;kwargs...) = @warn "Can't do this yet"