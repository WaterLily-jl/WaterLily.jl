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
    gray_img = Gray.(img)' # Transpose image to match x-y orientation for simulation
    gray_img_padded = pad_to_pow2_with_ghosts(gray_img)

    # Binary mask: true for solid, false for fluid
    mask = Float64.(gray_img_padded) .< threshold

    # Compute signed distance field
    sdf = distance_transform(feature_transform(mask)) .- distance_transform(feature_transform(.!mask))

    # Smooth volume fraction field
    μ₀_array = WaterLily.μ₀.(sdf, ϵ)

    display(heatmap(sdf, color=:coolwarm, title="Signed Distance Field (sdf)", clims=(-ϵ, ϵ)))
    display(heatmap(μ₀_array, color=:viridis, title="μ₀ Smoothed Mask"))

    return PixelBody(μ₀_array)
end


# T Required: domain size must be 2^n × 2^n for MultiLevelPoisson in WaterLily
function pad_to_pow2_with_ghosts(img)
    n, m = size(img)
    n2 = 2^ceil(Int, log2(n))
    m2 = 2^ceil(Int, log2(m))
    padded = fill(1.0, n2 + 2, m2 + 2)  # +2 for ghost cells (1-padded border)
    padded[2:n+1, 2:m+1] .= img  # offset by 1 to center image in interior
    return padded
end