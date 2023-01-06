using WaterLily
using StaticArrays, LinearAlgebra: norm2
include("TwoD_plots.jl")

function perforated(radius=8;perforations=3,solidfraction=3/4,Re=250,n=12,m=6,ϵ=0.5,thk=2ϵ+√2)
    # Define the symmetry sector angle and slope to first preforation
    sector = π/perforations
    angle = solidfraction*sector-thk/2radius
    slope = SVector(cos(angle),sin(angle))

    # Check for domain errors
    angle<0 && throw(DomainError(angle,"shell width not resolved"))
    sector-angle<thk/radius && throw(DomainError(sector-angle,"gap not resolved"))

    # Define body with the signed distance function
    body = AutoBody() do point,time
        # location relative to circle center
        x = point .- 0.5radius*m 
        # location within first symmetry sector
        α = round(atan(x[2],x[1])/2sector)*2sector
        y = abs.(SMatrix{2,2}([cos(α) sin(α); -sin(α) cos(α)]) * x)
        # return distance
        (y[1]*slope[2]<y[2]*slope[1] ? # above perf slope?
            norm2(y-radius*slope) :    # distance to perf edge
            abs(norm2(y)-radius)       # distance to circle
            )-thk/2                    # subtract shell thickness
    end

    # Create simulation: (domain size), [background flow], radius; viscosity, body, & kernel
    Simulation((n*radius+2,m*radius+2), [1.,0.], radius; ν=radius/Re, body, ϵ)
end

sim_gif!(shell(64);duration=20,step=0.4,legend=false,border=:none)
