using WaterLily
using LinearAlgebra: norm2
include("ThreeD_plots.jl")

function TGV(p=6,Re=1e5)
    # Define vortex size, velocity, viscosity
    L = 2^p; U = 1; ν = U*L/Re

    # Taylor-Green-Vortex initial velocity field
    function uλ(i,vx)
        x,y,z = @. (vx-1.5)*π/L                # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
        i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
        return 0.                              # u_z
    end

    # Initialize simulation
    return Simulation((L+2,L+2,L+2),zeros(3),L;U,uλ,ν)
end

function ω_mag_data(sim)
    # plot the vorticity modulus
    @inside sim.flow.σ[I] = WaterLily.ω_mag(I,sim.flow.u)*sim.L/sim.U
    return @view sim.flow.σ[2:end-1,2:end-1,2:end-1]
end

sim,fig = volume_plot(TGV(),ω_mag_data,fname='TGV.mp4',duration=10)
