using WaterLily
using LinearAlgebra: norm2
using Makie

function TGV_video(p=6,Re=1e5)
    # Define vortex size, velocity, viscosity
    L = 2^p; U = 1; ν = U*L/Re

    # Apply Taylor-Green-Vortex velocity field
    u = apply(L+2,L+2,L+2,3) do i,vx
        x,y,z = @. (vx-1.5)*π/L
        i==1 && return -U*sin(x)*cos(y)*cos(z)
        i==2 && return  U*cos(x)*sin(y)*cos(z)
        return 0.
    end

    # Initialize Flow and Poisson system
    c = ones(L+2,L+2,L+2,3)
    a = Flow(u,c,zeros(3),ν=ν)
    b = MultiLevelPoisson(c)

    # plot the vorticity modulus
    @inside a.σ[I] = norm2((WaterLily.curl(i,I,a.u) for i∈1:3))*L/U
    scene = Scene(backgroundcolor = :black)
    scene = volume!(scene,a.σ,colorrange=(π,4π),algorithm = :absorption)
    vol_plot = scene[end]

    # Plot flow evolution
    tprint,Δprint,nprint = 0.0,0.1,100
    record(scene,"file.mp4",1:nprint,framerate=10,compression=5) do i
        tprint-=Δprint
        while tprint<0
            println(round(Int,i/nprint*100),"%, tU/L=",
                    round(sum(a.Δt)*U/L,digits=4),", Δt=",
                    round(a.Δt[end],digits=3))
            mom_step!(a,b) # evolve Flow
            tprint += a.Δt[end]*U/L
        end
        # update volume plot data
        @inside a.σ[I] = norm2((WaterLily.curl(i,I,a.u) for i∈1:3))*L/U
        BC!(a.σ)
        vol_plot[1] = a.σ
    end
    return a
end
