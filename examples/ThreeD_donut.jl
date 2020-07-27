using WaterLily
using LinearAlgebra: norm2
using Makie

function donut_video(p=6,Re=1e3)
    # Define vortex size, velocity, viscosity
    n,U = 2^p, [1, 0, 0]
    center,R,r = ones(3) .* n/2 .+ 1, n/4, n/16
    ν = norm2(U)*R/Re

    # Immerse a torus (change for other shapes)
    c = BDIM_coef(2n+2,n+2,n+2,3) do xyz  # signed distance function
        x,y,z = xyz - center
        norm2([x,norm2([y,z])-R])-r
    end
    geom = sum(c,dims=4)[:,:,:,1]/3; BC!(geom)

    # Initialize Flow and Poisson system
    u = zeros(2n+2,n+2,n+2,3)
    a = Flow(u,c,U,ν=ν)
    b = MultiLevelPoisson(c)
    mom_step!(a,b)

    # plot the geometry and vorticity modulus
    scene = volume(geom,algorithm=:iso,isorange=0.2)
    scene = volume!(scene,a.σ,colorrange=(0.5,1),algorithm=:absorption)
    vol_plot = scene[end]

    # Plot flow evolution
    tprint,Δprint,nprint = 0.0,0.25,72
    record(scene,"file.mp4",1:nprint,framerate=24,compression=5) do i
        tprint-=Δprint
        while tprint<0
            println(round(Int,i/nprint*100),"%, tU/R=",
                    round(sum(a.Δt)*norm2(U)/R,digits=4),", Δt=",
                    round(a.Δt[end],digits=3))
            mom_step!(a,b) # evolve Flow
            tprint += a.Δt[end]*norm2(U)/R
        end
        # update volume plot data
        @inside a.σ[I] = norm2((WaterLily.curl(i,I,a.u) for i∈1:3))*R/norm2(U)
        BC!(a.σ)
        vol_plot[1] = a.σ
    end
    return a
end
