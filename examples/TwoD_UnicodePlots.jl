using WaterLily
using LinearAlgebra: norm2

function circle(radius=8;Re=250,n=10,m=6)
    center, ν = radius*m/2, radius/Re
    body = AutoBody((x,t)->norm2(x .- center) - radius)
    Simulation((n*radius+2,m*radius+2), [1.,0.], radius; ν, body)
end

using UnicodePlots,SparseArrays
function flood!(canvas,bitarray,color)
    x,y,_ = findnz(sparse(bitarray))
    points!(canvas,x,y,color)
end
function ascii_vort(sim,lines=10,body=false)
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    width,height = size(sim.flow.σ)
    canvas = AsciiCanvas((2*lines*width)÷height,lines,width=width,height=height)
    flood!(canvas,sim.flow.σ.<-1,:red)
    flood!(canvas,sim.flow.σ.>1,:blue)
    !body && return canvas
    @inside sim.flow.σ[I] = sum(WaterLily.ϕ(i,CartesianIndex(I,i),sim.flow.μ₀) for i ∈ 1:2)/2
    flood!(canvas,sim.flow.σ.<0.25,:white)
    return canvas
end

ansi_moveup(n::Int) = string("\e[", n, "A")
ansi_movecol1 = "\e[1G"
ansi_enablecursor = "\e[?25h"
ansi_disablecursor = "\e[?25l"

function sim_ascii!(sim;duration=10,step=0.25,remeasure=false,lines=10,body=false)
    t₀ = round(sim_time(sim))
    print(ansi_disablecursor)
    print(repeat("\n",lines))
    for tᵢ in range(t₀,t₀+duration;step)
        sim_step!(sim,tᵢ;remeasure)
        print(ansi_moveup(lines),ansi_movecol1,ascii_vort(sim,lines+1,body))
    end
    print(ansi_enablecursor)
end
