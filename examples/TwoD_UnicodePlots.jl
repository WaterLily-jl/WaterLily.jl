using UnicodePlots,SparseArrays
function flood!(canvas,bitarray,color)
    x,y,_ = findnz(sparse(bitarray))
    UnicodePlots.points!(canvas,x,y;color)
end
function ascii_vort(sim,lines=10,body=false)
    a = sim.flow.σ; b = @view(a[inside(a)])
    width,height = size(b)
    canvas = AsciiCanvas(lines,(2*lines*width)÷height,width=width,height=height)
    @inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood!(canvas,b.<-1,:red)
    flood!(canvas,b.>1,:blue)
    !body && return canvas
    WaterLily.measure_sdf!(a,sim.body,WaterLily.time(sim))
    flood!(canvas,b.<0,:white)
    return canvas
end

ansi_moveup(n::Int) = string("\e[", n, "A")
ansi_enablecursor = "\e[?25h"
ansi_disablecursor = "\e[?25l"

function sim_ascii!(sim;duration=10,step=0.25,remeasure=false,lines=10,body=false)
    t₀ = round(sim_time(sim))
    print(ansi_disablecursor,repeat("\n",lines+1))
    for tᵢ in range(t₀,t₀+duration;step)
        sim_step!(sim,tᵢ;remeasure)
        println(ansi_moveup(lines),ascii_vort(sim,lines,body))
    end
    print(ansi_enablecursor)
end

using WaterLily
function circle(radius=12;Re=250,n=(12,6),U=1)
    center = radius*n[2]/2
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation(radius .* n, (U,0), radius; ν=U*radius/Re, body)
end
sim = circle(); sim_ascii!(sim,lines=12,duration=20,body=true)