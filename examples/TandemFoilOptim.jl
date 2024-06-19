using WaterLily,StaticArrays

function make_foils(φ;two=true,L=32,Re=1e3,St=0.3,αₘ=-π/18,U=1,n=8,m=4)
    # Map from simulation coordinate x to surface coordinate ξ
    nose,pivot = SA[2L,m*L//2],SA[L//4,0]
    θ₀ = αₘ+atan(π*St); h₀=L; ω=π*St*U/h₀
    function map(x,t)
        back = two && x[1]>nose[1]+2L # back body?
        ϕ = back ? φ : zero(φ)        # phase shift
        S = back ? 3L : zero(L)       # horizontal shift
        s,c = sincos(θ₀*cos(ω*t+ϕ))   # sin & cos of angle
        h = SA[S,h₀*sin(ω*t+ϕ)]       # position
        # move to origin and align with x-axis
        ξ = SA[c -s; s c]*(x-nose-h-pivot)+pivot 
        return SA[ξ[1],abs(ξ[2])]    # reflect to positive y
    end

    # Line segment SDF
    function sdf(ξ,t)
        p = ξ-SA[clamp(ξ[1],0,L),0] # vector from closest point on [0,L] segment to ξ 
        √(p'*p)-2                   # distance (with thickness offset)
    end

    Simulation((n*L,m*L),(U,0),L;ν=U*L/Re,body=AutoBody(sdf,map),T=typeof(φ))
end

drag(flow,body,t) = sum(inside(flow.p)) do I
    d,n,_ = measure(body,WaterLily.loc(0,I),t)
    flow.p[I]*n[1]*WaterLily.kern(clamp(d,-1,1))
end

function Δimpulse!(sim)
    Δt = sim.flow.Δt[end]*sim.U/sim.L
    sim_step!(sim)
    Δt*drag(sim.flow,sim.body,WaterLily.time(sim))
end

function mean_drag(φ,two=true,St=0.3,N=3,period=2N/St)
    sim = make_foils(φ;two,St)
    sim_step!(sim,period) # warm-in transient period
    impulse = 0           # integrate impulse
    while sim_time(sim)<2period
        impulse += Δimpulse!(sim)
    end
    impulse/period        # return mean drag
end

using Optim
θ = Optim.minimizer(optimize(x->-mean_drag(first(x)), [0f0], Newton(),
    Optim.Options(show_trace=true,f_tol=1e-2); autodiff = :forward))