using WaterLily

# overwrites the function so we can see where we aplly it
@inline function WaterLily.ϕu∂(a,I,f,u,λ=WaterLily.quick) 
    if u>0
        println("CD at ", I," ",a)
        return u*WaterLily.ϕ(a,I,f)
    else
        println("Quick at ", I," ",a)
        return u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I])
    end
end

# Initialize simulation
L = 8; Re = 100;
U = (1.0,1.0)
sim = Simulation((L, L), U, L; ν=L/Re, T=Float64)

# update flow
WaterLily.conv_diff!(sim.flow.f,sim.flow.u⁰,sim.flow.σ;ν=sim.flow.ν)
# should use quick on the top and right boundaries, and CD on the bottom and left