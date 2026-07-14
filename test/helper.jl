## Shared helpers in the main test suite. Use it for reusable code throughout different test files

# Taylor-Green vortex simulation (`flow` and `forwarddiff` test files)
function TGVsim(mem;perdir=(1,2),Re=1e8,T=typeof(Re))
    # Define vortex size, velocity, viscosity
    L = 64; κ = T(2π/L); ν = T(1/(κ*Re));
    # TGV vortex in 2D
    function TGV(i,xy,t,κ,ν)
        x,y = @. (xy)*κ  # scaled coordinates
        i==1 && return -sin(x)*cos(y)*exp(-2*κ^2*ν*t) # u_x
        return          cos(x)*sin(y)*exp(-2*κ^2*ν*t) # u_y
    end
    # Initialize simulation
    return Simulation((L,L),(i,x,t)->TGV(i,x,t,κ,ν),L;U=1,ν,T,mem,perdir),TGV
end

# Laminar boundary-layer flow (`flow`, `metrics` and `ext` test files)
make_bl_flow(L=32;T=Float32,mem=Array) = Simulation((L,L),
    (i,x,t)-> i==1 ? convert(Float32,4.0*(((x[2]+0.5)/2L)-((x[2]+0.5)/2L)^2)) : 0.f0,
    L;ν=0.001,U=1,mem,T,exitBC=false
) # fails with exitBC=true, but the profile is maintained

# Constant-jerk accelerating flow (`flow` and `les` test files): periodic in x, Neumann in y.
# `use_g=true` drives it with the built-in body force `g`; otherwise pair it with the `gravity!`
# udf below to drive the same acceleration through the user-defined-function path.
function acceleratingFlow(N;use_g=false,T=Float64,perdir=(1,),jerk=4,mem=Array)
    # assuming gravitational scale is 1 and Fr is 1, U scale is Fr*√gL
    UScale = √N  # this is also initial U
    # constant jerk in x, zero acceleration in y (g must be type-stable: return T in both branches)
    g(i,x,t) = i==1 ? t*jerk : 0.
    !use_g && (g = nothing)
    return WaterLily.Simulation((N,N), (UScale,0.), N; ν=0.001, g, Δt=0.001, perdir, T, mem), jerk
end
gravity!(flow::AbstractFlow,t; jerk=4) = for i ∈ 1:last(size(flow.f))
    WaterLily.@loop flow.f[I,i] += i==1 ? t*jerk : 0 over I ∈ CartesianIndices(Base.front(size(flow.f)))
end
