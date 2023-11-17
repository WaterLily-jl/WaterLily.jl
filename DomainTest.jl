include("BioSavart_multilevel.jl")
#Quantify domain sensitivity
circ(D,n,m;Re=200,U=1,mem=Array) = Simulation((n*D,m*D), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m*D÷2)-D÷2),ν=U*D/Re,mem)
function wake_velocity(n=5,m=3;D=64,use_biotsavart=true,t_end=100)
    sim = circ(D,n,m); ω = MLArray(sim.flow.σ)
    u = Float32[]; I = CartesianIndex(D+m*D÷2,m*D÷2)
    use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
    @assert !all(sim.pois.n .== 32) "pressure problem"
    while sim_time(sim)<t_end
        use_biotsavart ? biot_mom_step!(sim.flow,sim.pois,ω) : WaterLily.mom_step!(sim.flow,sim.pois)
        sim_time(sim)%0.1<sim.flow.Δt[end]/sim.L && push!(u,sim.flow.u[I,2])
        sim_time(sim)%1<sim.flow.Δt[end]/sim.L && @show sim_time(sim)
    end
    return sim,u
end
params = [(5,2,true) (6,3,true) (8,5,true) (10,8,true) (5,2,false) (6,3,false) (8,5,false) (10,8,false)]
using JLD2
jldopen("smalltol.jld2", "w") do file
    for (i,θ) ∈ enumerate(params)
        mygroup = JLD2.Group(file,"case$(i)")
        mygroup["θ"] = θ
        @show θ
        n,m,use_biotsavart = θ
        (sim,u) = wake_velocity(n,m;t_end=100,use_biotsavart);
        mygroup["u"] = u
        mygroup["p"] = sim.flow.p
        @inside sim.flow.σ[I] = centered_curl(3,I,sim.flow.u)*sim.L/sim.U
        mygroup["ω"] = sim.flow.σ
    end
end

using Plots,FFTW
jldopen("smalltol.jld2", "r") do file
    plt = plot();
    for i ∈ 1:length(params)
        mygroup = file["case$(i)"]
        n,m,use_biotsavart = mygroup["θ"]
        m==3 && use_biotsavart && continue
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        u = Float32.(mygroup["u"])
        plot!(plt,range(0,10,length=100),u[1:100],label="$(n)Dx$(m)D using "*BC;ls)
    end
    title!("Domain size study");xlabel!("convective time");ylabel!("v/U near centerline")
    savefig("start.png")

    plt = plot();
    for i ∈ 1:length(params)
        mygroup = file["case$(i)"]
        n,m,use_biotsavart = mygroup["θ"]
        m==3 && use_biotsavart && continue
        BC = ifelse(use_biotsavart,"Biot-Savart","Reflection")
        ls = ifelse(use_biotsavart,:solid,:dash)
        u = Float32.(mygroup["u"][700:end])
        u_hat=fft(u)./0.5length(u)
        plot!(plt,range(0,2.5,length=50),abs.(u_hat[1:50]),label="$(n)Dx$(m)D using "*BC;ls)
    end
    title!("Domain size study");xlabel!("Strouhal");ylabel!("PSD(v) near centerline")
    savefig("fft.png")
end