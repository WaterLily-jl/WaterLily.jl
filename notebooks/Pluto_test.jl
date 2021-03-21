### A Pluto.jl notebook ###
# v0.12.12

using Markdown
using InteractiveUtils

# ╔═╡ bc0bf1e8-8a36-11eb-3cc2-ff079d49df5b
using WaterLily

# ╔═╡ 419fdcec-8a38-11eb-2f8c-fdede732476d
include("../examples/TwoD_plots.jl");

# ╔═╡ 7ac5d8a4-8a43-11eb-2ef0-e3c0aa4737b0
radius=24

# ╔═╡ 3e4f5c2a-8a3a-11eb-0465-433b036a7086
body = AutoBody((x,t)-> √sum(abs2, x .- 2radius) - radius);

# ╔═╡ 1dcb3cb0-8a44-11eb-18cd-cf841f9222b1
sim = Simulation((6radius+2,4radius+2),[1.,0.],radius; body, ν=radius/1000);

# ╔═╡ b8e4b74e-8a3a-11eb-27a0-87eea85497d9
begin 
	sim_step!(sim,sim_time(sim)+1)
	@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
	flood(sim.flow.σ)
end

# ╔═╡ Cell order:
# ╠═bc0bf1e8-8a36-11eb-3cc2-ff079d49df5b
# ╠═419fdcec-8a38-11eb-2f8c-fdede732476d
# ╠═7ac5d8a4-8a43-11eb-2ef0-e3c0aa4737b0
# ╠═3e4f5c2a-8a3a-11eb-0465-433b036a7086
# ╠═1dcb3cb0-8a44-11eb-18cd-cf841f9222b1
# ╠═b8e4b74e-8a3a-11eb-27a0-87eea85497d9
