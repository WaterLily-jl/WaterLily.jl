### A Pluto.jl notebook ###
# v0.12.12

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ bc0bf1e8-8a36-11eb-3cc2-ff079d49df5b
using WaterLily, PlutoUI

# ╔═╡ c9eca20a-8a82-11eb-1e65-c735d7bee4c0
include("../examples/TwoD_plots.jl");

# ╔═╡ 7ac5d8a4-8a43-11eb-2ef0-e3c0aa4737b0
radius=16

# ╔═╡ 3e4f5c2a-8a3a-11eb-0465-433b036a7086
body = AutoBody((x,t)-> √sum(abs2, x .- 4radius) - radius);

# ╔═╡ 1dcb3cb0-8a44-11eb-18cd-cf841f9222b1
sim = Simulation((12radius+2,8radius+2),[1.,0.],radius; body, ν=radius/200);

# ╔═╡ 6612b570-8a80-11eb-20e3-69d56c5ff765
@bind tick PlutoUI.Clock(0.001)

# ╔═╡ b8e4b74e-8a3a-11eb-27a0-87eea85497d9
begin
	tick
	sim_step!(sim,sim_time(sim)+0.25)
	@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
	flood(sim.flow.σ,clims=(-10,10))
end

# ╔═╡ Cell order:
# ╠═bc0bf1e8-8a36-11eb-3cc2-ff079d49df5b
# ╠═c9eca20a-8a82-11eb-1e65-c735d7bee4c0
# ╠═7ac5d8a4-8a43-11eb-2ef0-e3c0aa4737b0
# ╠═3e4f5c2a-8a3a-11eb-0465-433b036a7086
# ╠═1dcb3cb0-8a44-11eb-18cd-cf841f9222b1
# ╠═6612b570-8a80-11eb-20e3-69d56c5ff765
# ╠═b8e4b74e-8a3a-11eb-27a0-87eea85497d9
