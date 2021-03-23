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
begin
	using WaterLily, PlutoUI
	using LinearAlgebra: norm2
	include("../examples/TwoD_plots.jl")
end;

# ╔═╡ 5a2b2292-8bd9-11eb-1478-75b6bd59dc7b
md"Define the size and angle $\alpha$ of a square cylinder" 

# ╔═╡ 63399f40-8bdd-11eb-35a9-1b23580343f8
@bind size PlutoUI.Slider(10.:30., default=16.)

# ╔═╡ 7e80b328-8bde-11eb-160f-73a67b8db0a2
Text(string("size=",size))

# ╔═╡ 7ac5d8a4-8a43-11eb-2ef0-e3c0aa4737b0
@bind α PlutoUI.Slider(range(0.,π/4,step=π/16), default=0.)

# ╔═╡ f956022e-8bde-11eb-311b-2dfb01626549
Text(α==0 ? "α=0" : string("α=π/",floor(Int,π/α))) 

# ╔═╡ 1a9e4234-8bda-11eb-21f0-894c61269b18
md"Click `Start` (and then `stop`) to run (and then pause) the simulation."

# ╔═╡ 6612b570-8a80-11eb-20e3-69d56c5ff765
@bind tick PlutoUI.Clock(0.001)

# ╔═╡ d48df7f8-8bd9-11eb-2f0b-1d3f58b046df
md"Define the signed distance function and the `Simulation`. Note that you need to use `norm2`, as it plays nicely with ForwardDiff."

# ╔═╡ dbfb0b2c-8be6-11eb-2f47-2d817a6def83
n,m = 3*2^6,2^7;

# ╔═╡ dbb2dd16-8be6-11eb-2682-e3a289d36c15
body = AutoBody() do x,t
	x = [cos(α) sin(α); -sin(α) cos(α)] * (x .- m/2)
	x = abs.(x) .- size
	norm2(max.(x, 0.))+min(maximum(x),0.)
end;

# ╔═╡ b9c97692-8be6-11eb-1c4a-29c67b3b354d
sim = Simulation((n+2,m+2),[1.,0.],size; body, ν=size/100.);

# ╔═╡ b8e4b74e-8a3a-11eb-27a0-87eea85497d9
begin
	tick
	sim_step!(sim,sim_time(sim)+0.25)
	@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
	flood(sim.flow.σ,clims=(-10,10))
end

# ╔═╡ Cell order:
# ╟─5a2b2292-8bd9-11eb-1478-75b6bd59dc7b
# ╟─63399f40-8bdd-11eb-35a9-1b23580343f8
# ╟─7e80b328-8bde-11eb-160f-73a67b8db0a2
# ╟─7ac5d8a4-8a43-11eb-2ef0-e3c0aa4737b0
# ╟─f956022e-8bde-11eb-311b-2dfb01626549
# ╟─1a9e4234-8bda-11eb-21f0-894c61269b18
# ╟─6612b570-8a80-11eb-20e3-69d56c5ff765
# ╟─b8e4b74e-8a3a-11eb-27a0-87eea85497d9
# ╟─d48df7f8-8bd9-11eb-2f0b-1d3f58b046df
# ╠═bc0bf1e8-8a36-11eb-3cc2-ff079d49df5b
# ╠═dbfb0b2c-8be6-11eb-2f47-2d817a6def83
# ╠═dbb2dd16-8be6-11eb-2682-e3a289d36c15
# ╠═b9c97692-8be6-11eb-1c4a-29c67b3b354d
