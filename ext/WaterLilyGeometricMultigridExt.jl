module WaterLilyGeometricMultigridExt

if isdefined(Base, :get_extension)
    using GeometricMultigrid
else
    using ..GeometricMultigrid
end

using WaterLily
import WaterLily: AbstractPoisson,GeomMultigridPoisson,solver!,update!,L₂,L∞

struct GMGPoisson{T,Vf<:AbstractArray{T},Mf<:AbstractArray{T},St<:SolveState} <: AbstractPoisson{T,Vf,Mf}
    x::Vf  # WaterLily approximate solution
    L::Mf  # WaterLily lower diagonal coefficients
    z::Vf  # WaterLily source
    st::St
    function GMGPoisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};perdir=()) where T
        st = SolveState(GeometricMultigrid.Poisson(L),FieldVector(x),FieldVector(z))
        new{T,typeof(x),typeof(L),typeof(st)}(x,L,z,GeometricMultigrid.fill_children!(st))
    end
end
GeomMultigridPoisson(x,L,z;perdir=()) = GMGPoisson(x,L,z;perdir=perdir)
export GeomMultigridPoisson

update!(p::GMGPoisson) = for I ∈ p.st.A.R
    p.st.A.D[I] = GeometricMultigrid.calcdiag(I,p.st.A.L)
end

function solver!(p::GMGPoisson;tol=1e-4,itmx=32,kw...)
    GeometricMultigrid.resid!(p.st.r,p.st.A,p.st.x);
    GeometricMultigrid.iterate!(p.st,GeometricMultigrid.Vcycle!;mxiter=Int(itmx),abstol=tol,kw...)
end

L₂(p::GMGPoisson) = WaterLily.L₂(p.z)
L∞(p::GMGPoisson) = maximum(abs,p.z)

end #module