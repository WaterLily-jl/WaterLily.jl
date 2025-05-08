module WaterLilyMeshingExt

if isdefined(Base, :get_extension)
    using GLMakie, Meshing, GeometryBasics
else
    using ..GLMakie
    using ..Meshing
    using ..GeometryBasics
end

using WaterLily
import WaterLily: get_body, plot_body_obs!

"""
    get_body(sdf_array, ::Val{true})

Gets a 3D signed distance function array and returns a GeometryBasics.Mesh object which can be rendered with GLMakie.mesh.
This function is only called when passing body2mesh=true to viz!.
"""
function get_body(sdf_array::Array{T,3} where T, ::Val{true})
    ranges = range.((0, 0, 0), size(sdf_array))
    points, faces = Meshing.isosurface(sdf_array, Meshing.MarchingCubes(iso=0), ranges...)
    GeometryBasics.Mesh(Point3.(points), GLTriangleFace.(faces))
end

"""
    plot_body_obs!(ax, body_mesh; color=:black)

Plot the 3D body mesh `body_mesh::Observable{GeometryBasics.Mesh}` in a 3D axis.
"""
plot_body_obs!(ax, body_mesh; color=(:grey, 0.9)) = Makie.mesh!(ax, body_mesh;
    shading=MultiLightShading, color
)

end # module