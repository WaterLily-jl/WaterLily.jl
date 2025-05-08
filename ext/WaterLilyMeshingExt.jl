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

function get_body(sdf_array, ::Val{true})
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