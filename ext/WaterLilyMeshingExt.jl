module WaterLilyMeshingExt

using Makie, Meshing, WaterLily
using Makie.GeometryBasics
import WaterLily: get_body, plot_body_obs!, isosurface_mesh

"""
    isosurface_mesh(a, level)

Triangulate the `level` set of the 3D array `a` with marching cubes and return it as a GeometryBasics.Mesh object
which can be rendered with Makie.mesh. The mesh is empty if `a` does not cross `level`.
"""
function isosurface_mesh(a::Array{T,3}, level) where T
    ranges = range.((0, 0, 0), size(a))
    points, faces = Meshing.isosurface(a, Meshing.MarchingCubes(iso=T(level)), ranges...)
    GeometryBasics.Mesh(Point3.(points), GLTriangleFace.(faces))
end

"""
    get_body(sdf_array, ::Val{true})

Gets a 3D signed distance function array and returns a GeometryBasics.Mesh object which can be rendered with Makie.mesh
This function is only called when passing body2mesh=true to viz!
"""
get_body(sdf_array::Array{T,3} where T, ::Val{true}) = isosurface_mesh(sdf_array, 0)

"""
    plot_body_obs!(ax, body_mesh; color=:black)

Plot the 3D body mesh `body_mesh::Observable{GeometryBasics.Mesh}` in a 3D axis.
"""
plot_body_obs!(ax, body_mesh; color=(:grey, 0.9)) = Makie.mesh!(ax, body_mesh;
    shading=true, color
)

end # module