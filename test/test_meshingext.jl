using Makie, Meshing

@testset "WaterLilyMeshingExt.jl" begin
    n = 16
    center = (n + 1) / 2
    radius = 5

    sdf = [
        sqrt((i - center)^2 + (j - center)^2 + (k - center)^2) - radius
        for i in 1:n, j in 1:n, k in 1:n
    ]

    mesh = WaterLily.isosurface_mesh(sdf, 0)

    @test !isempty(mesh.position)
    @test !isempty(mesh.faces)
end

@testset "isosurface_mesh with no surface" begin
    no_surface = fill(1.0, 16, 16, 16)
    empty_mesh = WaterLily.isosurface_mesh(no_surface, 0)

    @test isempty(empty_mesh.position)
    @test isempty(empty_mesh.faces)
end

