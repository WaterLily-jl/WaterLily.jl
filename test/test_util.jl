@testset "util.jl" begin
    for f ∈ arrays
        a = zeros(Float32,8,8,2) |> f; b = zeros(Float64,8,8) |> f
        apply!((i,x)->x[i],a); apply!(x->x[1],b) # offset for start of grid
        @test GPUArrays.@allowscalar all(WaterLily.interp(SVector(2.5f0,1.f0),a) .≈ [2.5f0,1.0f0])
        @test GPUArrays.@allowscalar all(WaterLily.interp(SVector(3.5f0,3.f0),a) .≈ [3.5f0,3.0f0])
        @test GPUArrays.@allowscalar eltype(WaterLily.interp(SVector(2.5f0,1.f0),a))==Float32
        @test_throws MethodError GPUArrays.@allowscalar WaterLily.interp(SVector(2.50,1.0),a)
        @test GPUArrays.@allowscalar WaterLily.interp(SVector(2.5,1),b) ≈ 2.5
        @test GPUArrays.@allowscalar WaterLily.interp(SVector(3.5,3),b) ≈ 3.5
        @test GPUArrays.@allowscalar eltype(WaterLily.interp(SVector(3.5,3),b))==Float64
        @test_throws MethodError GPUArrays.@allowscalar WaterLily.interp(SVector(2.5f0,1.f0),b)
        @test GPUArrays.@allowscalar all(WaterLily.interp(SVector(-1.f0,4.f0),a) .≈ Float32[-0.5, 4])
        @test GPUArrays.@allowscalar WaterLily.interp(SVector(10.,10.),b) ≈ 6.0

        src2 = rand(2,3) |> f
        dest3 = zeros(2,3,4) |> f
        WaterLily.spread!(dest3, src2; dim=3)
        @test GPUArrays.@allowscalar all(dest3[:,:,1] .≈ dest3[:,:,2] .≈ dest3[:,:,3] .≈ dest3[:,:,4] .≈ src2)
        src2 = rand(2,3,2) |> f
        dest3 = zeros(2,3,4,3) |> f
        WaterLily.spread!(dest3, src2; dim=3)
        @test GPUArrays.@allowscalar all(dest3[:,:,1,1:2] .≈ dest3[:,:,2,1:2] .≈ dest3[:,:,3,1:2] .≈ dest3[:,:,4,1:2] .≈ src2)
        @test_throws MethodError WaterLily.spread!(src2, src2; dim=3)
        @test_throws MethodError WaterLily.spread!(zeros(2, 2, 4), src2; dim=3)
        @test_throws MethodError WaterLily.spread!(src2, dest3; dim=1)
        body = AutoBody((x,t)->√sum(abs2,SA[x[1]-8,x[2]-8])-6)
        sim2D = Simulation((32,16),(1.0,0.0),1.0;body,mem=f)
        apply!(x->x[1],sim2D.flow.p); apply!((i,x)->x[i],sim2D.flow.u)
        sim3D = Simulation((32,16,8),(1.0,0.0,0.0),1.0;body,perdir=(3,),mem=f)
        WaterLily.spread!(sim3D, sim2D; dim=3, ϵ=0.0)
        @test GPUArrays.@allowscalar all(sim3D.flow.u[:,:,1,1:2] .≈ sim3D.flow.u[:,:,3,1:2] .≈ sim3D.flow.u[:,:,6,1:2] .≈ sim3D.flow.u[:,:,8,1:2] .≈ sim2D.flow.u)
        @test GPUArrays.@allowscalar all(sim3D.flow.p[:,:,1] .≈ sim3D.flow.p[:,:,3] .≈ sim3D.flow.p[:,:,6] .≈ sim3D.flow.p[:,:,8] .≈ sim2D.flow.p)
        @test_throws AssertionError WaterLily.spread!(sim3D, sim2D; dim=1)
        sim3D = Simulation((32,16,8),(1.0,0.0,0.0),1.0;body=AutoBody((x,t)->√sum(abs2,x.-8)-6),perdir=(3,),mem=f)
        @test_throws AssertionError WaterLily.spread!(sim3D, sim2D; dim=3)
    end
end
