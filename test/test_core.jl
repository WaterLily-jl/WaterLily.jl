@testset "core.jl" begin
    I = CartesianIndex(1,2,3,4)
    @test I+δ(3,I) == CartesianIndex(1,2,4,4)
    @test WaterLily.CI(I,5)==CartesianIndex(1,2,3,4,5)
    @test WaterLily.CIj(3,I,5)==CartesianIndex(1,2,5,4)
    @test WaterLily.CIj(2,CartesianIndex(16,16,16,3),14)==CartesianIndex(16,14,16,3)

    @test loc(3,CartesianIndex(3,4,5)) == SVector(3,4,4.5) .- 1.5
    I = CartesianIndex(rand(2:10,3)...)
    @test loc(0,I) == SVector(I.I...) .- 1.5

    ex,sym = :(a[I,i] = Math.add(p.b[I],func(I,q))),[]
    WaterLily.grab!(sym,ex)
    @test ex == :(a[I, i] = Math.add(b[I], func(I, q)))
    @test sym == [:a, :I, :i, :(p.b), :q]
    sym = [:a, :b, :c]
    @test WaterLily.joinsymtype(sym,[:A,:B,:C]) == Expr[:(a::A), :(b::B), :(c::C)]

    for f ∈ arrays
        Ng, D, U = (6, 6), 2, (1.0, 0.5)
        u = rand(Ng..., D) |> f # vector
        σ = rand(Ng...) |> f # scalar
        BC!(u, U)
        @test GPUArrays.@allowscalar all(u[1, :, 1] .== U[1]) && all(u[2, :, 1] .== U[1]) && all(u[end, :, 1] .== U[1]) &&
            all(u[3:end-1, 1, 1] .== u[3:end-1, 2, 1]) && all(u[3:end-1, end, 1] .== u[3:end-1, end-1, 1])
        @test GPUArrays.@allowscalar all(u[:, 1, 2] .== U[2]) && all(u[:, 2, 2] .== U[2]) && all(u[:, end, 2] .== U[2]) &&
            all(u[1, 3:end-1, 2] .== u[2, 3:end-1, 2]) && all(u[end, 3:end-1, 2] .== u[end-1, 3:end-1, 2])

        GPUArrays.@allowscalar u[end,:,1] .= 3
        BC!(u,U,true) # save exit values
        @test GPUArrays.@allowscalar all(u[end, :, 1] .== 3)

        WaterLily.exitBC!(u,u,0) # conservative exit check
        @test GPUArrays.@allowscalar all(u[end,2:end-1, 1] .== U[1])

        # test BC with function
        Ubc(i,x,t) = i==1 ? 1.0 : 0.5
        v = rand(Ng..., D) |> f # vector
        BC!(v,Ubc,false); BC!(u,U,false) # make sure we apply the same
        @test GPUArrays.@allowscalar all(v[1, :, 1] .== u[1, :, 1]) && all(v[2, :, 1] .== u[2, :, 1]) && all(v[end, :, 1] .== u[end, :, 1])
        @test GPUArrays.@allowscalar all(v[:, 1, 2] .== u[:, 1, 2]) && all(v[:, 2, 2] .== u[:, 2, 2]) && all(v[:, end, 2] .== u[:, end, 2])
        # test exit bc
        GPUArrays.@allowscalar v[end,:,1] .= 3
        BC!(v,Ubc,true) # save exit values
        @test GPUArrays.@allowscalar all(v[end, :, 1] .== 3)

        BC!(u,U,true,(2,)) # periodic in y and save exit values
        @test GPUArrays.@allowscalar all(u[:, 1:2, 1] .== u[:, end-1:end, 1]) && all(u[:, 1:2, 1] .== u[:,end-1:end,1])
        WaterLily.perBC!(σ,(1,2)) # periodic in two directions
        @test GPUArrays.@allowscalar all(σ[1, 2:end-1] .== σ[end-1, 2:end-1]) && all(σ[2:end-1, 1] .== σ[2:end-1, end-1])

        u = rand(Ng..., D) |> f # vector
        BC!(u,U,true,(1,)) #saveexit has no effect here as x-periodic
        @test GPUArrays.@allowscalar all(u[1:2, :, 1] .== u[end-1:end, :, 1]) && all(u[1:2, :, 2] .== u[end-1:end, :, 2]) &&
                            all(u[:, 1, 2] .== U[2]) && all(u[:, 2, 2] .== U[2]) && all(u[:, end, 2] .== U[2])
        # test non-uniform BCs
        Ubc_1(i,x,t) = i==1 ? x[2] : x[1]
        v .= 0; BC!(v,Ubc_1)
        # the tangential BC change the value of the ghost cells on the other axis, so we cannot check it
        @test GPUArrays.@allowscalar all(v[1,2:end-1,1] .≈ v[end,2:end-1,1])
        @test GPUArrays.@allowscalar all(v[2:end-1,1,2] .≈ v[2:end-1,end,2])
        # more complex
        Ng, D = (8, 8, 8), 3
        u = zeros(Ng..., D) |> f # vector
        Ubc_2(i,x,t) = i==1 ? cos(2π*x[1]/8) : i==2 ? sin(2π*x[2]/8) : tan(π*x[3]/16)
        BC!(u,Ubc_2)
        @test GPUArrays.@allowscalar all(u[1,:,:,1] .≈ cos(-1π/4))  && all(u[2,:,:,1] .≈ cos(0)) && all(u[end,:,:,1] .≈ cos(6π/4))
        @test GPUArrays.@allowscalar all(u[:,1,:,2] .≈ sin(-1π/4))  && all(u[:,2,:,2] .≈ sin(0)) && all(u[:,end,:,2] .≈ sin(6π/4))
        @test GPUArrays.@allowscalar all(u[:,:,1,3] .≈ tan(-1π/16)) && all(u[:,:,2,3] .≈ tan(0)) && all(u[:,:,end,3].-tan(6π/16).<1e-6)
    end
end
