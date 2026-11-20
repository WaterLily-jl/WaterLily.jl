using WaterLily
using Test
using Plots

function make_egg_noarray(ra=40, rb=20, U=1)
    Simulation(
        (5 * ra, 5 * rb),
        (U, 0),
        rb;
        body=AutoBody(
            (x, t) -> √sum(
                (x[1] - 8)^2 / ra +
                (x[2] - 8)^2 / rb
            ) - 1
        )
    )
end

@testset "WaterLilyPlotsExt" begin
    # Test body_plot!
    mktempdir() do dir
        # body_plot!
        sim = make_egg_noarray()
        plots = plot()
        
        @test_nowarn body_plot!(sim)

        body_path = joinpath(dir, "body.png")
        @test_nowarn savefig(plots, body_path)
        @test isfile(body_path)
        @test filesize(body_path) > 0

        # flood
        field = [x^2 / 2 + y^2 / 3 
        for x in -1:0.01:1, y in -1:0.01:1]

        flood_plot = flood(field; clims=())
        flood_path = joinpath(dir, "flood.png")
        @test_nowarn savefig(flood_plot, flood_path)
        @test isfile(flood_path)
        @test filesize(flood_path) > 0

        # sim_gif!
        sim = make_egg_noarray()
        gif_path = joinpath(dir, "simulation.gif")

        sim_gif!(
            sim;
            duration=0.2,
            step=0.1,
            verbose=false,
            clims=(-5, 5),
            plotbody=true,
            video = gif_path
        )

        @test isfile(gif_path)
        @test filesize(gif_path) > 0

        # plot_logger
        sim = make_egg_noarray()
        log_base = joinpath(dir, "testLog")
        log_path = log_base * ".log"

        WaterLily.logger(log_base)
        sim_step!(sim, 0.5)

        @test isfile(log_path)

        log_plot = plot_logger(log_path)
        log_image = joinpath(dir, "log.png")

        @test_nowarn savefig(log_plot, log_image)
        @test isfile(log_image)
        @test filesize(log_image) > 0
    end
end