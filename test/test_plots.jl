using WaterLily
using Test
using Plots
using FileIO
using ImageCore
using ImageMagick

function make_egg_noarray(ra=40, rb=20, U=1)
    Simulation(
        (5 * ra, 5 * rb),
        (U, 0),
        rb;
        body = AutoBody(
            (x, t) -> √sum((x[1] - 8)^2 / ra + (x[2] - 8)^2 / rb) - 1
        )
    )
end

function image_error(test_image, reference_image)
    test_rgb = RGB.(test_image)
    reference_rgb = RGB.(reference_image)

    @test size(test_rgb) == size(reference_rgb)

    test_pixels = Float32.(channelview(test_rgb))
    reference_pixels = Float32.(channelview(reference_rgb))

    sum(abs.(test_pixels .- reference_pixels)) / length(test_pixels)
end

@testset "WaterLilyPlotsExt" begin
    Plots.plot()

    sim = make_egg_noarray()
    result = body_plot!(sim)

    output_path = "temp_body_plot.png"
    Plots.savefig(result, output_path)

    generated = load(output_path)
    reference = load("test/compdata/compbody.png")

    @test image_error(generated, reference) < 0.01

    rm(output_path)

    # Test sim_gif!
    Plots.plot()

    sim = make_egg_noarray()
    testgif = sim_gif!(
        sim;
        duration=0.5,
        clims=(-5, 5),
        plotbody=true
    )

    gif_path = testgif.filename
    generated = load(gif_path)
    reference = load("test/compdata/compgif.gif")

    @test image_error(generated, reference) < 0.01

    rm(gif_path)


    # Test flood
    Plots.plot()

    xr = -1:0.01:1
    yr = -1:0.01:1
    g(x, y) = x^2 / 2 + y^2 / 3

    flood(g.(xr, yr'))

    flood_path = "temp_flood.png"
    Plots.savefig(flood_path)

    generated = load(flood_path)
    reference = load("test/compdata/floodTestNoLims.png")

    @test image_error(generated, reference) < 0.01

    rm(flood_path)


    # Test plot_logger
    Plots.plot()

    sim = make_egg_noarray()

    WaterLily.logger("testLog")
    sim_step!(sim, 0.5)

    plot_logger("testLog.log")

    log_path = "temp_log_plot.png"
    Plots.savefig(log_path)

    generated = load(log_path)
    reference = load("test/compdata/compLogPlot.png")

    @test image_error(generated, reference) < 0.01

    rm(log_path)
    rm("testLog.log")
end