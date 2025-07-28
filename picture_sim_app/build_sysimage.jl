#!/usr/bin/env julia
"""
Build Julia Sysimage for WaterLily Simulations

This script creates a precompiled Julia sysimage containing all the packages
and code needed for the WaterLily simulations. This improves
startup time and performance for repeated simulations.

Usage:
    julia build_sysimage.jl

The resulting sysimage will be saved as 'julia_sysimage_pixelbody.so'
"""

using Pkg

# Ensure all required packages are installed
required_packages = [
    "WaterLily",
    "StaticArrays", 
    "StatsBase",     # Used in TestPixelCamSim.jl
    "Plots",
    "ReadVTK",
    "WriteVTK",
    "PlutoUI",
    "CUDA",          # Optional CUDA support
    "GLMakie",       # For particle plotting
    "ImageMorphology", # For image processing in plot_particles.jl
    "PackageCompiler"
]

println("🔍 Checking required packages...")
available_packages = String[]
for pkg in required_packages
    try
        if pkg == "CUDA"
            # CUDA is optional - only add if available
            try
                Pkg.add(pkg)
                push!(available_packages, pkg)
                println("✓ $pkg (optional)")
            catch
                println("⚠️  $pkg (optional - skipped)")
            end
        else
            Pkg.add(pkg)
            push!(available_packages, pkg)
            println("✓ $pkg")
        end
    catch e
        println("Failed to add $pkg: $e")
        if pkg in ["WaterLily", "StaticArrays", "StatsBase", "Plots"]
            error("Critical package $pkg failed to install. Cannot continue.")
        end
    end
end

println("\nPackages available for sysimage: $(length(available_packages))")
for pkg in available_packages
    println("  - $pkg")
end

# Load PackageCompiler
using PackageCompiler

# Path to the test script for sysimage creation
script_dir = dirname(@__FILE__)
# Use a simpler test script that doesn't have external path dependencies
test_script = joinpath(script_dir, "test_sysimage.jl")
sysimage_path = joinpath(script_dir, "julia_sysimage_pixelbody.so")

println("\n🏗️  Building sysimage...")
println("Test script: $test_script")
println("Sysimage path: $sysimage_path")

# Check if the test script exists
if !isfile(test_script)
    error("Test script not found: $test_script")
end

try
    # Create the sysimage with available packages
    create_sysimage(
        available_packages,
        sysimage_path=sysimage_path,
        precompile_execution_file=test_script,
        # precompile_statements_file="precompile_statements.jl",  # Optional: add custom precompile statements
    )
    
    println("\nSysimage created successfully!")
    println("Location: $sysimage_path")
    println("Size: $(round(stat(sysimage_path).size / 1024^2, digits=1)) MB")
    
    println("\nPerformance Benefits:")
    println("   - Faster Julia startup (seconds → milliseconds)")
    println("   - Pre-compiled package code")
    println("   - Optimized for repeated simulations")
    
    println("\nUsage:")
    println("   python live_simulation.py")
    println("   # The script will automatically use the sysimage")
    
catch e
    println("\nFailed to create sysimage: $e")
    println("\nTroubleshooting:")
    println("   1. Make sure all packages are properly installed")
    println("   2. Check that TestPixelCamSim.jl runs without errors")
    println("   3. Try running this script again")
    println("\nAlternative: Run without sysimage")
    println("   The live simulation will still work but be slower")
end
