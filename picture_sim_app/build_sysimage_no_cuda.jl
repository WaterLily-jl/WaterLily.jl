#!/usr/bin/env julia
"""
Build Julia Sysimage for WaterLily Simulations (CUDA-free version)

This script creates a precompiled Julia sysimage containing all the packages
and code needed for WaterLily simulations, but excludes CUDA to avoid
initialization errors on systems without CUDA support.

Usage:
    julia build_sysimage_no_cuda.jl

The resulting sysimage will be saved as 'julia_sysimage_pixelbody_no_cuda.so'
"""

using Pkg

# Ensure all required packages are installed (excluding CUDA)
required_packages = [
    "WaterLily",
    "StaticArrays", 
    "StatsBase",     # Used in TestPixelCamSim.jl
    "Plots",
    "ReadVTK",
    "WriteVTK",
    "PlutoUI",
    "GLMakie",       # For particle plotting
    "ImageMorphology", # For image processing in plot_particles.jl
    "PackageCompiler"
]

println("🔍 Checking required packages (CUDA excluded)...")
available_packages = String[]
for pkg in required_packages
    try
        Pkg.add(pkg)
        push!(available_packages, pkg)
        println("✓ $pkg")
    catch e
        println("❌ Failed to add $pkg: $e")
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
sysimage_path = joinpath(script_dir, "julia_sysimage_pixelbody_no_cuda.so")

println("\n🏗️  Building CUDA-free sysimage...")
println("Test script: $test_script")
println("Sysimage path: $sysimage_path")

# Check if the test script exists
if !isfile(test_script)
    error("Test script not found: $test_script")
end

try
    # Create the sysimage with available packages (no CUDA)
    create_sysimage(
        available_packages,
        sysimage_path=sysimage_path,
        precompile_execution_file=test_script,
    )
    
    println("\n✅ CUDA-free sysimage created successfully!")
    println("📁 Location: $sysimage_path")
    println("📏 Size: $(round(stat(sysimage_path).size / 1024^2, digits=1)) MB")
    
    println("\n🚀 Performance Benefits:")
    println("   - Faster Julia startup (seconds → milliseconds)")
    println("   - Pre-compiled package code")
    println("   - Optimized for repeated simulations")
    println("   - No CUDA dependency issues")
    
    println("\n💡 Usage:")
    println("   The live simulation will automatically detect and use this sysimage")
    println("   python live_simulation.py")
    
catch e
    println("\n❌ Failed to create sysimage: $e")
    println("\n🔧 Troubleshooting:")
    println("   1. Make sure all packages are properly installed")
    println("   2. Check that test_sysimage.jl runs without errors")
    println("   3. Try running this script again")
    println("\n💡 Alternative: Run without sysimage")
    println("   The live simulation will still work but be slower")
end
