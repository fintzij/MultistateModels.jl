using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Printf

println("Starting SQUAREM Benchmark...")

println("\nRunning TVC Long Tests (SQUAREM)...")
t_tvc = @elapsed include("../MultistateModelsTests/longtests/longtest_mcem_tvc.jl")
println("TVC Tests took: $t_tvc seconds")

println("\nRunning Spline Long Tests (SQUAREM)...")
t_splines = @elapsed include("../MultistateModelsTests/longtests/longtest_mcem_splines.jl")
println("Spline Tests took: $t_splines seconds")

println("\nTotal Time: $(t_tvc + t_splines) seconds")
