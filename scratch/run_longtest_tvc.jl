using Pkg
Pkg.activate(".")
using MultistateModels
using Test
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Printf

# Run the test
include("../MultistateModelsTests/longtests/longtest_mcem_tvc.jl")
