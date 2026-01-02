using Pkg
Pkg.activate(".")
using MultistateModels
using Test
using DataFrames
using Distributions
using LinearAlgebra
using Random

# Include TestFixtures
include("../MultistateModelsTests/fixtures/TestFixtures.jl")
using .TestFixtures

# Run the test
include("../MultistateModelsTests/unit/test_hazards.jl")
