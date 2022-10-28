module MultistateModels

using ArraysOfArrays
using Chain
using DataFrames
using Distributions
using ElasticArrays
using ExponentialUtilities
using ForwardDiff
using LinearAlgebra
using Optim # for simulation - keep
using Optimization # for fitting - keep
using OptimizationOptimJL
using StatsBase
using StatsModels
using StatsFuns

# Write your package code here.
export
    @formula,
    Hazard,
    multistatemodel,
    set_parameters!,
    simulate

# typedefs
include("common.jl")

# helpers
include("helpers.jl")

# hazard related functions
include("hazards.jl")

# likelihood functions
include("likelihoods.jl")

# miscellaneous functions
include("miscellaneous.jl")

# model fitting
include("modelfitting.jl")

# model generation
include("modelgeneration.jl")

# path functions
include("pathfunctions.jl")

# sampling functions
include("sampling.jl")

# simulation
include("simulation.jl")

end
