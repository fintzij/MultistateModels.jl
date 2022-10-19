module MultistateModels

using ArraysOfArrays
using Chain
using DataFrames
# using DifferentialEquations
using Distributions
using ExponentialUtilities
using ForwardDiff
using LinearAlgebra
using Optim # for simulation
using Optimization # for fitting
using OptimizationOptimJL
# using PreallocationTools
# using SciMLSensitivity
# using QuadGK # going to change this eventually
# using Integrals # replaces QuadGK
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

# model fitting
include("modelfitting.jl")

# model generation
include("modelgeneration.jl")

# path functions
include("pathfunctions.jl")

# simulation
include("simulation.jl")

end
