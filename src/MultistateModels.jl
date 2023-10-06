module MultistateModels

using ArraysOfArrays
using DataFrames
using DiffResults
using Distributions
using ElasticArrays
using ExponentialUtilities
using ForwardDiff
using LinearAlgebra
using OrderedCollections
using Optim # for simulation - keep
using Optimization # for fitting - keep
using OptimizationOptimJL
using QuadGK
using RCall
using StatsBase
using StatsModels
using StatsFuns

# need to import fit to overload and reexport it
import StatsBase.fit

# initialize R session for splines
function __init__()
    @eval RCall.R"if (!require('splines2', quietly=TRUE)) install.packages('splines2')"
    @eval RCall.@rlibrary splines2
    nothing
end

# Write your package code here.
export
    @formula,
    compute_hazard,
    compute_cumulative_hazard,
    cumulative_incidence,
    fit,
    GetConvergenceRecords,
    GetLoglik,
    GetParameters,
    GetVcov,
    Hazard,
    multistatemodel,
    set_crude_init!,
    set_parameters!,
    simulate,
    statetable,
    summary,
    __init__

# typedefs
include("common.jl")

# helpers
include("helpers.jl")

# hazard related functions
include("hazards.jl")

# crude parameter initialization functions
include("initiation.jl")

# likelihood functions
include("likelihoods.jl")

# Monte Carlo EM functions
include("mcem.jl")

# miscellaneous functions
include("miscellaneous.jl")

# model fitting
include("modelfitting.jl")

# model generation
include("modelgeneration.jl")

# model output
include("modeloutput.jl")

# path functions
include("pathfunctions.jl")

# sampling functions
include("sampling.jl")

# simulation
include("simulation.jl")

# smooths
include("smooths.jl")

end
