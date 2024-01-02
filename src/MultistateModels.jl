module MultistateModels

using ArraysOfArrays
using Chain
using DataFrames
using DiffResults
using Distributions
using ElasticArrays
using ExponentialUtilities
using ForwardDiff
using LinearAlgebra
using MacroTools
using Optim # for simulation - keep
using Optimization # for fitting - keep
using OrderedCollections
using OptimizationOptimJL
using ParetoSmooth
using Preferences
using QuadGK
using RCall
using RuntimeGeneratedFunctions
using StatsBase
using StatsFuns
using StatsModels

# make sure ForwardDiff is nan safe
Preferences.set_preferences!(ForwardDiff, "nansafe_mode" => true)

# need to import fit to overload and reexport it
import StatsBase.fit

# initialize R session for splines
function __init__()
    @eval RCall.R"if (!require('splines2', quietly=TRUE)) install.packages('splines2')"
    @eval RCall.@rlibrary splines2
    nothing
end

# initialize runtime generated function cache
RuntimeGeneratedFunctions.init(@__MODULE__)

# Write your package code here.
export
    @formula,
    compute_hazard,
    compute_cumulative_hazard,
    collapse_data,
    cumulative_incidence,
    fit,
    fit_surrogate,
    get_ConvergenceRecords,
    get_loglik,
    get_parameters,
    get_parnames,
    get_vcov,
    Hazard,
    make_constraints,
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
include("initialization.jl")

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

# surrogate
include("surrogates.jl")

end
