module MultistateModels

using ArraysOfArrays
using BSplineKit
using Chain
using ComponentArrays
using DataFrames
using DiffResults
using Distributions
using ElasticArrays
using ExponentialUtilities
using ForwardDiff
using Ipopt
using LinearAlgebra
using MacroTools
using Optim # for simulation - keep
using Optimization # for fitting - keep
using OptimizationMOI
using OrderedCollections
using ParameterHandling
using ParetoSmooth
using Preferences
using QuadGK
using RuntimeGeneratedFunctions
using StatsBase
using StatsFuns
using StatsModels

# make sure ForwardDiff is nan safe
Preferences.set_preferences!(ForwardDiff, "nansafe_mode" => true)

# need to import fit to overload and reexport it
import StatsBase.fit

# initialize runtime generated function cache
RuntimeGeneratedFunctions.init(@__MODULE__)

# Write your package code here.
export
    @formula,
    aic, 
    bic,
    compute_hazard,
    compute_cumulative_hazard,
    collapse_data,
    cumulative_incidence,
    draw_paths,
    estimate_loglik,
    fit,
    fit_surrogate,
    get_convergence_records,
    get_loglik,
    get_parameters,
    get_parameters_flat,
    get_parameters_transformed,
    get_parameters_natural,
    get_unflatten_fn,
    get_parnames,
    get_vcov,
    Hazard,
    initialize_parameters,
    initialize_parameters!,
    initialize_surrogate!,
    make_constraints,
    multistatemodel,
    set_parameters,
    set_parameters!,
    simulate,
    # statetable,
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
