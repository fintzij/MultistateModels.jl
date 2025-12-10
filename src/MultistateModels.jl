module MultistateModels

using BSplineKit
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
using OptimizationOptimJL
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

# =============================================================================
# Exports - User-facing API only
# =============================================================================
# Internal functions and types are not exported but remain accessible via
# MultistateModels.internal_function() for advanced users and testing.

export
    # Re-exports from StatsModels
    @formula,
    
    # --------------------------------------------------------------------------
    # Core API: Model specification and fitting
    # --------------------------------------------------------------------------
    Hazard,
    @hazard,
    multistatemodel,
    fit,
    
    # --------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------
    simulate,
    simulate_data,
    simulate_paths,
    simulate_path,
    
    # --------------------------------------------------------------------------
    # Model information criteria
    # --------------------------------------------------------------------------
    aic, 
    bic,
    
    # --------------------------------------------------------------------------
    # Model accessors (unified API)
    # --------------------------------------------------------------------------
    # Parameters: get_parameters(model; scale=:natural/:flat/:nested)
    # VCov: get_vcov(model; type=:model/:ij/:jk)
    # Pseudovalues: get_pseudovalues(model; type=:jk/:ij)
    get_loglik,
    get_parameters,
    get_parnames,
    get_vcov,
    get_pseudovalues,
    get_convergence_records,
    get_expanded_parameters,
    
    # --------------------------------------------------------------------------
    # Variance estimation diagnostics
    # --------------------------------------------------------------------------
    get_subject_gradients,
    get_loo_perturbations,
    get_influence_functions,
    compare_variance_estimates,
    
    # --------------------------------------------------------------------------
    # Parameter manipulation
    # --------------------------------------------------------------------------
    set_parameters!,
    set_surrogate!,
    is_surrogate_fitted,
    initialize_parameters,
    initialize_parameters!,
    
    # --------------------------------------------------------------------------
    # Phase-type model accessors
    # --------------------------------------------------------------------------
    is_phasetype_fitted,
    get_phasetype_parameters,
    get_mappings,
    get_original_data,
    get_original_tmat,
    get_convergence,
    
    # --------------------------------------------------------------------------
    # Simulation strategy types
    # --------------------------------------------------------------------------
    OptimJumpSolver,
    CachedTransformStrategy,
    DirectTransformStrategy,
    
    # --------------------------------------------------------------------------
    # AD Backend selection
    # --------------------------------------------------------------------------
    ADBackend,
    ForwardDiffBackend,
    EnzymeBackend,
    MooncakeBackend,
    
    # --------------------------------------------------------------------------
    # MCEM proposal configuration
    # --------------------------------------------------------------------------
    ProposalConfig,
    MarkovProposal,
    PhaseTypeProposal,
    
    # --------------------------------------------------------------------------
    # Phase-type types (user may receive/inspect)
    # --------------------------------------------------------------------------
    PhaseTypeDistribution,
    PhaseTypeModel,
    
    # --------------------------------------------------------------------------
    # Threading utilities
    # --------------------------------------------------------------------------
    get_physical_cores,
    recommended_nthreads,
    
    # --------------------------------------------------------------------------
    # User utilities
    # --------------------------------------------------------------------------
    calibrate_splines,
    calibrate_splines!,
    compute_hazard,
    compute_cumulative_hazard,
    cumulative_incidence,
    draw_paths,
    estimate_loglik,
    make_constraints,
    path_to_dataframe,
    paths_to_dataset,
    summary

# typedefs
include("common.jl")

# helpers
include("helpers.jl")

# shared stats utilities
include("statsutils.jl")

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

# phase-type distributions for improved surrogates
include("phasetype.jl")

# surrogate
include("surrogates.jl")

# model fitting
include("modelfitting.jl")

# model generation
include("modelgeneration.jl")

# declarative macros
include("macros.jl")

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

# cross-validation and robust covariance estimation
include("crossvalidation.jl")

end
