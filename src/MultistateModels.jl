module MultistateModels

using BSplineKit
using ComponentArrays
using DataFrames
using DifferentiationInterface
using DiffResults
using Distributions
using ElasticArrays
using ExponentialUtilities
using ForwardDiff
using LinearAlgebra
using MacroTools
using Optim # for simulation - keep
using Optimization # for fitting - keep
using OptimizationIpopt
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
    s,
    te,
    
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
    is_fitted,
    initialize_parameters,
    initialize_parameters!,
    
    # --------------------------------------------------------------------------
    # Type hierarchy
    # --------------------------------------------------------------------------
    AbstractSurrogate,
    MarkovSurrogate,
    PhaseTypeSurrogate,
    
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
    ExponentialJumpSolver,
    HybridJumpSolver,
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
    # Spline penalty configuration
    # --------------------------------------------------------------------------
    SplinePenalty,
    PenaltyConfig,
    has_penalties,
    compute_penalty,
    build_penalty_config,
    select_smoothing_parameters,
    
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
    PhaseTypeExpansion,
    has_phasetype_expansion,
    
    # --------------------------------------------------------------------------
    # Model classification traits
    # --------------------------------------------------------------------------
    is_markov,
    is_panel_data,
    
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
    build_penalty_matrix,
    build_spline_hazard_info,
    place_interior_knots_pooled,
    validate_shared_knots,
    compute_hazard,
    compute_cumulative_hazard,
    cumulative_incidence,
    draw_paths,
    estimate_loglik,
    generate_parameter_bounds,
    make_constraints,
    path_to_dataframe,
    paths_to_dataset,
    summary,
    
    # --------------------------------------------------------------------------
    # Transition enumeration (for per-transition obstype in simulation)
    # --------------------------------------------------------------------------
    enumerate_transitions,
    transition_index_map,
    print_transition_map

# =============================================================================
# Type Definitions (from types/ subfolder)
# =============================================================================
# Order matters: abstract types first, then concrete types that depend on them

# Abstract type hierarchy
include("types/abstract.jl")

# Package-wide constants (numerical tolerances, thresholds)
# Loaded early so other modules can use them
include("utilities/constants.jl")

# Hazard metadata and caching types
include("types/hazard_metadata.jl")

# Hazard struct definitions (internal runtime types)
include("types/hazard_structs.jl")

# User-facing hazard specification types
include("types/hazard_specs.jl")

# Model struct definitions (MultistateModel, MultistateModelFitted, etc.)
include("types/model_structs.jl")

# Data container types (SamplePath, ExactData, MPanelData, etc.)
include("types/data_containers.jl")

# Infrastructure types (AD backends, threading config)
include("types/infrastructure.jl")

# =============================================================================
# Utilities (from utilities/ subfolder)
# =============================================================================
# Order matters: type definitions first, then functions that depend on them

# Parameter flattening type system and construction functions
include("utilities/flatten.jl")

# ReConstructor struct and flatten/unflatten API
include("utilities/reconstructor.jl")

# shared stats utilities
include("utilities/stats.jl")

# Parameter scale transformations (estimation <-> natural)
include("utilities/transforms.jl")

# Parameter manipulation functions (unflatten, set/get, build)
include("utilities/parameters.jl")

# Data and parameter validation functions
include("utilities/validation.jl")

# Transition enumeration and per-transition obstype validation
include("utilities/transition_helpers.jl")

# TPM bookkeeping, data containers, and data manipulation
include("utilities/books.jl")

# =============================================================================
# Hazard Functions (from hazard/ subfolder)
# =============================================================================
# Order matters: covariates first, then transforms, generators, evaluation

# Covariate extraction and linear predictor
include("hazard/covariates.jl")

# Time transform optimizations
include("hazard/time_transform.jl")

# Hazard generator functions (runtime code generation)
include("hazard/generators.jl")

# Callable hazard interface and eval_hazard/eval_cumhaz API
include("hazard/evaluation.jl")

# Total hazard and survival probability functions
include("hazard/total_hazard.jl")

# Transition probability matrix functions
include("hazard/tpm.jl")

# User-facing API (cumulative_incidence, compute_hazard, etc.)
include("hazard/api.jl")

# crude parameter initialization functions
include("utilities/initialization.jl")

# likelihood functions (split for maintainability)
include("likelihood/loglik_utils.jl")       # ForwardDiff helpers, parameter prep
include("likelihood/loglik_batched.jl")     # Batched hazard-centric infrastructure
include("likelihood/loglik_markov.jl")      # Markov panel likelihood, forward algorithm
include("likelihood/loglik_markov_functional.jl")  # Reverse-mode AD compatible
include("likelihood/loglik_semi_markov.jl") # Semi-Markov MCEM path-based likelihood
include("likelihood/loglik_exact.jl")       # Exact data likelihood, fused computation

# Monte Carlo EM functions
include("inference/mcem.jl")

# miscellaneous functions
include("utilities/misc.jl")

# parameter bounds for box-constrained optimization
include("utilities/bounds.jl")

# phase-type distributions for improved surrogates (split for maintainability)
include("phasetype/types.jl")
include("phasetype/surrogate.jl")
include("phasetype/expansion_mappings.jl")    # State space mappings
include("phasetype/expansion_hazards.jl")     # Hazard expansion
include("phasetype/expansion_constraints.jl") # SCTP constraint generation
include("phasetype/expansion_model.jl")       # Model building
include("phasetype/expansion_loglik.jl")      # Log-likelihood computation
include("phasetype/expansion_ffbs_data.jl")   # Data expansion for FFBS

# surrogate
include("surrogate/markov.jl")

# sampling importance resampling
include("inference/sir.jl")

# model fitting (split for maintainability)
include("inference/fit_common.jl")   # Entry point, dispatch, common helpers
include("inference/fit_exact.jl")    # Exact data fitting
include("inference/fit_markov.jl")   # Markov panel fitting
include("inference/fit_mcem.jl")     # Semi-Markov MCEM fitting

# smooths / spline hazards (must be before multistatemodel.jl)
include("utilities/spline_utils.jl")
include("hazard/smooth_terms.jl")

# model generation
include("construction/multistatemodel.jl")

# declarative macros
include("hazard/macros.jl")

# model output / accessors
include("output/accessors.jl")

# path functions
include("simulation/path_utilities.jl")

# sampling functions
include("inference/sampling.jl")

# simulation
include("simulation/simulate.jl")

# spline baseline hazards (requires smooth_terms.jl already loaded)
include("hazard/spline.jl")

# penalty configuration builder (must be after spline.jl)
include("utilities/penalty_config.jl")

# cross-validation and robust covariance estimation
include("output/variance.jl")

# smoothing parameter selection (PIJCV, EFS, PERF, CV) - must be after variance.jl
include("inference/smoothing_selection.jl")

end
