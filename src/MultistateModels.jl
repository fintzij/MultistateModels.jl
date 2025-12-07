module MultistateModels

using ADTypes
using BSplineKit
using Chain
using ComponentArrays
using DataFrames
using DiffResults
using Distributions
using ElasticArrays
using Enzyme
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
using Mooncake

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
    truncate_distribution,
    draw_paths,
    estimate_loglik,
    enable_time_transform_cache!,
    maybe_time_transform_context,
    fit,
    fit_surrogate,
    set_surrogate!,
    get_convergence_records,
    get_loglik,
    get_parameters,
    get_expanded_parameters,
    get_parameters_flat,
    get_parameters_nested,
    get_parameters_transformed,  # Backward compat alias for get_parameters_nested
    get_parameters_natural,
    get_log_scale_params,
    get_unflatten_fn,
    get_parnames,
    get_vcov,
    get_ij_vcov,
    get_jk_vcov,
    get_subject_gradients,
    get_loo_perturbations,
    get_jk_pseudovalues,
    get_ij_pseudovalues,
    get_influence_functions,
    compare_variance_estimates,
    ij_vcov,
    jk_vcov,
    loo_perturbations_direct,
    loo_perturbations_cholesky,
    compute_robust_vcov,
    # NCV exports
    NCVState,
    cholesky_downdate!,
    cholesky_downdate_copy,
    ncv_loo_perturbation_cholesky,
    ncv_loo_perturbation_woodbury,
    ncv_loo_perturbation_direct,
    compute_ncv_perturbations!,
    ncv_criterion,
    ncv_criterion_quadratic,
    ncv_criterion_derivatives,
    ncv_degeneracy_test,
    ncv_get_loo_estimates,
    ncv_get_perturbations,
    ncv_vcov,
    Hazard,
    @hazard,
    initialize_parameters,
    initialize_parameters!,
    initialize_surrogate!,
    # Threading configuration
    get_physical_cores,
    recommended_nthreads,
    ThreadingConfig,
    set_threading_config!,
    get_threading_config,
    should_parallelize,
    # Batched likelihood infrastructure (for advanced users / semi-Markov batched)
    is_separable,
    cache_path_data,
    CachedPathData,
    BatchedODEData,
    StackedHazardData,
    LightweightInterval,
    SubjectCovarCache,
    # Likelihood functions
    loglik,
    loglik_exact,
    loglik_markov,
    loglik_markov_functional,
    loglik_semi_markov,
    make_constraints,
    multistatemodel,
    set_parameters,
    set_parameters!,
    simulate,
    simulate_data,
    simulate_paths,
    simulate_path,
    path_to_dataframe,
    paths_to_dataset,
    OptimJumpSolver,
    CachedTransformStrategy,
    DirectTransformStrategy,
    # Legacy aliases (deprecated, use CachedTransformStrategy/DirectTransformStrategy)
    TangTransformStrategy,
    LegacyTransformStrategy,
    # AD Backend selection
    ADBackend,
    ForwardDiffBackend,
    EnzymeBackend,
    MooncakeBackend,
    default_ad_backend,
    # Phase-type surrogates
    PhaseTypeDistribution,
    PhaseTypeConfig,
    PhaseTypeSurrogate,
    absorption_rates,
    subintensity,
    # MCEM proposal configuration
    ProposalConfig,
    MarkovProposal,
    PhaseTypeProposal,
    needs_phasetype_proposal,
    resolve_proposal_config,
    collapse_phases,
    expand_initial_state,
    # Phase-type hazard models (:pt family)
    PhaseTypeHazardSpec,
    PhaseTypeCoxianHazard,
    PhaseTypeMappings,
    PhaseTypeModel,
    # Phase-type fitted model accessors
    is_phasetype_fitted,
    get_phasetype_parameters,
    get_mappings,
    get_original_data,
    get_original_tmat,
    get_convergence,
    get_expanded_parameters,
    # Spline utilities
    place_knots_from_paths!,
    default_nknots,
    place_interior_knots,
    # statetable,
    summary,
    __init__

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
