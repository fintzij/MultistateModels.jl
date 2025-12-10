# =============================================================================
# Long Test Configuration
# =============================================================================
# 
# Shared configuration for all inference long tests.
# All tests use a 3-state progressive model: State 1 → State 2 → State 3
# =============================================================================

# Sample size (same for all tests)
const N_SUBJECTS = 1000

# Time settings
const MAX_TIME = 15.0
const PANEL_TIMES = collect(1.0:1.0:10.0)  # Observations at t = 1, 2, ..., 10
const TVC_CHANGEPOINT = 5.0                 # Time-varying covariate changes at t=5
const EVAL_TIMES = collect(0.0:0.5:MAX_TIME)

# Simulation settings
const N_SIM_TRAJ = 5000  # Number of trajectories for prevalence/cumincid plots
const RNG_SEED = 2882347045

# Pass criteria
const PASS_THRESHOLD = 10.0  # Max |relative error| ≤ 10%

# MCEM settings (Markov proposals for all)
const MCEM_TOL = 0.01
const MCEM_ESS_INITIAL = 100
const MCEM_ESS_MAX = 2000
const MCEM_MAX_ITER = 50

# Spline settings
const SPLINE_DEGREE = 1  # Linear between knots (degree 1)

# Output settings
const OUTPUT_DIR = joinpath(@__DIR__, "reports")
const ASSETS_DIR = joinpath(OUTPUT_DIR, "assets", "diagnostics")

# =============================================================================
# TestResult Structure
# =============================================================================

"""
    TestResult

Container for a single test's results including parameters, estimates, 
errors, and diagnostic data for plotting.

Uses @kwdef for keyword-argument construction.
"""
Base.@kwdef mutable struct TestResult
    # Test identification
    name::String
    family::String
    parameterization::Symbol = :none     # :none, :ph, :aft
    covariates::Symbol = :none           # :none, :tfc, :tvc
    data_type::Symbol = :exact           # :exact, :panel
    n_subjects::Int = N_SUBJECTS
    
    # Parameters
    true_params::Dict{String, Float64} = Dict{String, Float64}()
    estimated_params::Dict{String, Float64} = Dict{String, Float64}()
    rel_errors::Dict{String, Float64} = Dict{String, Float64}()
    
    # Pass/fail
    max_rel_error::Float64 = NaN
    passed::Bool = false
    
    # Diagnostic data
    eval_times::Vector{Float64} = Float64[]
    prevalence_true::Union{Nothing, Matrix{Float64}} = nothing
    prevalence_observed::Union{Nothing, Matrix{Float64}} = nothing
    prevalence_fitted::Union{Nothing, Matrix{Float64}} = nothing
    cumincid_12_true::Union{Nothing, Vector{Float64}} = nothing
    cumincid_12_observed::Union{Nothing, Vector{Float64}} = nothing
    cumincid_12_fitted::Union{Nothing, Vector{Float64}} = nothing
    cumincid_23_true::Union{Nothing, Vector{Float64}} = nothing
    cumincid_23_observed::Union{Nothing, Vector{Float64}} = nothing
    cumincid_23_fitted::Union{Nothing, Vector{Float64}} = nothing
end

# Global results storage
const ALL_RESULTS = TestResult[]

# =============================================================================
# Test Categories for Reporting
# =============================================================================

const FAMILY_NAMES = Dict(
    "exp" => "Exponential",
    "wei" => "Weibull", 
    "gom" => "Gompertz",
    "phasetype" => "Phase-Type",
    "spline" => "Spline"
)

const PARAM_NAMES = Dict(
    "none" => "Baseline",
    "ph" => "PH",
    "aft" => "AFT"
)

const COV_NAMES = Dict(
    "none" => "No Covariates",
    "tfc" => "Time-Fixed",
    "tvc" => "Time-Varying"
)

const DATA_NAMES = Dict(
    "exact" => "Exact",
    "panel" => "Panel"
)
