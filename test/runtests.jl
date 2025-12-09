using DataFrames
using LinearAlgebra
using MultistateModels
using Random
using Test

# =============================================================================
# Test Harness Entry Point
# =============================================================================
#
# Centralizes setup code, fixture includes, and the list of executable suites.
# Keep this file lightweight—add heavy test logic in dedicated `test_*.jl`
# files and simply `include` them here so `Pkg.test()` automatically runs
# everything. When adding a new suite, document it in test/testcoverage.md.
#
# Test execution is controlled by environment variables:
#
# MSM_TEST_LEVEL:
#   "quick" (default) - Unit tests only (~1 minute)
#   "full"            - Unit tests + long statistical validation tests (~15-20 min)
#
# Individual longtest toggles (only apply when MSM_TEST_LEVEL=full):
#   MSM_LONGTEST_EXACT_DATA=true/false       - Exact data MLE (Exp/Wei/Gom/Spline/TVC)
#   MSM_LONGTEST_MCEM_PARAMETRIC=true/false  - Panel data MCEM (Exp/Wei/Gom)
#   MSM_LONGTEST_MCEM_SPLINES=true/false     - Panel data MCEM with spline hazards
#   MSM_LONGTEST_MCEM_TVC=true/false         - Panel data MCEM with time-varying covariates
#   MSM_LONGTEST_SIM_DIST=true/false         - Simulation distributional fidelity
#   MSM_LONGTEST_SIM_TVC=true/false          - Simulation with time-varying covariates
#   MSM_LONGTEST_ROBUST_EXACT=true/false     - Large-n exact data (tight tolerances)
#   MSM_LONGTEST_MARKOV_PHASETYPE_VALIDATION=true/false - Markov/PhaseType validation
#   MSM_LONGTEST_PHASETYPE_EXACT=true/false  - Phase-type hazard models (exact data)
#   MSM_LONGTEST_PHASETYPE_PANEL=true/false  - Phase-type hazard models (panel data)
#
# Example: Run only the exact data longtest:
#   MSM_TEST_LEVEL=full MSM_LONGTEST_ONLY=exact_data julia -e 'using Pkg; Pkg.test()'
#
# MSM_LONGTEST_ONLY options: exact_data, mcem_parametric, mcem_splines, mcem_tvc,
#                            sim_dist, sim_tvc, robust_exact, markov_phasetype_validation,
#                            phasetype_exact, phasetype_panel

const TEST_LEVEL = get(ENV, "MSM_TEST_LEVEL", "quick")

# Helper to check if a specific longtest should run
function should_run_longtest(name::String)
    # If MSM_LONGTEST_ONLY is set, only run that specific test
    only_test = get(ENV, "MSM_LONGTEST_ONLY", "")
    if !isempty(only_test)
        return lowercase(only_test) == lowercase(name)
    end
    
    # Otherwise check individual toggle (default: true)
    env_key = "MSM_LONGTEST_" * uppercase(replace(name, "_" => "_"))
    return lowercase(get(ENV, env_key, "true")) == "true"
end

# Deterministic RNG so regression failures are reproducible in CI.
Random.seed!(52787)

# setup file for generating model objects 
# include("setup.jl") # this could be included in the individual .jl files
include("setup_3state_expwei.jl")
include("setup_3state_weiph.jl")
include("setup_2state_trans.jl")
include("setup_gompertz.jl")
include("fixtures/TestFixtures.jl")
using .TestFixtures
include("setup_splines.jl")  # Spline hazards now implemented

# =============================================================================
# Quick Tests (Unit Tests) - Always Run
# =============================================================================
@testset "Unit Tests" begin
    include("test_modelgeneration.jl")
    include("test_hazards.jl")
    include("test_helpers.jl")
    include("test_make_subjdat.jl")
    include("test_simulation.jl")
    include("test_ncv.jl")
    include("test_phasetype_is.jl")
    include("test_phasetype_correctness.jl")
    include("test_phasetype_fitting.jl")
    include("test_phasetype_simulation.jl")
    include("test_splines.jl")
    include("test_surrogates.jl")
    include("test_mcem.jl")
    include("test_reversible_tvc_loglik.jl")
    include("test_parallel_likelihood.jl")
    include("test_parameter_ordering.jl")
end

# =============================================================================
# Long Tests - Statistical Validation (Run with MSM_TEST_LEVEL=full)
# =============================================================================
if TEST_LEVEL == "full"
    @info "Running full test suite including long statistical tests..."
    
    # Check if any specific test is requested
    only_test = get(ENV, "MSM_LONGTEST_ONLY", "")
    if !isempty(only_test)
        @info "Running only: $only_test"
    end
    
    if should_run_longtest("exact_data")
        @testset "Long Tests - Exact Data" begin
            include("longtest_exact_markov.jl")
        end
    end
    
    if should_run_longtest("mcem_parametric")
        @testset "Long Tests - Panel Data MCEM (Parametric Hazards)" begin
            include("longtest_mcem.jl")
        end
    end
    
    if should_run_longtest("mcem_splines")
        @testset "Long Tests - Panel Data MCEM (Spline Hazards)" begin
            include("longtest_mcem_splines.jl")
        end
    end
    
    if should_run_longtest("mcem_tvc")
        @testset "Long Tests - Panel Data MCEM (Time-Varying Covariates)" begin
            include("longtest_mcem_tvc.jl")
        end
    end
    
    if should_run_longtest("sim_dist")
        @testset "Long Tests - Simulation Distributional Fidelity" begin
            include("longtest_simulation_distribution.jl")
        end
    end
    
    if should_run_longtest("sim_tvc")
        @testset "Long Tests - Simulation (Time-Varying Covariates)" begin
            include("longtest_simulation_tvc.jl")
        end
    end
    
    if should_run_longtest("robust_exact")
        @testset "Long Tests - Robust Exact Data (Tight Tolerances)" begin
            include("longtest_robust_parametric.jl")
        end
    end
    
    if should_run_longtest("markov_phasetype_validation")
        @testset "Long Tests - Markov/PhaseType Validation" begin
            include("longtest_robust_markov_phasetype.jl")
        end
    end
    
    if should_run_longtest("phasetype_exact")
        @testset "Long Tests - Phase-Type Hazard Models (Exact Data)" begin
            include("longtest_phasetype_exact.jl")
        end
    end
    
    if should_run_longtest("phasetype_panel")
        @testset "Long Tests - Phase-Type Hazard Models (Panel Data)" begin
            include("longtest_phasetype_panel.jl")
        end
    end
else
    @info "Running quick tests only. Set MSM_TEST_LEVEL=full for complete suite."
end
