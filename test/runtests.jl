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
# Test execution is controlled by MSM_TEST_LEVEL environment variable:
#   "quick" (default) - Unit tests only (~1 minute)
#   "full"            - Unit tests + long statistical validation tests (~15-20 min)
#
# Example: MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'

const TEST_LEVEL = get(ENV, "MSM_TEST_LEVEL", "quick")

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
    include("test_exact_data_fitting.jl")
    include("test_phasetype_is.jl")
    include("test_splines.jl")
end

# =============================================================================
# Long Tests - Statistical Validation (Run with MSM_TEST_LEVEL=full)
# =============================================================================
if TEST_LEVEL == "full"
    @info "Running full test suite including long statistical tests..."
    
    @testset "Long Tests - Exact Markov" begin
        include("longtest_exact_markov.jl")
    end
    
    @testset "Long Tests - MCEM" begin
        include("longtest_mcem.jl")
    end
    
    @testset "Long Tests - MCEM Splines" begin
        include("longtest_mcem_splines.jl")
    end
    
    @testset "Long Tests - Simulation Distribution" begin
        include("longtest_simulation_distribution.jl")
    end
    
    @testset "Long Tests - Simulation TVC" begin
        include("longtest_simulation_tvc.jl")
    end
else
    @info "Running quick tests only. Set MSM_TEST_LEVEL=full for complete suite."
end
