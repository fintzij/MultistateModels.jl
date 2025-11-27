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
# include("setup_splines.jl")  # TODO: Splines not yet implemented in infrastructure_changes

@testset "runtests" begin
    include("test_modelgeneration.jl")
    include("test_hazards.jl")
    include("test_helpers.jl")
    include("test_make_subjdat.jl")
    include("test_simulation.jl")
end
