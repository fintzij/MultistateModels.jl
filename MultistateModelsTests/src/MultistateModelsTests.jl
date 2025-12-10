module MultistateModelsTests

using DataFrames
using LinearAlgebra
using MultistateModels
using Random
using Test

const TEST_LEVEL = get(ENV, "MSM_TEST_LEVEL", "quick")

function should_run_longtest(name::String)
    only_test = get(ENV, "MSM_LONGTEST_ONLY", "")
    if !isempty(only_test)
        return lowercase(only_test) == lowercase(name)
    end
    env_key = "MSM_LONGTEST_" * uppercase(replace(name, "_" => "_"))
    return lowercase(get(ENV, env_key, "true")) == "true"
end

# Bring in fixtures (now lives within the test package)
include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
using .TestFixtures

# Paths into the new package layout
const UNIT_DIR = joinpath(@__DIR__, "..", "unit")
const INTEGRATION_DIR = joinpath(@__DIR__, "..", "integration")
const LONGTESTS_DIR = joinpath(@__DIR__, "..", "longtests")

function runtests()
    # Deterministic RNG so regression failures are reproducible in CI.
    Random.seed!(52787)

    @testset "Unit Tests" begin
        include(joinpath(UNIT_DIR, "test_modelgeneration.jl"))
        include(joinpath(UNIT_DIR, "test_hazards.jl"))
        include(joinpath(UNIT_DIR, "test_helpers.jl"))
        include(joinpath(UNIT_DIR, "test_simulation.jl"))
        include(joinpath(UNIT_DIR, "test_ncv.jl"))
        include(joinpath(UNIT_DIR, "test_phasetype.jl"))
        include(joinpath(UNIT_DIR, "test_splines.jl"))
        include(joinpath(UNIT_DIR, "test_surrogates.jl"))
        include(joinpath(UNIT_DIR, "test_mcem.jl"))
        include(joinpath(UNIT_DIR, "test_reconstructor.jl"))
        include(joinpath(UNIT_DIR, "test_reversible_tvc_loglik.jl"))
        include(joinpath(UNIT_DIR, "test_initialization.jl"))
        include(joinpath(UNIT_DIR, "test_variance.jl"))
        include(joinpath(INTEGRATION_DIR, "test_parallel_likelihood.jl"))
        include(joinpath(INTEGRATION_DIR, "test_parameter_ordering.jl"))
    end

    if TEST_LEVEL == "full"
        @info "Running full test suite including long statistical tests..."

        only_test = get(ENV, "MSM_LONGTEST_ONLY", "")
        if !isempty(only_test)
            @info "Running only: $only_test"
        end

        if should_run_longtest("exact_data")
            @testset "Long Tests - Exact Data" begin
                include(joinpath(LONGTESTS_DIR, "longtest_exact_markov.jl"))
            end
        end

        if should_run_longtest("mcem_parametric")
            @testset "Long Tests - Panel Data MCEM (Parametric Hazards)" begin
                include(joinpath(LONGTESTS_DIR, "longtest_mcem.jl"))
            end
        end

        if should_run_longtest("mcem_splines")
            @testset "Long Tests - Panel Data MCEM (Spline Hazards)" begin
                include(joinpath(LONGTESTS_DIR, "longtest_mcem_splines.jl"))
            end
        end

        if should_run_longtest("mcem_tvc")
            @testset "Long Tests - Panel Data MCEM (Time-Varying Covariates)" begin
                include(joinpath(LONGTESTS_DIR, "longtest_mcem_tvc.jl"))
            end
        end

        if should_run_longtest("sim_dist")
            @testset "Long Tests - Simulation Distributional Fidelity" begin
                include(joinpath(LONGTESTS_DIR, "longtest_simulation_distribution.jl"))
            end
        end

        if should_run_longtest("sim_tvc")
            @testset "Long Tests - Simulation (Time-Varying Covariates)" begin
                include(joinpath(LONGTESTS_DIR, "longtest_simulation_tvc.jl"))
            end
        end

        if should_run_longtest("robust_exact")
            @testset "Long Tests - Robust Exact Data (Tight Tolerances)" begin
                include(joinpath(LONGTESTS_DIR, "longtest_robust_parametric.jl"))
            end
        end

        if should_run_longtest("markov_phasetype_validation")
            @testset "Long Tests - Markov/PhaseType Validation" begin
                include(joinpath(LONGTESTS_DIR, "longtest_robust_markov_phasetype.jl"))
            end
        end

        if should_run_longtest("phasetype")
            @testset "Long Tests - Phase-Type Hazard Models (Exact + Panel Data)" begin
                include(joinpath(LONGTESTS_DIR, "longtest_phasetype.jl"))
            end
        end

        if should_run_longtest("variance_validation")
            @testset "Long Tests - Variance Estimation Validation" begin
                include(joinpath(LONGTESTS_DIR, "longtest_variance_validation.jl"))
            end
        end
    else
        @info "Running quick tests only. Set MSM_TEST_LEVEL=full for complete suite."
    end

    return nothing
end

# Diagnostics directory path
const DIAGNOSTICS_DIR = joinpath(@__DIR__, "..", "diagnostics")

"""
    generate_simulation_diagnostics()

Regenerate all simulation diagnostic plots (hazard/cumulative hazard/survival curves
and simulation distribution validation). Outputs PNG files to 
`MultistateModelsTests/diagnostics/assets/`.

Requires CairoMakie (loaded on-demand).
"""
function generate_simulation_diagnostics()
    script = joinpath(DIAGNOSTICS_DIR, "generate_model_diagnostics.jl")
    @info "Running simulation diagnostics generator..."
    include(script)
    Main.generate_all()
    @info "Diagnostics saved to $(joinpath(DIAGNOSTICS_DIR, "assets"))"
end

"""
    diagnostics_path()

Return the path to the diagnostics directory.
"""
diagnostics_path() = DIAGNOSTICS_DIR

"""
    diagnostics_assets_path()

Return the path to the diagnostics assets directory containing generated plots.
"""
diagnostics_assets_path() = joinpath(DIAGNOSTICS_DIR, "assets")

export runtests, generate_simulation_diagnostics, diagnostics_path, diagnostics_assets_path

end # module
