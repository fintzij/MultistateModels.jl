
using Test
using MultistateModels
using DataFrames
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics

# Include configuration and helpers
include("../longtest_config.jl")
include("../longtest_helpers.jl")
include("report_generator.jl")

# Include all test files
include("exponential_tests.jl")
include("weibull_tests.jl")
include("gompertz_tests.jl")
include("phasetype_proposal_tests.jl")  # Phase-type proposals for semi-Markov MCEM
include("phasetype_hazard_tests.jl")    # Phase-type hazard family (:pt) inference tests
include("phasetype_simulation_tests.jl") # Phase-type simulation validation tests
include("spline_tests.jl")

function run_all_longtests()
    # Clear previous results
    empty!(ALL_RESULTS)
    
    @testset "All Long Tests" begin
        run_all_gompertz_tests()
        run_all_weibull_tests()
        run_all_spline_tests()
        run_all_exponential_tests()
        run_all_phasetype_hazard_tests()    # Phase-type hazard family (:pt)
        run_all_phasetype_simulation_tests() # Phase-type simulation validation
        run_all_phasetype_proposal_tests()  # Phase-type proposals for semi-Markov
    end
    
    # Generate report
    generate_longtest_report(ALL_RESULTS)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_longtests()
end
