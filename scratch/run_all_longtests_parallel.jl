# Run all long tests in parallel processes
# Each test runs in its own Julia process to avoid interference

using Dates

const TESTS_DIR = joinpath(@__DIR__, "..", "MultistateModelsTests", "longtests")
const LOG_DIR = joinpath(@__DIR__, "..", "MultistateModelsTests", "cache", "longtest_logs")
const PROJECT_DIR = joinpath(@__DIR__, "..")

mkpath(LOG_DIR)

# List of long test files to run
const LONG_TESTS = [
    "longtest_parametric_suite.jl",
    "longtest_exact_markov.jl", 
    "longtest_mcem.jl",
    "longtest_mcem_splines.jl",
    "longtest_mcem_tvc.jl",
    "longtest_simulation_distribution.jl",
    "longtest_simulation_tvc.jl",
    "longtest_robust_parametric.jl",
    "longtest_robust_markov_phasetype.jl",
    "longtest_phasetype.jl",
    "longtest_variance_validation.jl",
]

function run_test(testfile::String)
    testname = replace(testfile, ".jl" => "")
    logfile = joinpath(LOG_DIR, "$(testname)_$(Dates.format(now(), "HHMMss")).log")
    
    # Build the Julia command - run test file directly with proper setup
    cmd = ```julia --project=$PROJECT_DIR -e """
        using Pkg
        Pkg.instantiate()
        using MultistateModels
        using Test
        using Random
        Random.seed!(52787)
        
        # Load config and helpers
        include(joinpath(\"$TESTS_DIR\", \"longtest_config.jl\"))
        include(joinpath(\"$TESTS_DIR\", \"longtest_helpers.jl\"))
        
        # Run the specific test
        @testset \"$testname\" begin
            include(joinpath(\"$TESTS_DIR\", \"$testfile\"))
        end
    """```
    
    return (testname, cmd, logfile)
end

function main()
    println("="^70)
    println("Starting parallel long test execution")
    println("Time: $(now())")
    println("="^70)
    println()
    
    # Create tasks for all tests
    tasks = Dict{String, Task}()
    
    for testfile in LONG_TESTS
        testname, cmd, logfile = run_test(testfile)
        println("Starting: $testname -> $logfile")
        
        # Run in background
        tasks[testname] = @async begin
            open(logfile, "w") do io
                try
                    run(pipeline(cmd; stdout=io, stderr=io))
                    return (testname, :passed, logfile)
                catch e
                    return (testname, :failed, logfile)
                end
            end
        end
    end
    
    println()
    println("All tests launched. Waiting for completion...")
    println()
    
    # Wait and collect results
    results = Dict{String, Symbol}()
    for (testname, task) in tasks
        result = fetch(task)
        results[result[1]] = result[2]
        status = result[2] == :passed ? "✓" : "✗"
        println("$status $testname completed (see $(result[3]))")
    end
    
    println()
    println("="^70)
    println("Summary")
    println("="^70)
    passed = count(v -> v == :passed, values(results))
    failed = count(v -> v == :failed, values(results))
    println("Passed: $passed")
    println("Failed: $failed")
    
    if failed > 0
        println("\nFailed tests:")
        for (name, status) in results
            if status == :failed
                println("  - $name")
            end
        end
    end
    
    return failed == 0
end

main()
