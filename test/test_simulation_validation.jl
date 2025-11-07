# Test: Simulation Validation
# Verify that manually simulated trajectories match package simulation

using MultistateModels
using DataFrames
using Distributions
using Random
using Test

println("=" ^ 80)
println("SIMULATION VALIDATION TESTS")
println("=" ^ 80)

Random.seed!(12345)

@testset "Simulation Validation" begin
    
    # =========================================================================
    # Test 1: Simple 2-state exponential model (intercept-only)
    # =========================================================================
    @testset "2-state exponential (intercept-only)" begin
        println("\n--- Test 1: 2-state exponential (intercept-only) ---")
        
        # True parameter
        true_lambda = 0.2  # hazard rate
        true_log_lambda = log(true_lambda)
        
        # Create model
        haz = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # Initial data (just for structure)
        n_subjects = 1000
        init_data = DataFrame(
            id = 1:n_subjects,
            tstart = zeros(n_subjects),
            tstop = fill(10.0, n_subjects),
            statefrom = ones(Int, n_subjects),
            stateto = ones(Int, n_subjects),
            obstype = ones(Int, n_subjects)
        )
        
        model = multistatemodel(haz; data = init_data)
        
        # Set parameters (needs vector of vectors, one per hazard)
        set_parameters!(model, [[true_log_lambda]])
        
        # Manual simulation
        manual_times = Float64[]
        manual_censored = 0
        tmax = 10.0
        
        for i in 1:n_subjects
            event_time = rand(Exponential(1.0 / true_lambda))
            if event_time <= tmax
                push!(manual_times, event_time)
            else
                manual_censored += 1
            end
        end
        
        println("  Manual simulation:")
        println("    Events: ", length(manual_times))
        println("    Censored: ", manual_censored)
        println("    Mean event time: ", round(mean(manual_times), digits=3))
        
        # Package simulation
        sim_data = simulate(model; nsim = 1, data = true, paths = false)
        
        # Extract event times from simulation
        pkg_times = Float64[]
        pkg_censored = 0
        
        for i in 1:n_subjects
            subj_data = sim_data[sim_data.id .== i, :]
            if nrow(subj_data) > 0
                last_row = subj_data[end, :]
                if last_row.stateto == 2  # Event occurred
                    push!(pkg_times, last_row.tstop)
                else  # Censored
                    pkg_censored += 1
                end
            end
        end
        
        println("  Package simulation:")
        println("    Events: ", length(pkg_times))
        println("    Censored: ", pkg_censored)
        println("    Mean event time: ", round(mean(pkg_times), digits=3))
        
        # Statistical tests
        # 1. Number of events should be similar (binomial test)
        expected_events = n_subjects * (1 - exp(-true_lambda * tmax))
        @test abs(length(manual_times) - expected_events) < 3 * sqrt(expected_events * (1 - exp(-true_lambda * tmax)))
        @test abs(length(pkg_times) - expected_events) < 3 * sqrt(expected_events * (1 - exp(-true_lambda * tmax)))
        
        # 2. Event times should have similar mean
        expected_mean = 1/true_lambda * (1 - (1 + true_lambda*tmax)*exp(-true_lambda*tmax)) / (1 - exp(-true_lambda*tmax))
        @test abs(mean(manual_times) - expected_mean) < 0.5
        @test abs(mean(pkg_times) - expected_mean) < 0.5
        
        # 3. Kolmogorov-Smirnov test: distributions should be similar
        # Sort both samples
        manual_sorted = sort(manual_times)
        pkg_sorted = sort(pkg_times)
        
        # Empirical CDF comparison at quartiles
        quantiles = [0.25, 0.5, 0.75]
        for q in quantiles
            manual_q = quantile(manual_times, q)
            pkg_q = quantile(pkg_times, q)
            # Should be within 1 unit of each other
            @test abs(manual_q - pkg_q) < 1.0
        end
        
        println("  ✓ Distributions match")
    end
    
    # =========================================================================
    # Test 2: 2-state exponential - event/censoring proportions
    # =========================================================================
    @testset "Event/censoring proportions" begin
        println("\n--- Test 2: Event/censoring proportions ---")
        
        # Low hazard rate -> more censoring
        low_lambda = 0.05
        haz_low = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        n_subjects = 500
        tmax = 5.0
        
        init_data = DataFrame(
            id = 1:n_subjects,
            tstart = zeros(n_subjects),
            tstop = fill(tmax, n_subjects),
            statefrom = ones(Int, n_subjects),
            stateto = ones(Int, n_subjects),
            obstype = ones(Int, n_subjects)
        )
        
        model_low = multistatemodel(haz_low; data = init_data)
        set_parameters!(model_low, [[log(low_lambda)]])
        
        # Expected proportion of events
        expected_event_prop = 1 - exp(-low_lambda * tmax)
        
        println("  Expected event proportion: ", round(expected_event_prop, digits=3))
        
        # Simulate
        sim_data_low = simulate(model_low; nsim = 1, data = true)
        
        # Count events
        n_events = sum(sim_data_low.stateto .== 2)
        obs_event_prop = n_events / n_subjects
        
        println("  Observed event proportion: ", round(obs_event_prop, digits=3))
        
        # Should be within reasonable range (3 standard errors)
        se = sqrt(expected_event_prop * (1 - expected_event_prop) / n_subjects)
        @test abs(obs_event_prop - expected_event_prop) < 3 * se
        
        println("  ✓ Event proportion matches expected")
    end
    
    # =========================================================================
    # Test 3: 3-state competing risks
    # =========================================================================
    @testset "3-state competing risks" begin
        println("\n--- Test 3: 3-state competing risks (1→2, 1→3) ---")
        
        # Two competing transitions from state 1
        lambda_12 = 0.3
        lambda_13 = 0.2
        total_lambda = lambda_12 + lambda_13
        
        haz_12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        haz_13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        
        n_subjects = 1000
        tmax = 10.0
        
        init_data = DataFrame(
            id = 1:n_subjects,
            tstart = zeros(n_subjects),
            tstop = fill(tmax, n_subjects),
            statefrom = ones(Int, n_subjects),
            stateto = ones(Int, n_subjects),
            obstype = ones(Int, n_subjects)
        )
        
        model_competing = multistatemodel(haz_12, haz_13; data = init_data)
        set_parameters!(model_competing, [[log(lambda_12)], [log(lambda_13)]])
        
        # Expected proportions
        # P(no event) = exp(-total_lambda * tmax)
        # P(event to 2 | event) = lambda_12 / total_lambda
        # P(event to 3 | event) = lambda_13 / total_lambda
        
        p_no_event = exp(-total_lambda * tmax)
        p_event_to_2 = (1 - p_no_event) * (lambda_12 / total_lambda)
        p_event_to_3 = (1 - p_no_event) * (lambda_13 / total_lambda)
        
        println("  Expected proportions:")
        println("    No event (censored): ", round(p_no_event, digits=3))
        println("    Transition to 2: ", round(p_event_to_2, digits=3))
        println("    Transition to 3: ", round(p_event_to_3, digits=3))
        
        # Simulate
        sim_data_competing = simulate(model_competing; nsim = 1, data = true)
        
        # Count outcomes
        n_to_2 = sum(sim_data_competing.stateto .== 2)
        n_to_3 = sum(sim_data_competing.stateto .== 3)
        n_censored = sum(sim_data_competing.stateto .== 1)
        
        obs_p_to_2 = n_to_2 / n_subjects
        obs_p_to_3 = n_to_3 / n_subjects
        obs_p_censored = n_censored / n_subjects
        
        println("  Observed proportions:")
        println("    No event (censored): ", round(obs_p_censored, digits=3))
        println("    Transition to 2: ", round(obs_p_to_2, digits=3))
        println("    Transition to 3: ", round(obs_p_to_3, digits=3))
        
        # Test proportions (with tolerance)
        @test abs(obs_p_censored - p_no_event) < 0.05
        @test abs(obs_p_to_2 - p_event_to_2) < 0.05
        @test abs(obs_p_to_3 - p_event_to_3) < 0.05
        
        # Test that transition probabilities among events match
        if n_to_2 + n_to_3 > 0
            cond_p_to_2 = n_to_2 / (n_to_2 + n_to_3)
            expected_cond_p_to_2 = lambda_12 / total_lambda
            @test abs(cond_p_to_2 - expected_cond_p_to_2) < 0.05
            
            println("  Conditional P(2|event): obs=", round(cond_p_to_2, digits=3), 
                    " vs exp=", round(expected_cond_p_to_2, digits=3))
        end
        
        println("  ✓ Competing risks proportions match")
    end
    
    # =========================================================================
    # Test 4: Multiple simulations consistency
    # =========================================================================
    @testset "Multiple simulations consistency" begin
        println("\n--- Test 4: Multiple simulations consistency ---")
        
        lambda = 0.25
        haz = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        n_subjects = 200
        n_sims = 5
        tmax = 8.0
        
        init_data = DataFrame(
            id = 1:n_subjects,
            tstart = zeros(n_subjects),
            tstop = fill(tmax, n_subjects),
            statefrom = ones(Int, n_subjects),
            stateto = ones(Int, n_subjects),
            obstype = ones(Int, n_subjects)
        )
        
        model = multistatemodel(haz; data = init_data)
        set_parameters!(model, [[log(lambda)]])
        
        # Run multiple simulations
        event_props = Float64[]
        mean_times = Float64[]
        
        for sim in 1:n_sims
            sim_data = simulate(model; nsim = 1, data = true)
            n_events = sum(sim_data.stateto .== 2)
            push!(event_props, n_events / n_subjects)
            
            event_times = [row.tstop for row in eachrow(sim_data) if row.stateto == 2]
            if length(event_times) > 0
                push!(mean_times, mean(event_times))
            end
        end
        
        println("  Event proportions across ", n_sims, " simulations: ", 
                round.(event_props, digits=3))
        println("  Mean event times: ", round.(mean_times, digits=3))
        
        # Check consistency: std should be reasonable
        @test std(event_props) < 0.1
        if length(mean_times) > 1
            @test std(mean_times) < 1.5
        end
        
        println("  ✓ Multiple simulations are consistent")
    end
    
end

println("\n" * "=" ^ 80)
println("SIMULATION VALIDATION COMPLETE")
println("=" ^ 80)
