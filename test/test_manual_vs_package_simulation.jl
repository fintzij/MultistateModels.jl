# Test: Manual vs Package Simulation Comparison
# Verify that manually simulated trajectories match package simulation distributions

using MultistateModels
using DataFrames
using Distributions
using Random
using Test
using HypothesisTests  # For KS test

println("=" ^ 80)
println("MANUAL VS PACKAGE SIMULATION COMPARISON")
println("=" ^ 80)

"""
    manual_simulate_exponential(lambda, n, tmax)

Manually simulate from exponential distribution with censoring at tmax.
Returns event times (for those who had events).
"""
function manual_simulate_exponential(lambda, n, tmax)
    event_times = Float64[]
    n_censored = 0
    
    for i in 1:n
        t = rand(Exponential(1.0 / lambda))
        if t <= tmax
            push!(event_times, t)
        else
            n_censored += 1
        end
    end
    
    return event_times, n_censored
end

"""
    manual_simulate_competing_risks(lambda_12, lambda_13, n, tmax)

Manually simulate competing risks: state 1 → 2 or state 1 → 3.
Returns vectors of times and destinations.
"""
function manual_simulate_competing_risks(lambda_12, lambda_13, n, tmax)
    total_lambda = lambda_12 + lambda_13
    prob_to_2 = lambda_12 / total_lambda
    
    times_to_2 = Float64[]
    times_to_3 = Float64[]
    n_censored = 0
    
    for i in 1:n
        # Time to any event
        t = rand(Exponential(1.0 / total_lambda))
        
        if t <= tmax
            # Which event?
            if rand() < prob_to_2
                push!(times_to_2, t)
            else
                push!(times_to_3, t)
            end
        else
            n_censored += 1
        end
    end
    
    return times_to_2, times_to_3, n_censored
end

"""
    manual_simulate_illness_death(lambda_12, lambda_13, lambda_23, n, tmax)

Manually simulate 3-state illness-death model:
  1 (Healthy) → 2 (Illness) → 3 (Death)
  1 (Healthy) → 3 (Death)
  
Returns DataFrame with transition information.
"""
function manual_simulate_illness_death(lambda_12, lambda_13, lambda_23, n, tmax)
    results = []
    
    for i in 1:n
        current_state = 1
        current_time = 0.0
        
        while current_state < 3 && current_time < tmax
            if current_state == 1
                # Competing risks: 1→2 vs 1→3
                total_rate = lambda_12 + lambda_13
                t = rand(Exponential(1.0 / total_rate))
                event_time = current_time + t
                
                if event_time > tmax
                    # Censored in state 1
                    push!(results, (id=i, state=1, time=tmax, censored=true))
                    break
                else
                    # Event occurs
                    prob_to_2 = lambda_12 / total_rate
                    if rand() < prob_to_2
                        # Transition to state 2
                        push!(results, (id=i, state=2, time=event_time, censored=false))
                        current_state = 2
                        current_time = event_time
                    else
                        # Transition to state 3
                        push!(results, (id=i, state=3, time=event_time, censored=false))
                        current_state = 3
                        current_time = event_time
                    end
                end
            elseif current_state == 2
                # Only one possible transition: 2→3
                t = rand(Exponential(1.0 / lambda_23))
                event_time = current_time + t
                
                if event_time > tmax
                    # Censored in state 2
                    push!(results, (id=i, state=2, time=tmax, censored=true))
                    break
                else
                    # Transition to state 3
                    push!(results, (id=i, state=3, time=event_time, censored=false))
                    current_state = 3
                    current_time = event_time
                end
            end
        end
    end
    
    return DataFrame(results)
end

Random.seed!(42)

@testset "Manual vs Package Simulation" begin
    
    # =========================================================================
    # Test 1: Simple 2-state exponential
    # =========================================================================
    @testset "2-state exponential" begin
        println("\n--- Test 1: 2-state exponential ---")
        
        lambda = 0.3
        n = 2000
        tmax = 10.0
        
        # Manual simulation
        manual_times, manual_censored = manual_simulate_exponential(lambda, n, tmax)
        
        println("  Manual simulation:")
        println("    Events: ", length(manual_times))
        println("    Censored: ", manual_censored)
        if length(manual_times) > 0
            println("    Mean event time: ", round(mean(manual_times), digits=3))
            println("    Median event time: ", round(median(manual_times), digits=3))
        end
        
        # Package simulation
        haz = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        init_data = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(tmax, n),
            statefrom = ones(Int, n),
            stateto = ones(Int, n),
            obstype = ones(Int, n)
        )
        
        model = multistatemodel(haz; data = init_data)
        set_parameters!(model, [[log(lambda)]])
        
        sim_data_matrix = simulate(model; nsim = 1, data = true, paths = false)
        sim_data = sim_data_matrix[1]  # Extract DataFrame from Matrix{DataFrame}
        
        # Extract event times
        pkg_times = [row.tstop for row in eachrow(sim_data) if row.stateto == 2]
        pkg_censored = sum(sim_data.stateto .== 1)
        
        println("  Package simulation:")
        println("    Events: ", length(pkg_times))
        println("    Censored: ", pkg_censored)
        if length(pkg_times) > 0
            println("    Mean event time: ", round(mean(pkg_times), digits=3))
            println("    Median event time: ", round(median(pkg_times), digits=3))
        end
        
        # Statistical comparison
        # 1. Event counts should be similar
        @test abs(length(manual_times) - length(pkg_times)) < 100  # Within 100 events
        
        # 2. Means should be similar
        if length(manual_times) > 0 && length(pkg_times) > 0
            @test abs(mean(manual_times) - mean(pkg_times)) < 0.5
        end
        
        # 3. Kolmogorov-Smirnov test: distributions should not be significantly different
        if length(manual_times) > 10 && length(pkg_times) > 10
            ks_result = ApproximateTwoSampleKSTest(manual_times, pkg_times)
            println("  KS test p-value: ", round(pvalue(ks_result), digits=4))
            @test pvalue(ks_result) > 0.01  # Should not reject at α=0.01
        end
        
        println("  ✓ Manual and package simulations match")
    end
    
    # =========================================================================
    # Test 2: Competing risks (1→2, 1→3)
    # =========================================================================
    @testset "Competing risks" begin
        println("\n--- Test 2: Competing risks ---")
        
        lambda_12 = 0.4
        lambda_13 = 0.2
        n = 2000
        tmax = 8.0
        
        # Manual simulation
        manual_to_2, manual_to_3, manual_censored = 
            manual_simulate_competing_risks(lambda_12, lambda_13, n, tmax)
        
        println("  Manual simulation:")
        println("    Transitions to 2: ", length(manual_to_2))
        println("    Transitions to 3: ", length(manual_to_3))
        println("    Censored: ", manual_censored)
        
        # Package simulation
        haz_12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        haz_13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        
        init_data = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(tmax, n),
            statefrom = ones(Int, n),
            stateto = ones(Int, n),
            obstype = ones(Int, n)
        )
        
        model = multistatemodel(haz_12, haz_13; data = init_data)
        set_parameters!(model, [[log(lambda_12)], [log(lambda_13)]])
        
        sim_data_matrix = simulate(model; nsim = 1, data = true)
        sim_data = sim_data_matrix[1]  # Extract DataFrame from Matrix{DataFrame}
        
        pkg_to_2 = [row.tstop for row in eachrow(sim_data) if row.stateto == 2]
        pkg_to_3 = [row.tstop for row in eachrow(sim_data) if row.stateto == 3]
        pkg_censored = sum(sim_data.stateto .== 1)
        
        println("  Package simulation:")
        println("    Transitions to 2: ", length(pkg_to_2))
        println("    Transitions to 3: ", length(pkg_to_3))
        println("    Censored: ", pkg_censored)
        
        # Compare counts
        @test abs(length(manual_to_2) - length(pkg_to_2)) < 100
        @test abs(length(manual_to_3) - length(pkg_to_3)) < 100
        @test abs(manual_censored - pkg_censored) < 100
        
        # Compare conditional probabilities
        manual_total_events = length(manual_to_2) + length(manual_to_3)
        pkg_total_events = length(pkg_to_2) + length(pkg_to_3)
        
        if manual_total_events > 0 && pkg_total_events > 0
            manual_prob_2 = length(manual_to_2) / manual_total_events
            pkg_prob_2 = length(pkg_to_2) / pkg_total_events
            
            println("  P(→2|event): manual=", round(manual_prob_2, digits=3),
                    " package=", round(pkg_prob_2, digits=3))
            
            @test abs(manual_prob_2 - pkg_prob_2) < 0.05
        end
        
        # KS tests for time distributions
        if length(manual_to_2) > 10 && length(pkg_to_2) > 10
            ks_2 = ApproximateTwoSampleKSTest(manual_to_2, pkg_to_2)
            println("  KS test (1→2 times) p-value: ", round(pvalue(ks_2), digits=4))
            @test pvalue(ks_2) > 0.01
        end
        
        if length(manual_to_3) > 10 && length(pkg_to_3) > 10
            ks_3 = ApproximateTwoSampleKSTest(manual_to_3, pkg_to_3)
            println("  KS test (1→3 times) p-value: ", round(pvalue(ks_3), digits=4))
            @test pvalue(ks_3) > 0.01
        end
        
        println("  ✓ Competing risks match")
    end
    
    # =========================================================================
    # Test 3: Illness-death model (1→2→3, 1→3)
    # =========================================================================
    @testset "Illness-death model" begin
        println("\n--- Test 3: Illness-death model ---")
        
        lambda_12 = 0.3
        lambda_13 = 0.1
        lambda_23 = 0.5
        n = 1500
        tmax = 10.0
        
        # Manual simulation
        manual_results = manual_simulate_illness_death(lambda_12, lambda_13, lambda_23, n, tmax)
        
        # Count final states
        manual_in_1 = sum((manual_results.state .== 1) .& manual_results.censored)
        manual_in_2 = sum((manual_results.state .== 2) .& manual_results.censored)
        manual_in_3 = sum(manual_results.state .== 3)
        
        println("  Manual simulation final states:")
        println("    State 1 (censored): ", manual_in_1)
        println("    State 2 (censored): ", manual_in_2)
        println("    State 3 (dead): ", manual_in_3)
        
        # Package simulation
        haz_12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        haz_13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        haz_23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        init_data = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(tmax, n),
            statefrom = ones(Int, n),
            stateto = ones(Int, n),
            obstype = ones(Int, n)
        )
        
        model = multistatemodel(haz_12, haz_13, haz_23; data = init_data)
        set_parameters!(model, [[log(lambda_12)], [log(lambda_13)], [log(lambda_23)]])
        
        sim_data_matrix = simulate(model; nsim = 1, data = true)
        sim_data = sim_data_matrix[1]  # Extract DataFrame from Matrix{DataFrame}
        
        # Get final state for each subject
        final_states = combine(groupby(sim_data, :id)) do subj_df
            DataFrame(final_state = subj_df[end, :stateto])
        end
        
        pkg_in_1 = sum(final_states.final_state .== 1)
        pkg_in_2 = sum(final_states.final_state .== 2)
        pkg_in_3 = sum(final_states.final_state .== 3)
        
        println("  Package simulation final states:")
        println("    State 1 (censored): ", pkg_in_1)
        println("    State 2 (censored): ", pkg_in_2)
        println("    State 3 (dead): ", pkg_in_3)
        
        # Compare final state distributions
        @test abs(manual_in_1 - pkg_in_1) < 100
        @test abs(manual_in_2 - pkg_in_2) < 100
        @test abs(manual_in_3 - pkg_in_3) < 100
        
        # Compare proportions
        manual_props = [manual_in_1, manual_in_2, manual_in_3] ./ n
        pkg_props = [pkg_in_1, pkg_in_2, pkg_in_3] ./ n
        
        println("  Proportions (manual): ", round.(manual_props, digits=3))
        println("  Proportions (package): ", round.(pkg_props, digits=3))
        
        for i in 1:3
            @test abs(manual_props[i] - pkg_props[i]) < 0.1
        end
        
        # Extract times to state 3 for both paths: direct (1→3) and via illness (1→2→3)
        # For manual simulation, we need to trace back transitions
        # This is complex, so we'll just compare overall time to death distribution
        
        manual_death_times = manual_results[manual_results.state .== 3, :time]
        
        # For package, get time of death for those who died
        pkg_death_times = Float64[]
        for id in unique(sim_data.id)
            subj_data = sim_data[sim_data.id .== id, :]
            if subj_data[end, :stateto] == 3
                # Find first time they entered state 3
                state_3_rows = subj_data[subj_data.stateto .== 3, :]
                if nrow(state_3_rows) > 0
                    push!(pkg_death_times, state_3_rows[1, :tstop])
                end
            end
        end
        
        if length(manual_death_times) > 10 && length(pkg_death_times) > 10
            println("  Death times - manual: mean=", round(mean(manual_death_times), digits=3),
                    " median=", round(median(manual_death_times), digits=3))
            println("  Death times - package: mean=", round(mean(pkg_death_times), digits=3),
                    " median=", round(median(pkg_death_times), digits=3))
            
            @test abs(mean(manual_death_times) - mean(pkg_death_times)) < 0.5
            
            ks_death = ApproximateTwoSampleKSTest(manual_death_times, pkg_death_times)
            println("  KS test (death times) p-value: ", round(pvalue(ks_death), digits=4))
            @test pvalue(ks_death) > 0.01
        end
        
        println("  ✓ Illness-death model matches")
    end
    
    # =========================================================================
    # Test 4: With covariates
    # =========================================================================
    @testset "With covariates" begin
        println("\n--- Test 4: Exponential with binary covariate ---")
        
        baseline_lambda = 0.2
        trt_effect = 0.5  # log hazard ratio
        trt_lambda = baseline_lambda * exp(trt_effect)
        
        n_control = 1000
        n_treated = 1000
        n_total = n_control + n_treated
        tmax = 10.0
        
        # Manual simulation
        manual_control, _ = manual_simulate_exponential(baseline_lambda, n_control, tmax)
        manual_treated, _ = manual_simulate_exponential(trt_lambda, n_treated, tmax)
        
        println("  Manual simulation:")
        println("    Control events: ", length(manual_control), 
                " (mean time: ", round(mean(manual_control), digits=3), ")")
        println("    Treated events: ", length(manual_treated),
                " (mean time: ", round(mean(manual_treated), digits=3), ")")
        
        # Package simulation
        haz = Hazard(@formula(0 ~ 1 + treatment), "exp", 1, 2)
        
        init_data = DataFrame(
            id = 1:n_total,
            tstart = zeros(n_total),
            tstop = fill(tmax, n_total),
            statefrom = ones(Int, n_total),
            stateto = ones(Int, n_total),
            obstype = ones(Int, n_total),
            treatment = [zeros(Int, n_control); ones(Int, n_treated)]
        )
        
        model = multistatemodel(haz; data = init_data)
        # Parameters: [log_baseline (intercept), trt_coef]
        # Intercept is now explicitly included in the formula
        set_parameters!(model, [[log(baseline_lambda), trt_effect]])
        
        sim_data_matrix = simulate(model; nsim = 1, data = true)
        sim_data = sim_data_matrix[1]  # Extract DataFrame from Matrix{DataFrame}
        
        # Simulated data already contains treatment column from model.data
        # No need to join with init_data
        
        pkg_control = [row.tstop for row in eachrow(sim_data) 
                       if row.treatment == 0 && row.stateto == 2]
        pkg_treated = [row.tstop for row in eachrow(sim_data) 
                       if row.treatment == 1 && row.stateto == 2]
        
        println("  Package simulation:")
        println("    Control events: ", length(pkg_control),
                " (mean time: ", round(mean(pkg_control), digits=3), ")")
        println("    Treated events: ", length(pkg_treated),
                " (mean time: ", round(mean(pkg_treated), digits=3), ")")
        
        # Compare
        if length(manual_control) > 10 && length(pkg_control) > 10
            @test abs(mean(manual_control) - mean(pkg_control)) < 0.5
            ks_control = ApproximateTwoSampleKSTest(manual_control, pkg_control)
            println("  KS test (control) p-value: ", round(pvalue(ks_control), digits=4))
            @test pvalue(ks_control) > 0.01
        end
        
        if length(manual_treated) > 10 && length(pkg_treated) > 10
            @test abs(mean(manual_treated) - mean(pkg_treated)) < 0.5
            ks_treated = ApproximateTwoSampleKSTest(manual_treated, pkg_treated)
            println("  KS test (treated) p-value: ", round(pvalue(ks_treated), digits=4))
            @test pvalue(ks_treated) > 0.01
        end
        
        println("  ✓ Covariate effects match")
    end
    
end

println("\n" * "=" ^ 80)
println("MANUAL VS PACKAGE SIMULATION COMPARISON COMPLETE")
println("=" ^ 80)
