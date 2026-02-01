# Precompilation workload for MultistateModels.jl
# This reduces time-to-first-execution (TTFX) for common workflows
# 
# The workload runs during package precompilation and caches compiled native code.
# Users can disable this via Preferences if needed:
#   using MultistateModels, Preferences
#   set_preferences!(MultistateModels, "precompile_workload" => false)

using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    # Setup code that doesn't need to be precompiled
    using DataFrames
    
    # Create minimal valid test data for 3-state illness-death model
    # State 1 -> 2 -> 3 (absorbing)
    test_data = DataFrame(
        id = repeat(1:5, inner=2),
        tstart = repeat([0.0, 1.0], 5),
        tstop = repeat([1.0, 2.0], 5),
        statefrom = repeat([1, 2], 5),
        stateto = repeat([2, 3], 5),
        obstype = repeat([1, 1], 5),
        x = randn(10)
    )
    
    @compile_workload begin
        # === Exponential hazard (Markov) ===
        h_exp = [
            Hazard(@formula(0 ~ 1), :exp, 1, 2),
            Hazard(@formula(0 ~ 1), :exp, 2, 3)
        ]
        model_exp = multistatemodel(h_exp...; data=test_data, initialize=false)
        get_parameters(model_exp)
        
        # === With covariates ===
        h_cov = [
            Hazard(@formula(0 ~ 1 + x), :exp, 1, 2),
            Hazard(@formula(0 ~ 1), :exp, 2, 3)
        ]
        model_cov = multistatemodel(h_cov...; data=test_data, initialize=false)
        
        # === Weibull hazard (semi-Markov) ===
        h_wei = [
            Hazard(@formula(0 ~ 1), :wei, 1, 2),
            Hazard(@formula(0 ~ 1), :wei, 2, 3)
        ]
        model_wei = multistatemodel(h_wei...; data=test_data, initialize=false)
        
        # === Gompertz hazard ===
        h_gom = [
            Hazard(@formula(0 ~ 1), :gom, 1, 2),
            Hazard(@formula(0 ~ 1), :gom, 2, 3)
        ]
        model_gom = multistatemodel(h_gom...; data=test_data, initialize=false)
        
        # === Simulation (catch errors since random data may not work) ===
        try
            simulate(model_exp; tmax=3.0, nsim=2)
        catch e
            @debug "Precompile simulation skipped" exception=(e, catch_backtrace())
        end
        
        # === Fitting (short iterations, catch errors) ===
        try
            fit(model_exp; maxiter=2, verbose=false)
        catch e
            @debug "Precompile fit skipped" exception=(e, catch_backtrace())
        end
    end
end
