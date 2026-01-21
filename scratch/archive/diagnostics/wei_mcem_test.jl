using Test, MultistateModels, DataFrames, Random

# Get absolute path to project root
const PROJECT_ROOT = dirname(dirname(@__FILE__))

# Include the test infrastructure
include(joinpath(PROJECT_ROOT, "MultistateModelsTests/longtests/longtest_config.jl"))
include(joinpath(PROJECT_ROOT, "MultistateModelsTests/longtests/longtest_helpers.jl"))
include(joinpath(PROJECT_ROOT, "MultistateModelsTests/src/LongTestResults.jl"))

import MultistateModels: Hazard, @formula, multistatemodel, fit, set_parameters!, simulate, get_parameters_flat, get_parameters

# Weibull parameters from longtest_parametric_suite.jl
const TRUE_WEIBULL_SHAPE_12 = 1.3
const TRUE_WEIBULL_SHAPE_23 = 1.1
const TRUE_WEIBULL_SCALE_12 = 0.15
const TRUE_WEIBULL_SCALE_23 = 0.12

function run_weibull_panel_nocov()
    # Setup
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    true_params = (h12 = [TRUE_WEIBULL_SHAPE_12, TRUE_WEIBULL_SCALE_12], 
                   h23 = [TRUE_WEIBULL_SHAPE_23, TRUE_WEIBULL_SCALE_23])
    
    # Generate panel data  
    panel_times = collect(0.0:2.0:14.0)
    nobs = length(panel_times) - 1
    template = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(panel_times[1:end-1], N_SUBJECTS),
        tstop = repeat(panel_times[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs)
    )
    
    test_seed = hash(("wei", "mcem", "nocov", RNG_SEED))
    Random.seed!(test_seed)
    
    model_sim = multistatemodel(h12, h23; data=template)
    set_parameters!(model_sim, true_params)
    obstype_map = Dict(1 => 2, 2 => 1)
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false, obstype_by_transition=obstype_map)
    panel_data = sim_result[1, 1]
    
    println("Generated $(nrow(panel_data)) obs, $(length(unique(panel_data.id))) subjects")
    
    # Count events
    n_state3 = sum(panel_data[panel_data.obstype .== 1, :stateto] .== 3)
    println("Exact 2→3 transitions observed: $n_state3")
    
    # Fit
    model = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)
    fitted = fit(model; 
        verbose=true, 
        compute_vcov=true,
        method=:MCEM,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        max_ess=MCEM_ESS_MAX,
        maxiter=MCEM_MAX_ITER
    )
    
    # Results
    est = get_parameters(fitted)
    println("\nResults:")
    println("  h12_shape: $(est[:h12][1]) (true: $TRUE_WEIBULL_SHAPE_12)")
    println("  h12_scale: $(est[:h12][2]) (true: $TRUE_WEIBULL_SCALE_12)")
    println("  h23_shape: $(est[:h23][1]) (true: $TRUE_WEIBULL_SHAPE_23)")
    println("  h23_scale: $(est[:h23][2]) (true: $TRUE_WEIBULL_SCALE_23)")
    
    # Check tolerances
    rel_err = [
        abs(est[:h12][1] - TRUE_WEIBULL_SHAPE_12) / TRUE_WEIBULL_SHAPE_12,
        abs(est[:h12][2] - TRUE_WEIBULL_SCALE_12) / TRUE_WEIBULL_SCALE_12,
        abs(est[:h23][1] - TRUE_WEIBULL_SHAPE_23) / TRUE_WEIBULL_SHAPE_23,
        abs(est[:h23][2] - TRUE_WEIBULL_SCALE_23) / TRUE_WEIBULL_SCALE_23
    ]
    println("\nRelative errors: ", round.(rel_err .* 100, digits=1), "%")
    println("PARAM_REL_TOL = $(PARAM_REL_TOL*100)%")
    println("Test: ", all(rel_err .< PARAM_REL_TOL) ? "PASS" : "FAIL")
    
    return (est=est, rel_err=rel_err, fitted=fitted)
end

result = run_weibull_panel_nocov()
