using MultistateModels, Random, DataFrames, Printf

println("="^60)
println("Weibull Panel MCEM: Markov vs PhaseType Surrogate Comparison")
println("="^60)

Random.seed!(12345)

# True parameters (shape, rate) on NATURAL scale
# Use lower rates for more intermediate states observed
true_shape_12, true_rate_12 = 1.3, 0.15
true_shape_23, true_rate_23 = 1.2, 0.12

true_params = (
    h12 = [true_shape_12, true_rate_12],
    h23 = [true_shape_23, true_rate_23]
)

println("\nTrue parameters:")
println("  h12: shape=$(true_shape_12), rate=$(true_rate_12)")
println("  h23: shape=$(true_shape_23), rate=$(true_rate_23)")

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

# Generate panel data with 7 observation times (6 intervals)
n_subj = 500
obs_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

println("\nGenerating panel data:")
println("  n_subjects = $n_subj")
println("  obs_times = $obs_times")

nobs = length(obs_times)
template = DataFrame(
    id = repeat(1:n_subj, inner=nobs-1),
    tstart = repeat(obs_times[1:end-1], n_subj),
    tstop = repeat(obs_times[2:end], n_subj),
    statefrom = fill(1, n_subj * (nobs-1)),
    stateto = fill(1, n_subj * (nobs-1)),
    obstype = fill(2, n_subj * (nobs-1))
)

sim_model = multistatemodel(h12, h23; data=template)
set_parameters!(sim_model, true_params)
simdats, paths = simulate(sim_model; paths=true, data=true)
simdat = reduce(vcat, simdats)

# Check data quality
state_counts = combine(groupby(simdat, :stateto), nrow => :count)
println("\nSimulated data state distribution:")
for row in eachrow(state_counts)
    println("  State $(row.stateto): $(row.count) observations")
end

n_reached_2 = sum(any(simdat[simdat.id .== i, :stateto] .== 2) for i in 1:n_subj)
n_reached_3 = sum(any(simdat[simdat.id .== i, :stateto] .== 3) for i in 1:n_subj)
println("  Subjects reaching state 2: $n_reached_2")
println("  Subjects reaching state 3: $n_reached_3")

# ============================================================
# Fit with MARKOV surrogate
# ============================================================
println("\n" * "="^60)
println("Fitting with MARKOV surrogate...")
println("="^60)

model_markov = multistatemodel(h12, h23; data=simdat, surrogate=:markov)
markov_time = @elapsed fitted_markov = fit(model_markov;
    proposal = :markov, verbose = true, maxiter = 50, tol = 0.01, nsim = 500, maxiter_optim = 200)

est_markov = get_parameters(fitted_markov; scale=:natural)
println("\nMarkov results (time: $(round(markov_time, digits=1))s):")
println("  h12: shape=$(round(est_markov.h12[1], digits=4)), rate=$(round(est_markov.h12[2], digits=4))")
println("  h23: shape=$(round(est_markov.h23[1], digits=4)), rate=$(round(est_markov.h23[2], digits=4))")

rel_err_markov = (
    h12_shape = abs(est_markov.h12[1] - true_shape_12) / true_shape_12,
    h12_rate = abs(est_markov.h12[2] - true_rate_12) / true_rate_12,
    h23_shape = abs(est_markov.h23[1] - true_shape_23) / true_shape_23,
    h23_rate = abs(est_markov.h23[2] - true_rate_23) / true_rate_23
)
println("\nMarkov relative errors:")
println("  h12 shape: $(round(100*rel_err_markov.h12_shape, digits=2))%")
println("  h12 rate:  $(round(100*rel_err_markov.h12_rate, digits=2))%")
println("  h23 shape: $(round(100*rel_err_markov.h23_shape, digits=2))%")
println("  h23 rate:  $(round(100*rel_err_markov.h23_rate, digits=2))%")

# ============================================================
# Fit with PHASETYPE surrogate
# ============================================================
println("\n" * "="^60)
println("Fitting with PHASETYPE surrogate...")
println("="^60)

model_pt = multistatemodel(h12, h23; data=simdat, surrogate=:markov)

try
    pt_time = @elapsed fitted_pt = fit(model_pt;
        proposal = :phasetype, verbose = true, maxiter = 50, tol = 0.01, 
        ess_target_initial = 25, max_ess = 100,  # Lower ESS targets
        nsim = 100, maxiter_optim = 200)

    est_pt = get_parameters(fitted_pt; scale=:natural)
    println("\nPhaseType results (time: $(round(pt_time, digits=1))s):")
    println("  h12: shape=$(round(est_pt.h12[1], digits=4)), rate=$(round(est_pt.h12[2], digits=4))")
    println("  h23: shape=$(round(est_pt.h23[1], digits=4)), rate=$(round(est_pt.h23[2], digits=4))")

    rel_err_pt = (
        h12_shape = abs(est_pt.h12[1] - true_shape_12) / true_shape_12,
        h12_rate = abs(est_pt.h12[2] - true_rate_12) / true_rate_12,
        h23_shape = abs(est_pt.h23[1] - true_shape_23) / true_shape_23,
        h23_rate = abs(est_pt.h23[2] - true_rate_23) / true_rate_23
    )
    println("\nPhaseType relative errors:")
    println("  h12 shape: $(round(100*rel_err_pt.h12_shape, digits=2))%")
    println("  h12 rate:  $(round(100*rel_err_pt.h12_rate, digits=2))%")
    println("  h23 shape: $(round(100*rel_err_pt.h23_shape, digits=2))%")
    println("  h23 rate:  $(round(100*rel_err_pt.h23_rate, digits=2))%")
    
catch e
    println("\nPhaseType fitting FAILED with error:")
    println("  $(typeof(e))")
    if e isa TaskFailedException
        println("  Nested error: $(e.task.result)")
    else
        println("  $(e)")
    end
    println("\nStack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\n" * "="^60)
println("TEST COMPLETE")
println("="^60)