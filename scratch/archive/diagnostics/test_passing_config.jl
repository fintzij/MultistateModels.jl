# Test with same settings as passing longtest_mcem_splines.jl
using MultistateModels
using DataFrames
using Random
using Statistics

Random.seed!(0xABCD5678)

N_SUBJECTS = 1000
tmax = 5.0
obs_times = [0.0, 2.0, 4.0]
nobs = length(obs_times) - 1

# Simulate from Weibull (like Test 4 which passes)
true_shape = 1.8
true_scale = 0.25

h12_wei = Hazard(@formula(0 ~ 1), :wei, 1, 2)

sim_data = DataFrame(
    id = repeat(1:N_SUBJECTS, inner=nobs),
    tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
    tstop = repeat(obs_times[2:end], N_SUBJECTS),
    statefrom = ones(Int, N_SUBJECTS * nobs),
    stateto = ones(Int, N_SUBJECTS * nobs),
    obstype = fill(2, N_SUBJECTS * nobs)
)

model_sim = multistatemodel(h12_wei; data=sim_data, initialize=false)
set_parameters!(model_sim, (h12 = [true_shape, true_scale],))

obstype_map = Dict(1 => 1)  # transition 1 is exact
sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=obstype_map)
panel_data = sim_result[1, 1]

println("Data generated: $(nrow(panel_data)) rows")

# Fit with degree=2, 1 interior knot (like passing test)
h12_sp = Hazard(@formula(0 ~ 1), :sp, 1, 2; 
    degree=2,
    knots=[2.0],
    boundaryknots=[0.0, 5.0],
    extrapolation="constant")

model_sp = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)

println("\nFitting spline (degree=2, 1 knot, boundaries [0,5])...")
fitted = fit(model_sp;
    verbose=true, compute_vcov=false, penalty=:none,
    maxiter=25, tol=0.05, ess_target_initial=25, max_ess=300)

pars = get_parameters(fitted)
println("\nFitted parameters: $(pars.h12)")

# Evaluate hazard
pars_12 = get_parameters(fitted, 1, scale=:log)
h_vals = [fitted.hazards[1](t, pars_12, NamedTuple()) for t in 1.0:1.0:4.0]
println("Hazard at t=[1,2,3,4]: $(round.(h_vals, digits=4))")

# True Weibull hazard for comparison
wei_haz(t) = true_shape * true_scale * t^(true_shape - 1)
h_true = [wei_haz(t) for t in 1.0:1.0:4.0]
println("True Weibull hazard:   $(round.(h_true, digits=4))")

# Relative errors
rel_errs = abs.((h_vals .- h_true) ./ h_true)
println("\nRelative errors: $(round.(rel_errs * 100, digits=1))%")
println("Max relative error: $(round(maximum(rel_errs) * 100, digits=1))%")
