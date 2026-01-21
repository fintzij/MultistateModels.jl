using MultistateModels
using DataFrames
using Random
using Statistics

Random.seed!(12345)

println("="^70)
println("DIAGNOSTIC: Degree-0 Spline with Exponential DGP")
println("="^70)

true_rate = 0.3
expected_mean_sojourn = 1 / true_rate
println("
1. DATA GENERATION")
println("   True exponential rate: ", true_rate)
println("   Expected mean sojourn time: ", round(expected_mean_sojourn, digits=3))

N_SUBJECTS = 500
h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)

obs_times = [0.0, 2.0, 5.0, 10.0]
nobs = length(obs_times) - 1

sim_data = DataFrame(
    id = repeat(1:N_SUBJECTS, inner=nobs),
    tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
    tstop = repeat(obs_times[2:end], N_SUBJECTS),
    statefrom = ones(Int, N_SUBJECTS * nobs),
    stateto = ones(Int, N_SUBJECTS * nobs),
    obstype = fill(2, N_SUBJECTS * nobs)
)

model_sim = multistatemodel(h12_exp; data=sim_data, initialize=false)
MultistateModels.set_parameters!(model_sim, (h12 = [true_rate],))

obstype_map = Dict(1 => 1)
sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false, obstype_by_transition=obstype_map)
panel_data = sim_result[1, 1]

transitioned = panel_data[panel_data.statefrom .!= panel_data.stateto, :]
observed_sojourns = transitioned.tstop .- transitioned.tstart
actual_mean_sojourn = mean(observed_sojourns)
println("   Simulated data: ", N_SUBJECTS, " subjects")
println("   Observed transitions: ", nrow(transitioned))
println("   Actual mean observed sojourn: ", round(actual_mean_sojourn, digits=3))

println("
2. SPLINE MODEL SETUP")
max_time = maximum(panel_data.tstop)
interior_knot = actual_mean_sojourn
println("   Degree: 0 (piecewise constant)")
println("   Interior knot: ", round(interior_knot, digits=3))
println("   Boundary knots: [0.0, ", round(max_time, digits=2), "]")

h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=0, knots=[interior_knot], boundaryknots=[0.0, max_time], extrapolation="constant")

println("
3. FIT WITH MARKOV PROPOSAL")
model_markov = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
println("   Initial params: ", model_markov.parameters.flat)
println("   Number of spline coefficients: ", model_markov.hazards[1].npar_baseline)

fitted_markov = fit(model_markov; proposal=:markov, verbose=false, maxiter=30, tol=0.01, ess_target_initial=50, max_ess=500, compute_vcov=false)

pars_markov = MultistateModels.get_parameters(fitted_markov, 1, scale=:log)
println("   Fitted params: ", round.(pars_markov, digits=4))

println("
   Hazard evaluation:")
for t in [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    h_val = fitted_markov.hazards[1](t, pars_markov, NamedTuple())
    segment = t < interior_knot ? "segment 1" : "segment 2"
    println("     h(", t, ") = ", round(h_val, digits=4), " [", segment, "]")
end
