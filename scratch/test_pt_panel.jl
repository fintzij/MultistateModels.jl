using MultistateModels, DataFrames, Random

RNG_SEED = 12345
N_SUBJECTS = 200
PANEL_TIMES = [0.0, 2.0, 4.0, 6.0, 8.0]
MAX_TIME = 20.0

Random.seed!(RNG_SEED + 100)

println("--- 2-Phase Phase-Type with Fixed Covariate (Panel Data) ---")

n_subj = N_SUBJECTS
cov_vals = rand([0.0, 1.0], n_subj)

exact_template = DataFrame(
    id = 1:n_subj,
    tstart = zeros(n_subj),
    tstop = fill(MAX_TIME, n_subj),
    statefrom = ones(Int, n_subj),
    stateto = ones(Int, n_subj),
    obstype = ones(Int, n_subj),
    x = cov_vals
)

h12 = Hazard(@formula(0 ~ x), :pt, 1, 2; n_phases=2)
model_sim = multistatemodel(h12; data=exact_template, verbose=false)

true_lambda = 0.4
true_mu1 = 0.25
true_mu2 = 0.5
true_beta = 0.35

params_sim = (
    h1_ab = [true_lambda],
    h12_a = [true_mu1, true_beta],
    h12_b = [true_mu2, true_beta]
)
set_parameters!(model_sim, params_sim)

println("  True params: lambda=$true_lambda, mu1=$true_mu1, mu2=$true_mu2, beta=$true_beta")

sim_result = simulate(model_sim; paths=true, data=true, nsim=1)
paths = sim_result[2][1]

panel_rows = []
for path in paths
    subj_id = path.subj
    x_val = cov_vals[subj_id]
    for i in 1:(length(PANEL_TIMES)-1)
        t_start = PANEL_TIMES[i]
        t_stop = PANEL_TIMES[i+1]
        idx_start = searchsortedlast(path.times, t_start)
        state_start = idx_start >= 1 ? path.states[idx_start] : 1
        idx_stop = searchsortedlast(path.times, t_stop)
        state_stop = idx_stop >= 1 ? path.states[idx_stop] : 1
        push!(panel_rows, (id=subj_id, tstart=t_start, tstop=t_stop, statefrom=state_start, stateto=state_stop, obstype=2, x=x_val))
    end
end
panel_data = DataFrame(panel_rows)

println("  Panel data: $(nrow(panel_data)) observations")

model_fit = multistatemodel(h12; data=panel_data, verbose=false)

println("Fitting...")
fitted = fit(model_fit; verbose=false, compute_vcov=false)
fitted_params = MultistateModels.get_parameters_flat(fitted)

println("  True params:   [$true_lambda, $true_mu1, $true_beta, $true_mu2, $true_beta]")
println("  Fitted params: $(round.(fitted_params, digits=4))")

beta1_idx = 3
beta2_idx = 5

test1 = abs(fitted_params[beta1_idx] - fitted_params[beta2_idx]) < 0.1
test2 = fitted_params[beta1_idx] > -0.5
test3 = fitted_params[1] > 0.01
test4 = fitted_params[2] > 0.01
test5 = fitted_params[4] > 0.01

println()
println("=== Test Results ===")
println("  Homogeneous constraint: $test1 (diff=$(round(abs(fitted_params[beta1_idx] - fitted_params[beta2_idx]), digits=4)))")
println("  Beta reasonable: $test2 (beta=$(round(fitted_params[beta1_idx], digits=4)))")
println("  lambda > 0.01: $test3")
println("  mu1 > 0.01: $test4")
println("  mu2 > 0.01: $test5")
println()
println("All tests pass: $(test1 && test2 && test3 && test4 && test5)")
