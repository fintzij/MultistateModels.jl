using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using Printf

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, MarkovProposal, PhaseTypeProposal, @formula

const RNG_SEED = 0xABCD1234
const N_SUBJECTS = 500

# True Weibull parameters
true_shape_12, true_scale_12 = 1.3, 0.15
true_shape_23, true_scale_23 = 1.1, 0.12

println("="^70)
println("PhaseType MCEM Test - After Fix")
println("="^70)
println("\nTrue parameters:")
println("  h12: shape=$(true_shape_12), scale=$(true_scale_12)")
println("  h23: shape=$(true_shape_23), scale=$(true_scale_23)")

# Generate panel data
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

obs_times = collect(0.0:2.0:14.0)
nobs = length(obs_times) - 1

template = DataFrame(
    id = repeat(1:N_SUBJECTS, inner=nobs),
    tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
    tstop = repeat(obs_times[2:end], N_SUBJECTS),
    statefrom = ones(Int, N_SUBJECTS * nobs),
    stateto = ones(Int, N_SUBJECTS * nobs),
    obstype = fill(2, N_SUBJECTS * nobs)
)

Random.seed!(RNG_SEED + 10)
model_template = multistatemodel(h12, h23; data=template, initialize=false)
for (haz_idx, haz_name) in enumerate(keys((h12=[true_shape_12, true_scale_12], h23=[true_shape_23, true_scale_23])))
    set_parameters!(model_template, haz_idx, haz_idx == 1 ? [true_shape_12, true_scale_12] : [true_shape_23, true_scale_23])
end

# Simulate with obstype_map
obstype_map = Dict(1 => 2, 2 => 1)
sim_result = simulate(model_template; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=obstype_map)
panel_data = sim_result[1, 1]

println("\nGenerated panel data:")
println("  N subjects: $N_SUBJECTS")
println("  N observations: $(nrow(panel_data))")

# Fit with PhaseType proposal (3 phases)
println("\n" * "="^70)
println("Fitting with PhaseType proposal (3 phases, :unstructured)")
println("="^70)

h12_pt = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h23_pt = Hazard(@formula(0 ~ 1), "wei", 2, 3)
model_pt = multistatemodel(h12_pt, h23_pt; data=panel_data, surrogate=:markov)

# Fit
fitted_pt = fit(model_pt; 
    proposal=PhaseTypeProposal(n_phases=3),
    verbose=true, 
    maxiter=100, 
    tol=0.01,
    ess_target_initial=100,
    max_ess=500,
    compute_vcov=false
)

# Get estimates (correct API)
est_pt = get_parameters(fitted_pt)
shape_12_pt = est_pt[:h12][1]
scale_12_pt = est_pt[:h12][2]
shape_23_pt = est_pt[:h23][1]
scale_23_pt = est_pt[:h23][2]

println("\n" * "="^70)
println("RESULTS")
println("="^70)
println("\nPhaseType Proposal Results:")
@printf("  h12: shape=%.4f (true=%.2f, err=%.1f%%), scale=%.4f (true=%.2f, err=%.1f%%)\n",
    shape_12_pt, true_shape_12, 100*abs(shape_12_pt - true_shape_12)/true_shape_12,
    scale_12_pt, true_scale_12, 100*abs(scale_12_pt - true_scale_12)/true_scale_12)
@printf("  h23: shape=%.4f (true=%.2f, err=%.1f%%), scale=%.4f (true=%.2f, err=%.1f%%)\n",
    shape_23_pt, true_shape_23, 100*abs(shape_23_pt - true_shape_23)/true_shape_23,
    scale_23_pt, true_scale_23, 100*abs(scale_23_pt - true_scale_23)/true_scale_23)
