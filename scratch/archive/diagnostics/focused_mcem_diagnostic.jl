# Focused diagnostic: Compare single paths from Markov vs PhaseType proposals
# and their importance weights under the spline target model

using MultistateModels
using DataFrames
using Random
using Statistics
using Printf
using LinearAlgebra

import MultistateModels: 
    Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, get_hazard_params,
    MarkovSurrogate, PhaseTypeSurrogate, PhaseTypeProposal,
    _build_phasetype_from_markov, build_tpm_mapping,
    build_phasetype_tpm_book, fit_surrogate,
    loglik_phasetype_collapsed_path, loglik, @formula,
    SamplePath, unflatten_parameters

println("="^70)
println("Focused MCEM Path Diagnostic")
println("="^70)

# Generate data
Random.seed!(0xABCD5678)
true_rate = 0.3
n_subjects = 500

h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
obs_times = [0.0, 2.0, 4.0]
nobs = length(obs_times) - 1

sim_data = DataFrame(
    id = repeat(1:n_subjects, inner=nobs),
    tstart = repeat(obs_times[1:end-1], n_subjects),
    tstop = repeat(obs_times[2:end], n_subjects),
    statefrom = ones(Int, n_subjects * nobs),
    stateto = ones(Int, n_subjects * nobs),
    obstype = fill(2, n_subjects * nobs)
)

model_dgp = multistatemodel(h12_exp; data=sim_data, initialize=false)
set_parameters!(model_dgp, (h12 = [true_rate],))
obstype_map = Dict(1 => 1)
sim_result = simulate(model_dgp; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=obstype_map)
panel_data = sim_result[1, 1]

println("\n1. Data generated with true rate = $true_rate")
println("   N transitions: $(sum(panel_data.stateto .== 2 .&& panel_data.statefrom .== 1))")

# Create spline model
h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                degree=1, knots=Float64[], boundaryknots=[0.0, 5.0], 
                extrapolation="linear")
model = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)

println("\n2. Model initial parameters:")
println("   Flat: $(get_parameters_flat(model))")

# Fit surrogate
fit_surrogate(model; type=:markov, method=:mle, verbose=false)
markov_surrogate = model.markovsurrogate

surrog_rate = markov_surrogate.parameters.nested.h12.baseline.h12_rate
println("\n3. Markov surrogate rate: $(@sprintf("%.4f", surrog_rate))")

# Build PhaseType surrogate
pt_surrogate = _build_phasetype_from_markov(
    model, markov_surrogate;
    config=PhaseTypeProposal(n_phases=3),
    verbose=false
)

println("\n4. Creating test paths manually...")

# Create test paths that span the observation period
# Path 1: No transition (stays in state 1)
path_no_trans = SamplePath(1, [0.0, 4.0], [1, 1])

# Path 2: Transition at t=1.5
path_early = SamplePath(1, [0.0, 1.5], [1, 2])

# Path 3: Transition at t=3.0
path_late = SamplePath(1, [0.0, 3.0], [1, 2])

# Get parameters for evaluation
params_flat = get_parameters_flat(model)
target_pars = unflatten_parameters(params_flat, model)
surrogate_pars = get_hazard_params(markov_surrogate.parameters, markov_surrogate.hazards)

println("\n5. Log-likelihood comparison for test paths:")
println("   " * "-"^65)
println("   Path                  Target(sp)    Markov(sur)   PT(sur)     Weight(MK)  Weight(PT)")
println("   " * "-"^65)

for (name, path) in [("No transition t=4", path_no_trans),
                      ("Trans at t=1.5", path_early),
                      ("Trans at t=3.0", path_late)]
    ll_target = loglik(target_pars, path, model.hazards, model)
    ll_markov = loglik(surrogate_pars, path, markov_surrogate.hazards, model)
    ll_pt = loglik_phasetype_collapsed_path(path, pt_surrogate)
    
    w_mk = ll_target - ll_markov
    w_pt = ll_target - ll_pt
    
    println("   $(rpad(name, 20))  $(@sprintf("%10.4f", ll_target))    $(@sprintf("%10.4f", ll_markov))    $(@sprintf("%8.4f", ll_pt))    $(@sprintf("%8.4f", w_mk))    $(@sprintf("%8.4f", w_pt))")
end
println("   " * "-"^65)

println("\n6. Key question: Are importance weights different for MK vs PT?")
println("   If so, the problem is in the surrogate log-likelihood calculation.")
println("   If not, the problem is in sampling distribution or MCEM loop.")

# Now let's check the ACTUAL paths sampled during MCEM
println("\n7. Running single MCEM iteration to examine paths...")

# Run fit with return_proposed_paths=true
println("\n   Fitting with Markov proposal (return_proposed_paths=true)...")
model_mk = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
fitted_mk = fit(model_mk;
    proposal=:markov,
    penalty=:none,
    verbose=false,
    maxiter=3,  # Just a few iterations
    tol=0.001,
    ess_target_initial=10,
    max_ess=50,
    compute_vcov=false,
    compute_ij_vcov=false,
    return_proposed_paths=true)

println("   Markov fit spline coefficients: $(round.(get_parameters_flat(fitted_mk), digits=4))")

println("\n   Fitting with PhaseType proposal (return_proposed_paths=true)...")
model_pt = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
fitted_pt = fit(model_pt;
    proposal=PhaseTypeProposal(n_phases=3),
    penalty=:none,
    verbose=false,
    maxiter=3,  # Just a few iterations
    tol=0.001,
    ess_target_initial=10,
    max_ess=50,
    compute_vcov=false,
    compute_ij_vcov=false,
    return_proposed_paths=true)

println("   PhaseType fit spline coefficients: $(round.(get_parameters_flat(fitted_pt), digits=4))")

# Check the sampled paths
println("\n8. Examining sampled paths from each proposal...")

if hasfield(typeof(fitted_mk), :ProposedPaths) && !isnothing(fitted_mk.ProposedPaths)
    mk_paths = fitted_mk.ProposedPaths.samplepaths
    println("   Markov: $(length(mk_paths)) subjects with paths")
    
    # Look at first subject's paths
    if length(mk_paths) > 0 && length(mk_paths[1]) > 0
        println("   Subject 1 Markov paths (first 3):")
        for (j, p) in enumerate(mk_paths[1][1:min(3, length(mk_paths[1]))])
            println("     Path $j: times=$(round.(p.times, digits=2)), states=$(p.states)")
        end
    end
else
    println("   Markov: ProposedPaths not available")
end

if hasfield(typeof(fitted_pt), :ProposedPaths) && !isnothing(fitted_pt.ProposedPaths)
    pt_paths = fitted_pt.ProposedPaths.samplepaths
    println("   PhaseType: $(length(pt_paths)) subjects with paths")
    
    # Look at first subject's paths
    if length(pt_paths) > 0 && length(pt_paths[1]) > 0
        println("   Subject 1 PhaseType paths (first 3):")
        for (j, p) in enumerate(pt_paths[1][1:min(3, length(pt_paths[1]))])
            println("     Path $j: times=$(round.(p.times, digits=2)), states=$(p.states)")
        end
    end
else
    println("   PhaseType: ProposedPaths not available")
end

println("\n" * "="^70)
