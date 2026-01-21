# MCEM Iteration Trace - Find where Markov and PhaseType diverge
# Focus: What exactly happens in ONE subject's E-step and M-step contribution

using MultistateModels
using Random
using DataFrames
using LinearAlgebra

println("="^60)
println("MCEM ITERATION TRACE")
println("="^60)

# Create minimal test case: ONE subject with panel observation
# This is the simplest case that should work
Random.seed!(12345)

# True hazard rate = 0.3 (constant)
true_rate = 0.3

# Generate ONE subject with panel observation
# Subject 1: starts in state 1, observed at time 0 and time 2, still in state 1
subject_data = DataFrame(
    id = [1, 1],
    tstart = [0.0, 2.0],
    tstop = [2.0, 4.0],
    statefrom = [1, 1],  # Panel: observed in state 1 at t=0 and t=2
    stateto = [1, 1],    # Still in state 1 at t=2 and t=4 (censored)
    obstype = [2, 2]     # Panel observation
)

println("\nSubject data:")
println(subject_data)
println("\nThis subject: observed in state 1 at times 0, 2, 4")
println("Could have been in state 2 at any point without us knowing")

# Build models
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=1, knots=Float64[], boundaryknots=[0.0, 5.0])

model = multistatemodel(h12; data=subject_data, surrogate=:markov)

# Get initial parameters (should be log(0.3) ≈ -1.2 for constant hazard)
init_params = get_parameters(model, scale=:optimizer)
println("\nInitial parameters: ", init_params)

# Fit the surrogate
println("\n" * "="^60)
println("STEP 1: Fit Markov surrogate")
println("="^60)

surrogate_fitted = MultistateModels.fit(model; method=:MLE)
surrogate_params = get_parameters(surrogate_fitted.model, scale=:optimizer)
println("Surrogate fitted params: ", surrogate_params)

# Now let's trace what happens during MCEM E-step
println("\n" * "="^60)
println("STEP 2: Sample paths from surrogate")
println("="^60)

# For Markov proposal, paths are sampled using surrogate parameters
# For PhaseType proposal, paths are sampled using phase-type approximation

# Get surrogate info
data = MultistateModels.extract_data(model)
subj = data.subjects[1]

println("\nSubject info:")
println("  n_obs: ", subj.n_obs)
println("  times: ", subj.times)
println("  states: ", subj.states)
println("  obstypes: ", subj.obstypes)

# Build sample path matrix for this subject
# Markov sampler needs hazards, emat, tpm_book
hazards = model.hazards
emat = model.emat
tpm_book = model.tpm_book

println("\n--- Drawing Markov paths ---")
Random.seed!(42)
markov_paths = []
for i in 1:5
    path = MultistateModels.draw_samplepath(subj, hazards, emat, tpm_book, data.tpm_map)
    push!(markov_paths, (times=copy(path.times), states=copy(path.states)))
    println("Path $i: times=$(round.(path.times, digits=3)), states=$(path.states)")
end

# Now for PhaseType
println("\n--- Setting up PhaseType sampler ---")
pt_nphases = 3

# Build PhaseType model infrastructure
hazards_ph, model_ph = MultistateModels.build_phasetype_hazards_model(
    model, pt_nphases; surrogate=:none
)

# Transfer surrogate parameters to expanded model
MultistateModels.set_phasetype_parameters_from_collapsed!(model_ph, model)

# Get PhaseType sampling infrastructure
emat_ph = model_ph.emat
tpm_book_ph = model_ph.tpm_book

# Build FB matrices for this subject
fbmats_ph = MultistateModels.build_fbmats(subj, tpm_book_ph.tpm_trange, model_ph)

println("PhaseType model states: ", model_ph._totalnstates)

println("\n--- Drawing PhaseType paths ---")
Random.seed!(42)
pt_paths_expanded = []
pt_paths_collapsed = []
for i in 1:5
    # Sample in expanded space
    path_exp = MultistateModels.draw_samplepath_phasetype(subj, hazards_ph, emat_ph, fbmats_ph, model_ph)
    
    # Collapse to observed space
    path_collapsed = MultistateModels.collapse_phasetype_path(path_exp)
    
    push!(pt_paths_expanded, (times=copy(path_exp.times), states=copy(path_exp.states)))
    push!(pt_paths_collapsed, (times=copy(path_collapsed.times), states=copy(path_collapsed.states)))
    
    println("Path $i expanded: times=$(round.(path_exp.times, digits=3)), states=$(path_exp.states)")
    println("        collapsed: times=$(round.(path_collapsed.times, digits=3)), states=$(path_collapsed.states)")
end

# Now compute importance weights
println("\n" * "="^60)
println("STEP 3: Compute importance weights")
println("="^60)

# For MCEM, weight = f(path | θ_target) / q(path | θ_proposal)
# Both should use the SAME target parameters

target_params = surrogate_params  # Start from surrogate

println("\nTarget params: ", target_params)

println("\n--- Markov path weights ---")
for (i, path) in enumerate(markov_paths)
    # Create SamplePath object
    sp = MultistateModels.SamplePath(path.times, path.states)
    
    # Target log-lik (under spline model)
    ll_target = MultistateModels.loglik(target_params, sp, hazards, model)
    
    # Proposal log-lik (under Markov surrogate)
    ll_proposal = MultistateModels.loglik(target_params, sp, hazards, model)  # Same for Markov!
    
    # Weight
    log_weight = ll_target - ll_proposal
    
    println("Path $i: ll_target=$ll_target, ll_proposal=$ll_proposal, log_weight=$log_weight")
end

println("\n--- PhaseType path weights ---")
for (i, (path_exp, path_col)) in enumerate(zip(pt_paths_expanded, pt_paths_collapsed))
    # Create SamplePath objects
    sp_exp = MultistateModels.SamplePath(path_exp.times, path_exp.states)
    sp_col = MultistateModels.SamplePath(path_col.times, path_col.states)
    
    # Target log-lik (under spline model, on COLLAPSED path)
    ll_target = MultistateModels.loglik(target_params, sp_col, hazards, model)
    
    # Proposal log-lik (under PhaseType surrogate, on EXPANDED path)
    # This uses the PhaseType infrastructure
    ll_proposal_collapsed = MultistateModels.loglik_phasetype_collapsed_path(
        sp_exp.times, sp_exp.states, model_ph
    )
    
    # Weight
    log_weight = ll_target - ll_proposal_collapsed
    
    println("Path $i: ll_target=$ll_target, ll_proposal=$ll_proposal_collapsed, log_weight=$log_weight")
end

# KEY COMPARISON: Are the path distributions the same?
println("\n" * "="^60)
println("STEP 4: Compare path distributions")
println("="^60)

Random.seed!(12345)
n_samples = 1000

# Sample many Markov paths
markov_sojourns = Float64[]
markov_transitions = Int[]
for _ in 1:n_samples
    path = MultistateModels.draw_samplepath(subj, hazards, emat, tpm_book, data.tpm_map)
    # Record first sojourn time in state 1 (if transitions)
    for j in 2:length(path.states)
        if path.states[j-1] == 1 && path.states[j] != 1
            push!(markov_sojourns, path.times[j] - path.times[j-1])
            push!(markov_transitions, 1)
            break
        end
    end
end

# Sample many PhaseType paths (collapsed)
Random.seed!(12345)
pt_sojourns = Float64[]
pt_transitions = Int[]
for _ in 1:n_samples
    path_exp = MultistateModels.draw_samplepath_phasetype(subj, hazards_ph, emat_ph, fbmats_ph, model_ph)
    path_col = MultistateModels.collapse_phasetype_path(path_exp)
    
    for j in 2:length(path_col.states)
        if path_col.states[j-1] == 1 && path_col.states[j] != 1
            push!(pt_sojourns, path_col.times[j] - path_col.times[j-1])
            push!(pt_transitions, 1)
            break
        end
    end
end

println("\nMarkov: $(length(markov_transitions)) paths had 1→2 transition")
println("PhaseType: $(length(pt_transitions)) paths had 1→2 transition")

if length(markov_sojourns) > 0 && length(pt_sojourns) > 0
    println("\nMarkov sojourn times: mean=$(round(mean(markov_sojourns), digits=3)), std=$(round(std(markov_sojourns), digits=3))")
    println("PhaseType sojourn times: mean=$(round(mean(pt_sojourns), digits=3)), std=$(round(std(pt_sojourns), digits=3))")
end

println("\n" * "="^60)
println("DONE - Review above for discrepancies")
println("="^60)
