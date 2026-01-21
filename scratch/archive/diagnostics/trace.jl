using MultistateModels, Random, DataFrames, Statistics

println("="^60)
println("MCEM ITERATION TRACE - FOCUSED DIAGNOSTIC")
println("="^60)

# Panel data: subject observed in state 1 at t=0, t=2, t=4 (censored)
panel_data = DataFrame(id=[1,1], tstart=[0.0,2.0], tstop=[2.0,4.0], 
                       statefrom=[1,1], stateto=[1,1], obstype=[2,2])
println("\nPanel data:")
println(panel_data)

# Build model with spline hazard
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=1, knots=Float64[], boundaryknots=[0.0, 5.0])
model = multistatemodel(h12; data=panel_data, surrogate=:markov)

# Get data structures
data_obj = MultistateModels.extract_data(model)
subj = data_obj.subjects[1]
hazards, emat, tpm_book = model.hazards, model.emat, model.tpm_book

println("\nSubject info:")
println("  n_obs: ", subj.n_obs)
println("  times: ", subj.times)
println("  states: ", subj.states)
println("  obstypes: ", subj.obstypes)

println("\n" * "="^60)
println("Step 1: Draw Markov paths")
println("="^60)
Random.seed!(42)
for i in 1:5
    path = MultistateModels.draw_samplepath(subj, hazards, emat, tpm_book, data_obj.tpm_map)
    println("Path $i: times=$(round.(path.times,digits=3)), states=$(path.states)")
end

println("\n" * "="^60)
println("Step 2: Setup PhaseType and draw paths")
println("="^60)
pt_nphases = 3
hazards_ph, model_ph = MultistateModels.build_phasetype_hazards_model(model, pt_nphases; surrogate=:none)
MultistateModels.set_phasetype_parameters_from_collapsed!(model_ph, model)
fbmats_ph = MultistateModels.build_fbmats(subj, model_ph.tpm_book.tpm_trange, model_ph)
println("PhaseType expanded states: ", model_ph._totalnstates)

Random.seed!(42)
for i in 1:5
    path_exp = MultistateModels.draw_samplepath_phasetype(subj, hazards_ph, model_ph.emat, fbmats_ph, model_ph)
    path_col = MultistateModels.collapse_phasetype_path(path_exp)
    println("Path $i (exp): times=$(round.(path_exp.times,digits=3)), states=$(path_exp.states)")
    println("       (col): times=$(round.(path_col.times,digits=3)), states=$(path_col.states)")
end

println("\n" * "="^60)
println("Step 3: Compare sojourn time distributions (500 samples)")
println("="^60)

Random.seed!(12345)
markov_transitions = 0
markov_survivals = 0
markov_sojourns = Float64[]
for _ in 1:500
    path = MultistateModels.draw_samplepath(subj, hazards, emat, tpm_book, data_obj.tpm_map)
    had_transition = false
    for j in 2:length(path.states)
        if path.states[j-1] == 1 && path.states[j] != 1
            push!(markov_sojourns, path.times[j] - path.times[j-1])
            had_transition = true
            break
        end
    end
    if had_transition
        markov_transitions += 1
    else
        markov_survivals += 1
    end
end

Random.seed!(12345)
pt_transitions = 0
pt_survivals = 0
pt_sojourns = Float64[]
for _ in 1:500
    path_exp = MultistateModels.draw_samplepath_phasetype(subj, hazards_ph, model_ph.emat, fbmats_ph, model_ph)
    path_col = MultistateModels.collapse_phasetype_path(path_exp)
    had_transition = false
    for j in 2:length(path_col.states)
        if path_col.states[j-1] == 1 && path_col.states[j] != 1
            push!(pt_sojourns, path_col.times[j] - path_col.times[j-1])
            had_transition = true
            break
        end
    end
    if had_transition
        pt_transitions += 1
    else
        pt_survivals += 1
    end
end

println("\nMarkov: $(markov_transitions) transitions, $(markov_survivals) survivals")
println("PhaseType: $(pt_transitions) transitions, $(pt_survivals) survivals")

if length(markov_sojourns) > 10
    println("\nMarkov sojourn times: mean=$(round(mean(markov_sojourns),digits=3)), std=$(round(std(markov_sojourns),digits=3))")
end
if length(pt_sojourns) > 10
    println("PhaseType sojourn times: mean=$(round(mean(pt_sojourns),digits=3)), std=$(round(std(pt_sojourns),digits=3))")
end

# Check if transition rates differ significantly
if markov_transitions > 0 && pt_transitions > 0
    markov_rate = markov_transitions / 500
    pt_rate = pt_transitions / 500
    println("\nTransition probability - Markov: $(round(markov_rate,digits=3)), PhaseType: $(round(pt_rate,digits=3))")
    if abs(markov_rate - pt_rate) / max(markov_rate, pt_rate) > 0.1
        println("WARNING: Transition rates differ by more than 10%!")
    else
        println("OK: Transition rates match")
    end
end

println("\n" * "="^60)
println("DONE")
println("="^60)
