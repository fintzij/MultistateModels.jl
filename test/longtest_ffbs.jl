using MultistateModels
using MultistateModels: flatview, MPanelData, loglik
using Distributions
using Plots

include("test/setup_ffbs.jl")


#
# verify that emat is correctly formatted


#
# Sampling when there is no censoring

# true trajectory
subj_dat_no_censoring.stateto

# emission matrix
subj_emat_no_censoring

# sampled trajectory
m, p = ForwardFiltering(subj_dat_no_censoring, tpm_book, subj_tpm_map, subj_emat_no_censoring) 
h = BackwardSampling(m, p) 

# verify match
all(subj_dat_no_censoring.stateto .== h)


#
# Impossible censoring patterns
subj_emat_impossible # impossible to have a transition from states (3,4) to states (1,2) at step 5 in a progressive model
sample_skeleton!(subj_dat_impossible, tpm_book, subj_tpm_map, subj_emat_impossible)


#
# Comparing Monte Carlo estimates and analytical probabilities

# Monte Carlo
M = 10000000
paths = Array{Int64}(undef, M, 5)
m, p = ForwardFiltering(subj_dat_analytical, tpm_book, subj_tpm_map, subj_emat_analytical)

paths = map(x -> BackwardSampling(m,p), collect(1:M))

paths_id = map(x -> sum(x), paths)
MC_estimates = [sum(paths_id.==11) sum(paths_id.==12) sum(paths_id.==13) sum(paths_id.==14)] ./ M
sum(MC_estimates)

# analytical
subj_emat_analytical
possible_states = (2,3)
prob_analytical = DataFrame(state_2 = Int64[], state_3 = Int64[], state_4 = Int64[], likelihood = Float64[])
dat.obstype = fill(2, n_obs) # panel-observed data


for state_2 in possible_states, state_3 in possible_states, state_4 in possible_states

    dat.stateto[2:4] = [state_2, state_3, state_4]
    dat.statefrom[3:5] = [state_2, state_3, state_4]
    model = multistatemodel(h12, h23, h34; data = dat, censoring_patterns = censoring_patterns)
    books = build_tpm_mapping(model.data)
    data = MPanelData(model, books)

    likelihood = exp(loglik(parameters, data::MPanelData; neg = false))

    push!(prob_analytical, [state_2 state_3 state_4 likelihood])

end
prob_analytical.prob = prob_analytical.likelihood / sum(prob_analytical.likelihood)

# comparison of MC estimates and analytical answer

relative_error = abs.((prob_analytical.prob[[1,2,4,8]] .- MC_estimates[1,:]) ./ prob_analytical.prob[[1,2,4,8]])
relative_error