using MultistateModels
using MultistateModels: flatview, MPanelData, loglik, SampleSkeleton!
using Distributions
using Plots

include("setup_ffbs.jl")


#
# Censoring with degenerate censoring patterns

# sampled trajectory
m, p = ForwardFiltering(model_degenerate.data, tpm_book_degenerate, books_degenerate[2], model_degenerate.emat) 
h = BackwardSampling(m, p) 

# verify match
true_trajectory = [1,1,2,3,3] 
all(true_trajectory .== h)


#
# Comparing Monte Carlo estimates and analytical probabilities

# Monte Carlo estimates of the path probabilities based on the FFBS algorithm
M = 10^6
paths = Array{Int64}(undef, M, 5)
m, p = ForwardFiltering(model_MC.data, tpm_book_MC, books_MC[2], model_MC.emat) 

paths = map(x -> BackwardSampling(m,p), collect(1:M))

paths_id = map(x -> sum(x), paths)
estimates_MC = [sum(paths_id.==11), sum(paths_id.==12), sum(paths_id.==13), sum(paths_id.==14)] ./ M

# numerical expression of the path probabilities from matrix eponentiation
estimates_numerical = DataFrame(state_2 = Int64[], state_3 = Int64[], state_4 = Int64[], likelihood = Float64[])
possible_states = (2,3)

for state_2 in possible_states, state_3 in possible_states, state_4 in possible_states
    # impute the data
    dat.obstype = fill(2, n_obs)
    dat.statefrom = [1, 1, state_2, state_3, state_4]
    dat.stateto = [1, state_2, state_3, state_4, 4]

    # set up model
    model_loop = multistatemodel(h12, h23, h34; data = dat, CensoringPatterns = CensoringPatterns_MC);
    set_parameters!(model_loop, pars)

    # compute likelihood
    books_loop = build_tpm_mapping(model_loop.data)
    data_loop = MPanelData(model_loop, books)
    likelihood = exp(loglik(flatview(model_loop.parameters), data_loop; neg = false))

    # save results
    push!(estimates_numerical, [state_2 state_3 state_4 likelihood])
end
estimates_numerical.prob = estimates_numerical.likelihood / sum(estimates_numerical.likelihood)

# comparison of MC and numerical estimates and analytical answer
relative_error = abs.((estimates_numerical.prob[[1,2,4,8]] .- estimates_MC) ./ estimates_numerical.prob[[1,2,4,8]])
all(relative_error .< 0.01)


#
# Impossible censoring patterns
model_impossible.emat # impossible to have a transition from states (3,4) to states (1,2) at step 5 in a progressive model
SampleSkeleton!(model_impossible.data, tpm_book_impossible, books_impossible[2], model_impossible.emat)
