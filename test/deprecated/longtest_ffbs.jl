include("setup_ffbs.jl")



# Tests for the ffbs sampling algorithm
# 1. Check that the sampler returns teh correct path when there is a unique path that is possible
# 2. Verify that the MC estimates of the path probabilities based on the FFBS agree with the numerical estimates
# 3. Confirm that the function SampleSkeleton! returns an error message when there is no path possible

# test that the FFBS algorithm sinds the correct path when there is only one possible.
@testset "ffbs_degenerate" begin

    # sampled trajectory
    m, p = ForwardFiltering(model_degenerate.data, tpm_book_degenerate, books_degenerate[2], model_degenerate.emat) 
    h = BackwardSampling(m, p) 

    # verify match
    true_trajectory = [1,1,2,3,3] 
    @test all(true_trajectory .== h)
end



#
# Test that the Monte Carlo estimates from the FFBS and the numerical estimate of the path probabilities agree
@testset "ffbs_degenerate" begin

    # Monte Carlo estimates of the path probabilities based on the FFBS algorithm
    M = 10^6
    paths = Array{Int64}(undef, M, 5)
    m, p = ForwardFiltering(model_MC.data, tpm_book_MC, books_MC[2], model_MC.emat) 

    paths = map(x -> BackwardSampling(m,p), collect(1:M))

    paths_id = map(x -> sum(x), paths)
    estimates_MC = [sum(paths_id.==11), sum(paths_id.==12), sum(paths_id.==13), sum(paths_id.==14)] ./ M


    # numerical estimates of the path probabilities from matrix eponentiation
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
        data_loop = MPanelData(model_loop, books_MC)
        likelihood = exp(loglik(flatview(model_loop.parameters), data_loop; neg = false))

        # save results
        push!(estimates_numerical, [state_2 state_3 state_4 likelihood])
    end
    estimates_numerical.prob = estimates_numerical.likelihood / sum(estimates_numerical.likelihood)

    # comparison of MC and numerical estimates and analytical agree
    relative_error = abs.((estimates_numerical.prob[[1,2,4,8]] .- estimates_MC) ./ estimates_numerical.prob[[1,2,4,8]])
    @test all(relative_error .< 0.01)
end


#
# Impossible censoring patterns
@testset "ffbs_impossible" begin

    # We have a progressive model 1->2->3->4
    model_impossible.hazards

    # but at step 5, we have a transition from states (3,4) to states (1,2)
    model_impossible.emat

    # the following line should return an error message
    @test_logs (:error, "test error.")  SampleSkeleton!(model_impossible.data, tpm_book_impossible, books_impossible[2], model_impossible.emat)
end