########################################################
### Likelihod functions for sample paths, tpms, etc. ###
########################################################

### Exactly observed sample paths ----------------------
"""
    loglik_path(pars, subjectdata::DataFrame, hazards::Vector{<:_Hazard}, totalhazards::Vector{<:_TotalHazard}, tmat::Array{Int,2})

Log-likelihood for a single sample path. The subject data is provided as a DataFrame with columns including:
- `sojourn`: Time spent in current state at start of interval
- `increment`: Time increment for this interval
- `statefrom`: State at start of interval
- `stateto`: State at end of interval
- Additional covariate columns

This function is called after converting a SamplePath object to DataFrame format using `make_subjdat()`.
"""
loglik_path = function(pars, subjectdata::DataFrame, hazards::Vector{<:_Hazard}, totalhazards::Vector{<:_TotalHazard}, tmat::Array{Int,2})

     # initialize log likelihood
     ll = 0.0
 
     # recurse through the sample path
     for i in Base.OneTo(nrow(subjectdata))

        # accumulate survival probabilty
        ll += survprob(subjectdata.sojourn[i], subjectdata.sojourn[i] + subjectdata.increment[i], pars, subjectdata[i, :], totalhazards[subjectdata.statefrom[i]], hazards; give_log = true)
 
        # accumulate hazard if there is a transition
        if subjectdata.statefrom[i] != subjectdata.stateto[i]
            
            # index for transition
            transind = tmat[subjectdata.statefrom[i], subjectdata.stateto[i]]

            # log hazard at time of transition
            # Pass the DataFrame row for new hazard types (with name-based covariate matching)
            ll += call_haz(subjectdata.sojourn[i] + subjectdata.increment[i], pars[transind], subjectdata[i, :], hazards[transind]; give_log = true)
        end
     end
 
     # unweighted loglikelihood
     return ll
end

########################################################
##################### Wrappers #########################
########################################################

"""
    loglik(parameters, data::ExactData; neg = true) 

Return sum of (negative) log likelihoods for all sample paths. Use mapreduce() to call loglik() and sum the results. Each sample path object is `path::SamplePath` and contains the subject index and the jump chain. 
"""
function loglik_exact(parameters, data::ExactData; neg=true, return_ll_subj=false)

    # nest parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # snag the hazards
    hazards = data.model.hazards

    # remake spline parameters and calculate risk periods
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
            set_riskperiod!(hazards[i])
        end
    end

    if return_ll_subj
        # send each element of samplepaths to loglik
        # Convert SamplePath to DataFrame using make_subjdat
        map((path, w) -> begin
            subj_inds = data.model.subjectindices[path.subj]
            subj_dat = view(data.model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat) * w
        end, data.paths, data.model.SamplingWeights)
    else
        # Convert SamplePath to DataFrame using make_subjdat
        ll = mapreduce((path, w) -> begin
            subj_inds = data.model.subjectindices[path.subj]
            subj_dat = view(data.model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat) * w
        end, +, data.paths, data.model.SamplingWeights)
        neg ? -ll : ll
    end
end

"""
    loglik(parameters, data::ExactDataAD; neg = true) 

Return sum of (negative) log likelihoods for all sample paths. Use mapreduce() to call loglik() and sum the results. Each sample path object is `path::SamplePath` and contains the subject index and the jump chain. NOTE: Why is this different from loglik_exact?
"""
function loglik_AD(parameters, data::ExactDataAD; neg = true)

    # nest parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # snag the hazards
    hazards = data.model.hazards

    # remake spline parameters and calculate risk periods
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
            set_riskperiod!(hazards[i])
        end
    end

    # send each element of samplepaths to loglik
    # Convert SamplePath to DataFrame using make_subjdat
    path = data.path[1]
    subj_inds = data.model.subjectindices[path.subj]
    subj_dat = view(data.model.data, subj_inds, :)
    subjdat_df = make_subjdat(path, subj_dat)
    ll = loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat) * data.samplingweight[1]

    neg ? -ll : ll
end

"""
    loglik(parameters, data::MPanelData; neg = true)

Return sum of (negative) log likelihood for a Markov model fit to panel and/or exact and/or censored data. 
"""
function loglik_markov(parameters, data::MPanelData; neg = true, return_ll_subj = false)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # build containers for transition intensity and prob mtcs
    hazmat_book = build_hazmat_book(eltype(parameters), data.model.tmat, data.books[1])
    tpm_book = build_tpm_book(eltype(parameters), data.model.tmat, data.books[1])

    # allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    for t in eachindex(data.books[1])

        # compute the transition intensity matrix
        compute_hazmat!(
            hazmat_book[t],
            pars,
            data.model.hazards,
            data.books[1][t],
            data.model.data)

        # compute transition probability matrices
        compute_tmat!(
            tpm_book[t],
            hazmat_book[t],
            data.books[1][t],
            cache)
    end

    # number of subjects
    nsubj = length(data.model.subjectindices)

    # accumulate the log likelihood
    ll = 0.0

    # container for subject-level loglikelihood
    if return_ll_subj
        ll_subj = zeros(Float64, nsubj)
    end

    # number of states
    S = size(data.model.tmat, 1)

    # initialize Q 
    q = zeros(eltype(parameters), S, S)

    # for each subject, compute the likelihood contribution
    for subj in Base.OneTo(nsubj)

        # subject data
        subj_inds = data.model.subjectindices[subj]

        # no state is censored
        if all(data.model.data.obstype[subj_inds] .∈ Ref([1,2]))
            
            # subject contribution to the loglikelihood
            subj_ll = 0.0

            # add the contribution of each observation
            for i in subj_inds
                if data.model.data.obstype[i] == 1 # exact data
                    
                    subj_ll += survprob(0, data.model.data.tstop[i] - data.model.data.tstart[i], pars, i, data.model.totalhazards[data.model.data.statefrom[i]], data.model.hazards; give_log = true)
                                        
                    if data.model.data.statefrom[i] != data.model.data.stateto[i] # if there is a transition, add log hazard
                        subj_ll += call_haz(data.model.data.tstop[i] - data.model.data.tstart[i], pars[data.model.tmat[data.model.data.statefrom[i], data.model.data.stateto[i]]], i, data.model.hazards[data.model.tmat[data.model.data.statefrom[i], data.model.data.stateto[i]]]; give_log = true)
                    end

                else # panel data
                    subj_ll += log(tpm_book[data.books[2][i, 1]][data.books[2][i, 2]][data.model.data.statefrom[i], data.model.data.stateto[i]])
                end
            end

        else
            # initialize likelihood matrix
            lmat = zeros(eltype(parameters), S, length(subj_inds) + 1)
            lmat[data.model.data.statefrom[subj_inds[1]], 1] = 1

            # initialize counter for likelihood matrix
            ind = 1

            # update the vector l
            for i in subj_inds

                # increment counter for likelihood matrix
                ind += 1

                # compute q, the transition probability matrix
                if data.model.data.obstype[i] != 1
                    # if panel data, simply grab q from tpm_book
                    copyto!(q, tpm_book[data.books[2][i, 1]][data.books[2][i, 2]])
                    
                else
                    # if exact data (obstype = 1), compute q by hand
                    # reset Q
                    fill!(q, -Inf)
                    
                    # compute q(r,s)
                    for r in 1:S
                        if isa(data.model.totalhazards[r], _TotalHazardAbsorbing)
                            q[r,r] = 0.0
                        else
                            # survival probability
                            q[r, findall(data.model.tmat[r,:] .!= 0)] .= survprob(0, data.model.data.tstop[i] - data.model.data.tstart[i], pars, i, data.model.totalhazards[r], data.model.hazards; give_log = true) 
                            
                            # hazard
                            for s in 1:S
                                if (s != r) & (data.model.tmat[r,s] != 0)
                                    q[r, s] += call_haz(data.model.data.tstop[i] - data.model.data.tstart[i], pars[data.model.tmat[r, s]], i, data.model.hazards[data.model.tmat[r, s]]; give_log = true)
                                end
                            end
                        end

                        # pedantic b/c of numerical error
                        q[r,r] = maximum([1 - exp(logsumexp(q[r, Not(r)])), eps()])
                        q[r,Not(r)] = exp.(q[r, Not(r)])               
                    end
                end # end-compute q

                # compute the set of possible "states to"
                StatesTo = data.model.data.stateto[i] > 0 ? [data.model.data.stateto[i]] : findall(data.model.emat[i,:] .== 1)

                for s in 1:S
                    if s ∈ StatesTo
                        for r in 1:S
                            lmat[s, ind] += q[r,s] * lmat[r, ind - 1]
                        end
                    end
                end
            end

            # log likelihood
            subj_ll=log(sum(lmat[:,size(lmat, 2)]))
        end

        if return_ll_subj
            # weighted subject loglikelihood
            ll_subj[subj] = subj_ll * data.model.SamplingWeights[subj]
        else
            # weighted loglikelihood
            ll += subj_ll * data.model.SamplingWeights[subj]
        end        
    end

    if return_ll_subj
        ll_subj
    else
        neg ? -ll : ll
    end
end

"""
    loglik(parameters, data::SMPanelData; neg = true)

Return sum of (negative) complete data log-likelihood terms in the Monte Carlo maximum likelihood algorithm for fitting a semi-Markov model to panel data. 
"""
function loglik_semi_markov(parameters, data::SMPanelData; neg = true, use_sampling_weight = true)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # snag the hazards
    hazards = data.model.hazards

    # remake spline parameters and calculate risk periods
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
            set_riskperiod!(hazards[i])
        end
    end

    # compute the semi-markov log-likelihoods
    ll = 0.0
    for i in eachindex(data.paths)
        lls = 0.0
        for j in eachindex(data.paths[i])
            # mlm: function Q in the EM
            # Convert SamplePath to DataFrame using make_subjdat
            path = data.paths[i][j]
            subj_inds = data.model.subjectindices[path.subj]
            subj_dat = view(data.model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            lls += loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat) * data.ImportanceWeights[i][j]
        end
        if use_sampling_weight
            lls *= data.model.SamplingWeights[i]
        end
        ll += lls
    end

    # return the log-likelihood
    neg ? -ll : ll
end

"""
    loglik(parameters, data::SMPanelData)

Update log-likelihood for each individual and each path of panel data in a semi-Markov model.
"""
function loglik_semi_markov!(parameters, logliks::Vector{}, data::SMPanelData)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # snag the hazards
    hazards = data.model.hazards

    # remake spline parameters and calculate risk periods
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
            set_riskperiod!(hazards[i])
        end
    end

    for i in eachindex(data.paths)
        for j in eachindex(data.paths[i])
            # Convert SamplePath to DataFrame using make_subjdat
            path = data.paths[i][j]
            subj_inds = data.model.subjectindices[path.subj]
            subj_dat = view(data.model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            logliks[i][j] = loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat)
        end
    end
end
