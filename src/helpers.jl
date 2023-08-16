"""
    initialize_parameters!(model::MultistateProcess)

Set initial values for the model parameters. Description of how this happens...
"""
function initialize_parameters!(model::MultistateProcess)
    transmat = statetable(model)
    q_crude_init = crudeinit(transmat, model.tmat)
    # set_parameters!(model, q_crude_init)
end

"""
    set_parameters!(model::MultistateProcess, newvalues::Vector{Float64})

Set model parameters given a vector of values. Copies `newvalues`` to `model.parameters`.
"""
function set_parameters!(model::MultistateProcess, newvalues::Union{VectorOfVectors,Vector{Vector{Float64}}})
    
    # check that we have the right number of parameters
    if(length(model.parameters) != length(newvalues))
        error("New values and model parameters are not of the same length.")
    end

    for i in eachindex(model.parameters)
        if(length(model.parameters[i]) != length(newvalues[i]))
            error("New values for hazard $i and model parameters for that hazard are not of the same length.")
        end
        copyto!(model.parameters[i], newvalues[i])
    end
end

"""
    set_parameters!(model::MultistateProcess, newvalues::Tuple)

Set model parameters given a tuple of vectors parameterizing cause-specific hazards. Assigns new values to `model.parameters[i]`, where `i` indexes the cause-specific hazards in the order they appear in the model object.
"""
function set_parameters!(model::MultistateProcess, newvalues::Tuple)
    # check that there is a vector of parameters for each cause-specific hazard
    if(length(model.parameters) != length(newvalues))
        error("Number of supplied parameter vectors not equal to number of cause-specific hazards.")
    end

    for i in eachindex(newvalues)
        # check that we have the right number of parameters
        if(length(model.parameters[i]) != length(newvalues[i]))
            error("New values and parameters for cause-specific hazard $i are not of the same length.")
        end

        copyto!(model.parameters[i], newvalues[i])                   
    end
end

"""
    set_parameters!(model::MultistateProcess, newvalues::NamedTuple)

Set model parameters given a tuple of vectors parameterizing cause-specific hazards. Assignment is made by matching tuple keys in `newvalues` to the key in `model.hazkeys`.  
"""
function set_parameters!(model::MultistateProcess, newvalues::NamedTuple)
    
    # get keys for the new values
    value_keys = keys(newvalues)

    for i in eachindex(value_keys)

        # check length of supplied parameters
        if length(newvalues[value_keys[i]]) != 
                length(model.parameters[model.hazkeys[value_keys[i]]])
            error("The new parameter values for $value_keys[i] are not the expected length.")
        end

        copyto!(model.parameters[model.hazkeys[value_keys[i]]], newvalues[value_keys[i]])
    end
end

"""
    get_subjinds(data::DataFrame)

Return a vector with the row indices for each subject in the dataset.
"""
function get_subjinds(data::DataFrame)

    # number of subjects
    ids = unique(data.id)
    nsubj = length(ids)

    # initialize vector of indices
    subjinds = Vector{Vector{Int64}}(undef, nsubj)

    # get indices for each subject
    for i in eachindex(ids)
        subjinds[i] = findall(x -> x == ids[i], data.id)
    end

    # return indices
    return subjinds, nsubj
end

"""
    check_data!(data::DataFrame)

Validate a user-supplied data frame to ensure that it conforms to MultistateModels.jl requirements.
"""
function check_data!(data::DataFrame, tmat::Matrix, CensoringPatterns::Matrix{Int64})

    # validate column names and order
    if any(names(data)[1:6] .!== ["id", "tstart", "tstop", "statefrom", "stateto", "obstype"])
        error("The first 6 columns of the data should be 'id', 'tstart', 'tstop', 'statefrom', 'stateto', 'obstype'.")
    end

    # coerce id to Int64, times to Float64, states to Int64, obstype to Int64
    data.id        = convert(Vector{Int64},   data.id)
    data.tstart    = convert(Vector{Float64}, data.tstart)
    data.tstop     = convert(Vector{Float64}, data.tstop)
    data.obstype   = convert(Vector{Int64},   data.obstype)
    data.statefrom = convert(Vector{Union{Missing,Int64}}, data.statefrom)
    data.stateto   = convert(Vector{Union{Missing, Int64}}, data.stateto)        

    # verify that subject id's are (1, 2, ...)
    unique_id = unique(data.id)
    nsubj = length(unique_id)
    if any(unique_id .!= 1:nsubj)
        error("The subject id's should be 1, 2, 3, ... .")
    end

    # warn about individuals starting in absorbing states
    # check if there are any absorbing states
    absorbing = map(x -> all(x .== 0), eachrow(tmat))

    # look to see if any of the absorbing states are in statefrom
    if any(absorbing)
        which_absorbing = findall(absorbing .== true)
        abs_warn = map(x -> any(data.statefrom .== x), which_absorbing)

        if any(abs_warn)
            @warn "The data contains contains observations where a subject originates in an absorbing state."
        end
    end

    # error if any tstart < tstop
    if any(data.tstart >= data.tstop)
        error("The data should not contain time intervals where tstart is greater than or equal to tstop.")
    end

    # within each subject's data, error if tstart or tstop are out of order or there are discontinuities given multiple time intervals
    for i in unique_id
        inds = findall(data.id .== i)

        # check sorting
        if(!issorted(data.tstart[inds]) || !issorted(data.tstop[inds])) 
            error("tstart and tstop must be sorted for each subject.")
        end
        
        # check for discontinuities
        if(length(inds) > 1)
            if(any(data.tstart[inds[Not(begin)]] .!= 
                    data.tstop[inds[Not(end)]]))
                error("Time intervals for subject $i contain discontinuities.")
            end
        end
    end

    # warning if tmat specifies an allowed transition for which no such transitions were observed in the data
    n_rs = compute_number_transitions(data, tmat)
    for r in 1:size(tmat)[1]
        for s in 1:size(tmat)[2]
            if tmat[r,s]!=0 && n_rs[r,s]==0
                @warn "Data does not contain any transitions from state $r to state $s"
            end
        end
    end

    # check that obstype is one of the allowed censoring schemes
    if any(data.obstype .∉ Ref([1,2]))
        CensoringPatterns_id = CensoringPatterns[:,1]
        if any(data.obstype .∉ Ref([[1,2]; CensoringPatterns_id]))
            error("obstype should be one of 1, 2, or a censoring id from CensoringPatterns.")
        end
    end

    # check that stateto is 0 when obstype is not 1 or 2
    for i in Base.OneTo(nrow(data))
        if (data.obstype[i] > 2) & (data.stateto[i] .!= 0)            
            error("When obstype>2, stateto should be 0.")
        end
    end

    # check that subjects start in an observed state (statefrom!=0)
    for subj in Base.OneTo(nsubj)
        datasubj = filter(:id => ==(subj), data)
        if datasubj.statefrom[1] == 0          
            error("Subject $subj should not start in state 0.")
        end
    end

    # check that there is no row for a subject after they hit an absorbing state

end

function check_SamplingWeights(SamplingWeights::Vector{Float64}, data::DataFrame)
    
    # check that the number of sampling weights is correct
    if length(SamplingWeights) != length(unique(data.id))
        error("The length of SamplingWeights is not equal to the number of subjects.")
    end

    # check that the sampling weights are non-negative
    if any(SamplingWeights .<= 0)
        error("The elements of SamplingWeights should be non-negative.")
    end
end
"""
check_CensoringPatterns(data::DataFrame, emat::Matrix)

Validate a user-supplied data frame to ensure that it conforms to MultistateModels.jl requirements.
"""
function check_CensoringPatterns(CensoringPatterns::Matrix{Int64}, tmat::Matrix)
    
    nrow, ncol = size(CensoringPatterns)

    # check for empty
    if nrow == 0 | ncol < 2
        error("The matrix CensoringPatterns seems to be empty, while there are censored states.")
    end

    # censoring patterns must be labelled as 3, 4, ...
    if !all(CensoringPatterns[:,1] .== 3:(nrow+2))
        error("The first column of the matrix `CensoringPatterns` must be of the form (3, 4, ...) .")
    end

    # censoring patterns must be binary
    if any(CensoringPatterns[:,2:ncol] .∉ Ref([0,1]))
        error("Columns 2, 3, ... of CensoringPatterns must be binary.")
    end

    # censoring patterns must indicate the presence/absence of each state
    n_states = size(tmat, 1)
    if ncol - 1 .!= n_states
        error("The multistate model contains $n_states states, but CensoringPatterns contains $(ncol-1) states.")
    end

    # censoring patterns must have at least one possible state
    for i in 1:nrow
        if all(CensoringPatterns[i,2:ncol] .== 0)
            error("Censoring pattern $i has no allowed state.")
        end
        if all(CensoringPatterns[i,2:ncol] .== 1)
            println("All states are allowed in censoring pattern $(2+i).")
        end
        if sum(CensoringPatterns[i,2:ncol]) .== 1
            println("Censoring pattern $i has only one allowed state; if these observations are not censored there is no need to use a censoring pattern.")
        end
    end
end

"""
    build_tpm_book(tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})

Build container for holding transition probability matrices.
"""
function build_tpm_book(T::DataType, tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})

    # build the TPM container
    nstates = size(tmat, 1)
    nmats   = map(x -> nrow(x), tpm_index) 
    book    = [[zeros(T, nstates, nstates) for j in 1:nmats[i]] for i in eachindex(tpm_index)]

    return book
end

"""
    build_hazmat_book(tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})

Build container for holding transition intensity matrices.
"""
function build_hazmat_book(T::DataType, tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})
    # Making this "type aware" by using T::DataType so that autodiff worksA
    # build the TPM container
    nstates = size(tmat, 1)
    nmats   = map(x -> nrow(x), tpm_index) 
    book    = [zeros(T, nstates, nstates) for j in eachindex(tpm_index)]

    return book
end

"""
    build_tpm_mapping(data::DataFrame)

Construct bookkeeping objects for transition probability matrices for time intervals over which a multistate Markov process is piecewise homogeneous. The first bookkeeping object is a data frame that 
"""
function build_tpm_mapping(data::DataFrame) 

    # maps each row in dataset to TPM
    # first col is covar combn, second is tpm index
    tpm_map = zeros(Int64, nrow(data), 2)

    # check if the data contains covariates
    if ncol(data) == 6 # no covariates
        
        # get intervals
        gaps = data.tstop - data.tstart

        # get unique start and stop
        ugaps = sort(unique(gaps))

        # for solving Kolmogorov equations - saveats
        tpm_index = 
            [DataFrame(tstart = 0,
                       tstop  = ugaps,
                       datind = 0),]

        # first instance of each interval in the data
        for i in Base.OneTo(nrow(tpm_index[1]))
            tpm_index[1].datind[i] = 
                findfirst(gaps .== tpm_index[1].tstop[i])
        end

        # match intervals to unique tpms
        tpm_map[:,1] .= 1
        for i in Base.OneTo(size(tpm_map, 1))
            tpm_map[i,2] = findfirst(ugaps .== gaps[i])
        end    

    else
        # get unique covariates
        covars = data[:,Not(1:6)]
        ucovars = unique(data[:,Not(1:6)])

        # get gap times
        gaps = data.tstop - data.tstart

        # initialize tpm_index
        tpm_index = [DataFrame() for i in 1:nrow(ucovars)]

        # for each set of unique covariates find gaps
        for k in Base.OneTo(nrow(ucovars))

            # get indices for rows that have the covars
            covinds = findall(map(x -> all(x == ucovars[k,:]), eachrow(covars)) .== 1)

            # find unique gaps 
            ugaps = sort(unique(gaps[covinds]))

            # fill in tpm_index
            tpm_index[k] = DataFrame(tstart = 0, tstop = ugaps, datind = 0)

            # first instance of each interval in the data
            for i in Base.OneTo(nrow(tpm_index[k]))
                tpm_index[k].datind[i] = 
                    covinds[findfirst(gaps[covinds] .== tpm_index[k].tstop[i])]
            end

            # fill out the tpm_map 
            # match intervals to unique tpms
            tpm_map[covinds, 1] .= k
            for i in eachindex(covinds)
                tpm_map[covinds[i],2] = findfirst(ugaps .== gaps[covinds[i]])
            end  
        end
    end

    # return objects
    return tpm_index, tpm_map
end




"""
    loglik(model::MultistateModelFitted) 

Return the maximum likelihood estimates. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
"""
function loglik(model::MultistateModelFitted) 
    model.loglik
end

"""
    parameters(model::MultistateModelFitted; transformed::Bool = true) 

Return the maximum likelihood estimates. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
"""
function parameters(model::MultistateModelFitted)
    model.parameters
#    reduce(vcat, model.parameters)
end


"""
    vcov(model::MultistateModelFitted) 

Return the variance covariance matrix at the maximum likelihood estimate. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
"""
function vcov(model::MultistateModelFitted) 
    model.vcov
end


"""
    optim(model::MultistateModelFitted) 

Return the variance covariance matrix at the maximum likelihood estimate. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
"""
function optim(model::MultistateModelFitted) 
    model.optim
end

"""
    summary(model::MultistateModelFitted) 

Return the variance covariance matrix at the maximum likelihood estimate. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
- `confidence_level::Float64`: confidence level of the confidence intervals
"""
function summary(model::MultistateModelFitted; confidence_level::Float64 = 0.95) 

    #
    # summary table
    
    # maximum likelihood estimates
    mle=parameters(model)
    
    # standard error
    vcov = MultistateModels.vcov(model)
    se = sqrt.(vcov[diagind(vcov)])
    se_vv = VectorOfVectors(se, model.parameters.elem_ptr)

    # name of hazards
    haznames = map(x -> model.hazards[x].hazname, collect(1:length(model.hazards)))

    # table
    summary_table = Vector{DataFrame}(undef, length(haznames))
    z_critical = quantile(Normal(0.0, 1.0), 1-(1-confidence_level)/2)
    for s in eachindex(summary_table)
        # summary for hazard s
        summary_table[s] = DataFrame(
            estimate = reduce(vcat, mle[s]),
            se = reduce(vcat, se_vv[s]))
        summary_table[s].upper = summary_table[s].estimate .+ z_critical .* summary_table[s].se
        summary_table[s].lower = summary_table[s].estimate .- z_critical .* summary_table[s].se
    end

    # add hazard names to the table
    summary_table = (;zip(haznames, summary_table)...)
    
    #
    # log likelihood
    ll = loglik(model)

    #
    # information criteria
    p = length(reduce(vcat, mle))
    n = nrow(model.data)
    AIC = -2*ll + 2     *p
    BIC = -2*ll + log(n)*p

    return summary_table, ll, AIC, BIC
end