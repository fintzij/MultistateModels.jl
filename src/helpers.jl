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
function set_parameters!(model::MultistateProcess, newvalues::Vector{Vector{Float64}})
    
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
    return subjinds
end

"""
    check_data!(data::DataFrame)

Validate a user-supplied data frame to ensure that it conforms to MultistateModels.jl requirements.
"""
function check_data!(data::DataFrame, tmat::Matrix)

    # validate column names and order
    if any(names(data)[1:6] .!== ["id", "tstart", "tstop", "statefrom", "stateto", "obstype"])
        error("The first 6 columns of the data should be 'id', 'tstart', 'tstop', 'statefrom', 'stateto', 'obstype'.")
    end

    # coerce id to Int64, times to Float64, states to Int64, obstype to Int64
    data.id        = convert(Vector{Int64},   data.id)
    data.tstart    = convert(Vector{Float64}, data.tstart)
    data.tstop     = convert(Vector{Float64}, data.tstop)
    data.obstype   = convert(Vector{Int64},   data.obstype)
    data.statefrom = 
        convert(Vector{Union{Missing,Int64}}, data.statefrom)
    data.stateto   = 
        convert(Vector{Union{Missing, Int64}}, data.stateto)

    # warn about individuals starting in absorbing states
    # check if there are any absorbing states
    absorbing = map(x -> all(x .== 0), eachrow(tmat))

    # look to see if any of the absorbing states are in statefrom
    if any(absorbing)
        which_absorbing = findall(absorbing .== true)
        abs_warn = map(x -> any(data.statefrom .== x), which_absorbing)

        if any(abs_warn)
            println("The data contains contains observations where a subject originates in an absorbing state.")
        end
    end

    # error if any tstart < tstop
    if any(data.tstart >= data.tstop)
        error("The data should not contain time intervals where tstart is greater than or equal to tstop.")
    end

    # within each subject's data, error if tstart or tstop are out of order or there are discontinuities given multiple time intervals
    for i in unique(data.id)
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

    # check that obstype is one of the allowed censoring schemes

    # check that there are no rows for a subject after they hit an absorbing state
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