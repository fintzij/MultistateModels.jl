"""
    set_parameters!(model::MultistateModel, newvalues::Vector{Float64})

Set model parameters given a vector of values. Assigns `newvalues`` to `model.parameters`, which are then propagated to subarrays in cause-specific hazards.
"""
function set_parameters!(model::MultistateModel, newvalues::Vector{Float64})
    
    # check that we have the right number of parameters
    if(length(model.parameters) != length(newvalues))
        error("New values and model parameters are not of the same length.")
    end

    copyto!(model.parameters, newvalues)
end

"""
    set_parameters!(model::MultistateModel, newvalues::Tuple{Vararg{Vector{Float64}}})

Set model parameters given a tuple of vectors parameterizing cause-specific hazards. Assigns new values to `model.hazards[i].parameters`, where `i` indexes the cause-specific hazards in the order they appear in the model object. Parameters for cause-specific hazards are automatically propagated to `model.parameters`.
"""
function set_parameters!(model::MultistateModel, newvalues::Tuple{Vararg{Vector{Float64}}})
    # check that there is a vector of parameters for each cause-specific hazard
    if(length(model.hazards) != length(newvalues))
        error("Number of supplied parameter vectors not equal to number of cause-specific hazards.")
    end

    for i in eachindex(newvalues)
        # check that we have the right number of parameters
        if(length(model.hazards[i].parameters) != length(newvalues[i]))
            error("New values and parameters for cause-specific hazard $i are not of the same length.")
        end

        copyto!(model.hazards[i].parameters, newvalues[i])
    end
end

"""
    set_parameters!(model::MultistateModels.MultistateModel, newvalues::NamedTuple{<:Any, <:Tuple{Vararg{Vector{Float64}}}})

Set model parameters given a tuple of vectors parameterizing cause-specific hazards. Assignment is made by matching tuple keys in `newvalues` to the key in `model.hazkeys`.  Parameters for cause-specific hazards are automatically propagated to `model.parameters`.
"""
function set_parameters!(model::MultistateModels.MultistateModel, newvalues::NamedTuple{<:Any, <:Tuple{Vararg{Vector{Float64}}}})
    
    # get keys for the new values
    value_keys = keys(newvalues)

    for i in value_keys

        # check length of supplied parameters
        if length(newvalues[i]) != length(model.hazards[model.hazkeys[i]].parameters)
            error("The new parameter values for $i are not the expected length.")
        end

        copyto!(model.hazards[i].parameters, newvalues[i])
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
    data.statefrom = convert(Vector{Int64},   data.statefrom)
    data.stateto   = convert(Vector{Int64},   data.stateto)
    data.obstype   = convert(Vector{Int64},   data.obstype)

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
end
