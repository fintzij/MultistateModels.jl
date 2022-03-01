"""
    set_parameters!(model::MultistateModel, values::Vector{Float64})

Set model parameters given a vector of values. Assigns values to `model.parameters`, which are then propagated to subarrays in cause-specific hazards.
"""
function set_parameters!(model::MultistateModel, values::Vector{Float64})
    
    # check that we have the right number of parameters
    if(length(model.parameters) != length(values))
        error("Values and model parameters are not of the same length.")
    end

    copyto!(model.parameters, values)
end

"""
    set_parameters!(model::MultistateModel, Tuple{Vararg{Vector{Float64}}})

Set model parameters given a tuple of vectors parameterizing cause-specific hazards. Assigns values to `model.hazards[i].parameters`, where `i` indexes the cause-specific hazards in the order they appear in the model object. Parameters for cause-specific hazards are automatically propagated to `model.parameters`.
"""
function set_parameters!(model::MultistateModel, values::Tuple{Vararg{Vector{Float64}}})
    # check that there is a vector of parameters for each cause-specific hazard
    if(length(model.hazards) != length(values))
        error("Number of supplied parameter vectors not equal to number of cause-specific hazards.")
    end

    for i in eachindex(values)
        # check that we have the right number of parameters
        if(length(model.hazards[i].parameters) != length(values[i]))
            error("Values and parameters for cause-specific hazard $i are not of the same length.")
        end

        copyto!(model.hazards[i].parameters, values[i])
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