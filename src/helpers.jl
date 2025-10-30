"""
    set_parameters!(model::MultistateProcess, newvalues::Vector{Float64})

Set model parameters given a vector of values. Copies `newvalues`` to `model.parameters`.
"""
function set_parameters(model::MultistateProcess, newvalues::Union{VectorOfVectors,Vector{Vector{Float64}}})

    model = deepcopy(model)
    
    # check that we have the right number of parameters
    if(length(model.parameters) != length(newvalues))
        error("New values and model parameters are not of the same length.")
    end

    for i in eachindex(model.parameters)
        if(length(model.parameters[i]) != length(newvalues[i]))
            @error "New values for hazard $i and model parameters for that hazard are not of the same length."
        end
        copyto!(model.parameters[i], newvalues[i])

        # remake if a spline hazard
        if isa(model.hazards[i], _SplineHazard) 
            remake_splines!(model.hazards[i], newvalues[i])
            set_riskperiod!(model.hazards[i])
        end
    end

    return model
end

"""
    set_parameters!(model::MultistateProcess, newvalues::Tuple)

Set model parameters given a tuple of vectors parameterizing transition intensities. Assigns new values to `model.parameters[i]`, where `i` indexes the transition intensities in the order they appear in the model object.
"""
function set_parameters(model::MultistateProcess, newvalues::Tuple)

    model = deepcopy(model)

    # check that there is a vector of parameters for each cause-specific hazard
    if(length(model.parameters) != length(newvalues))
        error("Number of supplied parameter vectors not equal to number of transition intensities.")
    end

    for i in eachindex(newvalues)
        # check that we have the right number of parameters
        if(length(model.parameters[i]) != length(newvalues[i]))
            @error "New values and parameters for cause-specific hazard $i are not of the same length."
        end

        copyto!(model.parameters[i], newvalues[i])    
        
        # remake if a spline hazard
        if isa(model.hazards[i], _SplineHazard)
            remake_splines!(model.hazards[i], newvalues[i])
            set_riskperiod!(model.hazards[i])
        end
    end

    return model
end

"""
    set_parameters!(model::MultistateProcess, newvalues::NamedTuple)

Set model parameters given a tuple of vectors parameterizing transition intensities. Assignment is made by matching tuple keys in `newvalues` to the key in `model.hazkeys`.  
"""
function set_parameters(model::MultistateProcess, newvalues::NamedTuple)

    model = deepcopy(model)
    
    # get keys for the new values
    value_keys = keys(newvalues)

    for k in eachindex(value_keys)

        vind = value_keys[k]
        mind = model.hazkeys[vind]

        # check length of supplied parameters
        if length(newvalues[vind]) != length(model.parameters[mind])
            error("The new parameter values for $vind are not the expected length.")
        end

        copyto!(model.parameters[mind], newvalues[vind])

        # remake if a spline hazard
        if isa(model.hazards[mind], _SplineHazard)
            remake_splines!(model.hazards[mind], newvalues[vind])
            set_riskperiod!(model.hazards[mind])
        end
    end

    return model
end


"""
    set_parameters!(model::MultistateProcess, newvalues::Vector{Float64})

Set model parameters given a vector of values. Copies `newvalues` to `model.parameters`.

# Phase 3 Update: Now automatically updates `model.parameters_ph` to keep both representations synchronized.
"""
function set_parameters!(model::MultistateProcess, newvalues::Union{VectorOfVectors,Vector{Vector{Float64}}})
    
    # check that we have the right number of parameters
    if(length(model.parameters) != length(newvalues))
        error("New values and model parameters are not of the same length.")
    end

    for i in eachindex(model.parameters)
        if(length(model.parameters[i]) != length(newvalues[i]))
            @error "New values for hazard $i and model parameters for that hazard are not of the same length."
        end
        copyto!(model.parameters[i], newvalues[i])

        # remake if a spline hazard
        if isa(model.hazards[i], _SplineHazard) 
            remake_splines!(model.hazards[i], newvalues[i])
            set_riskperiod!(model.hazards[i])
        end
    end
    
    # Phase 3: Update parameters_ph automatically (now that structs are mutable)
    model.parameters_ph = update_parameters_ph!(model)
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, newvalues::Tuple)

Set model parameters given a tuple of vectors parameterizing transition intensities. 
Assigns new values to `model.parameters[i]`, where `i` indexes the transition intensities 
in the order they appear in the model object.

# Phase 3 Update: Now automatically updates `model.parameters_ph` to keep both representations synchronized.
"""
function set_parameters!(model::MultistateProcess, newvalues::Tuple)
    # check that there is a vector of parameters for each cause-specific hazard
    if(length(model.parameters) != length(newvalues))
        error("Number of supplied parameter vectors not equal to number of transition intensities.")
    end

    for i in eachindex(newvalues)
        # check that we have the right number of parameters
        if(length(model.parameters[i]) != length(newvalues[i]))
            @error "New values and parameters for cause-specific hazard $i are not of the same length."
        end

        copyto!(model.parameters[i], newvalues[i])    
        
        # remake if a spline hazard
        if isa(model.hazards[i], _SplineHazard)
            remake_splines!(model.hazards[i], newvalues[i])
            set_riskperiod!(model.hazards[i])
        end
    end
    
    # Phase 3: Update parameters_ph automatically
    model.parameters_ph = update_parameters_ph!(model)
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, newvalues::NamedTuple)

Set model parameters given a tuple of vectors parameterizing transition intensities. 
Assignment is made by matching tuple keys in `newvalues` to the key in `model.hazkeys`.

# Phase 3 Update: Now automatically updates `model.parameters_ph` to keep both representations synchronized.
"""
function set_parameters!(model::MultistateProcess, newvalues::NamedTuple)
    
    # get keys for the new values
    value_keys = keys(newvalues)

    for k in eachindex(value_keys)

        vind = value_keys[k]
        mind = model.hazkeys[vind]

        # check length of supplied parameters
        if length(newvalues[vind]) != length(model.parameters[mind])
            error("The new parameter values for $vind are not the expected length.")
        end

        copyto!(model.parameters[mind], newvalues[vind])

        # remake if a spline hazard
        if isa(model.hazards[mind], _SplineHazard)
            remake_splines!(model.hazards[mind], newvalues[vind])
            set_riskperiod!(model.hazards[mind])
        end
    end
    
    # Phase 3: Update parameters_ph automatically
    model.parameters_ph = update_parameters_ph!(model)
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, h::Int64, newvalues::Vector{Float64})

Set parameters for a single hazard by hazard index.

# Arguments
- `model::MultistateProcess`: The model to update
- `h::Int64`: Index of the hazard to update
- `newvalues::Vector{Float64}`: New parameter values (log scale for baseline/shape/scale, natural for covariates)

# Phase 3: Automatically updates both `parameters` and `parameters_ph` to keep them synchronized.

# Example
```julia
# Update hazard 1 with new baseline parameter
set_parameters!(model, 1, [log(0.5)])

# Update hazard with baseline and covariate effects
set_parameters!(model, 2, [log(2.0), log(1.5), 0.3, -0.2])
```
"""
function set_parameters!(model::MultistateProcess, h::Int64, newvalues::Vector{Float64})
    # Check hazard index
    if h < 1 || h > length(model.parameters)
        error("Hazard index $h is out of bounds. Model has $(length(model.parameters)) hazards.")
    end
    
    # Check parameter length
    if length(newvalues) != length(model.parameters[h])
        error("New values length ($(length(newvalues))) does not match expected length ($(length(model.parameters[h]))) for hazard $h.")
    end
    
    # Update parameters
    copyto!(model.parameters[h], newvalues)
    
    # Remake splines if needed
    if isa(model.hazards[h], _SplineHazard)
        remake_splines!(model.hazards[h], newvalues)
        set_riskperiod!(model.hazards[h])
    end
    
    # Phase 3: Update parameters_ph automatically
    model.parameters_ph = update_parameters_ph!(model)
    
    return nothing
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

#=============================================================================
PHASE 2: Parameter Update with ParameterHandling.jl
=============================================================================# 

"""
    update_parameters_ph!(model::MultistateProcess)

Rebuild the parameters_ph structure from current model.parameters.
Called internally after parameter updates to keep structures synchronized.

Note: Since NamedTuples are immutable, this creates a NEW parameters_ph structure.
The model struct field cannot be updated in-place. This will be addressed in Phase 3
by making model structs mutable or using Ref for parameters_ph.
"""
function update_parameters_ph!(model::MultistateProcess)
    # Rebuild parameters_ph from current VectorOfVectors parameters
    # Note: model.parameters are on log scale, but positive() expects natural scale
    params_transformed_pairs = [
        hazname => ParameterHandling.positive(exp.(Vector{Float64}(model.parameters[idx])))
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    
    # Return new structure (caller needs to update model if needed)
    return (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )
end

#=============================================================================
PHASE 3: Parameter Getter Functions
=============================================================================# 

"""
    get_parameters_flat(model::MultistateProcess)

Get model parameters as a flat vector suitable for optimization.

This is the representation that optimization algorithms expect - a single
flat Vector{Float64} with all parameters concatenated.

# Returns
- `Vector{Float64}`: Flat parameter vector (transformed/log scale)

# Example
```julia
flat_params = get_parameters_flat(model)
# Use in optimizer
result = optimize(flat_params) do p
    # unflatten and update model
    objective_function(p, model)
end
```

# See also
- [`get_parameters_transformed`](@ref) - Get parameters with transformations as NamedTuple
- [`get_parameters_natural`](@ref) - Get parameters on natural scale as NamedTuple
- [`get_unflatten_fn`](@ref) - Get function to unflatten parameters
"""
function get_parameters_flat(model::MultistateProcess)
    return model.parameters_ph.flat
end

"""
    get_parameters_transformed(model::MultistateProcess)

Get model parameters in transformed (log) scale as a NamedTuple.

Parameters are organized by hazard name with log transformations applied
for positive constraints (baseline, shape, scale). Covariate coefficients
remain on natural scale.

# Returns
- `NamedTuple`: Parameters by hazard name with transformations

# Example
```julia
params_trans = get_parameters_transformed(model)
# Access specific hazard
h12_trans = params_trans.h12  # e.g., positive([log(2.0)])
```

# See also
- [`get_parameters_flat`](@ref) - Get parameters as flat vector for optimization
- [`get_parameters_natural`](@ref) - Get parameters on natural scale
"""
function get_parameters_transformed(model::MultistateProcess)
    return model.parameters_ph.transformed
end

"""
    get_parameters_natural(model::MultistateProcess)

Get model parameters on natural scale as a NamedTuple.

All positive constraints are inverted (exp applied to log scale parameters).
This is the "human-readable" representation where baseline rates, shapes, 
and scales are on their natural positive scale.

# Returns
- `NamedTuple`: Parameters by hazard name on natural scale

# Example
```julia
params_nat = get_parameters_natural(model)
# Access specific hazard - values are on natural scale
h12_baseline = params_nat.h12[1]  # e.g., 2.0 (not log(2.0))
```

# See also
- [`get_parameters_flat`](@ref) - Get parameters as flat vector
- [`get_parameters_transformed`](@ref) - Get parameters with transformations
"""
function get_parameters_natural(model::MultistateProcess)
    return model.parameters_ph.natural
end

"""
    get_unflatten_fn(model::MultistateProcess)

Get the unflatten function to convert flat parameter vector back to NamedTuple.

This function is useful in optimization when you need to convert the flat
vector back to the structured NamedTuple representation.

# Returns
- `Function`: unflatten function that takes a flat vector and returns NamedTuple

# Example
```julia
unflatten = get_unflatten_fn(model)
flat_params = get_parameters_flat(model)
params_transformed = unflatten(flat_params)
params_natural = ParameterHandling.value(params_transformed)
```

# See also
- [`get_parameters_flat`](@ref) - Get flat parameter vector
- [`get_parameters_transformed`](@ref) - Get transformed parameters
"""
function get_unflatten_fn(model::MultistateProcess)
    return model.parameters_ph.unflatten
end

"""
    get_parameters(model::MultistateProcess, h::Int64; scale::Symbol=:natural)

Get parameters for a specific hazard by index.

# Arguments
- `model::MultistateProcess`: The model
- `h::Int64`: Hazard index (1-based)
- `scale::Symbol`: One of `:natural`, `:transformed`, or `:log` (default: `:natural`)

# Returns
- `Vector{Float64}`: Parameters for the specified hazard

# Scales
- `:natural` - Natural scale (baseline/shape/scale are positive values)
- `:transformed` - Transformed scale (from ParameterHandling, includes wrapper)
- `:log` - Log scale (from model.parameters VectorOfVectors, backward compat)

# Examples
```julia
# Get natural scale parameters for hazard 1
params_nat = get_parameters(model, 1)  # Returns [2.0, 1.5, ...]

# Get log scale parameters (legacy representation)
params_log = get_parameters(model, 1, scale=:log)  # Returns [log(2.0), log(1.5), ...]

# Get transformed parameters (with ParameterHandling wrapper)
params_trans = get_parameters(model, 1, scale=:transformed)
```

# See also
- [`get_parameters_flat`](@ref) - Get all parameters as flat vector
- [`get_parameters_natural`](@ref) - Get all parameters on natural scale
- [`set_parameters!`](@ref) - Set parameters for a hazard
"""
function get_parameters(model::MultistateProcess, h::Int64; scale::Symbol=:natural)
    # Validate hazard index
    if h < 1 || h > length(model.parameters)
        throw(BoundsError(model.parameters, h))
    end
    
    if scale == :log
        # Legacy representation - direct from VectorOfVectors
        return model.parameters[h]
    elseif scale == :natural || scale == :transformed
        # Find hazard name from index
        hazname = nothing
        for (name, idx) in model.hazkeys
            if idx == h
                hazname = name
                break
            end
        end
        if hazname === nothing
            error("Could not find hazard name for index $h")
        end
        
        # Return appropriate representation
        if scale == :natural
            return model.parameters_ph.natural[hazname]
        else  # :transformed
            return model.parameters_ph.transformed[hazname]
        end
    else
        throw(ArgumentError("scale must be :natural, :transformed, or :log (got :$scale)"))
    end
end

"""
    check_data!(data::DataFrame, tmat::Matrix, CensoringPatterns::Matrix{Int64}; verbose = true)

Validate a user-supplied data frame to ensure that it conforms to MultistateModels.jl requirements.
"""
function check_data!(data::DataFrame, tmat::Matrix, CensoringPatterns::Matrix{Int64}; verbose = true)

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
        abs_warn = any(map(x -> any(data.statefrom .== x), which_absorbing))

        if verbose && abs_warn
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

    # error if data includes states not in the unique states
    statespace = sort(vcat(0, collect(1:size(tmat,1)), CensoringPatterns[:,1]))
    allstates = sort(vcat(unique(data.stateto), unique(data.statefrom)))
    if !all(allstates .∈ Ref(statespace))
        @error "Data contains states that are not in the state space."
    end

    # warning if tmat specifies an allowed transition for which no such transitions were observed in the data
    n_rs = compute_number_transitions(data, tmat)
    for r in 1:size(tmat)[1]
        for s in 1:size(tmat)[2]
            if verbose && tmat[r,s]!=0 && n_rs[r,s]==0 
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
    build_fbmats(model)

Build the forward recursion matrices.
"""
function build_fbmats(model)

    # get sizes of stuff
    n_states = size(model.tmat, 1)
    n_times = [sum(model.data.id .== s) for s in unique(model.data.id)]

    # create the forward matrices
    fbmats = [zeros(Float64, n_times[s], n_states, n_states) for s in eachindex(n_times)]

    return fbmats
end

"""
    collapse_data(data::DataFrame; SamplingWeights::Vector{Float64} = ones(unique(data.id)))

Collapse subjects to create an internal representation of a dataset and optionally recompute a vector of sampling weights.
"""
function collapse_data(data::DataFrame; SamplingWeights::Vector{Float64} = ones(Float64, length(unique(data.id))))
    
    # find unique subjects
    ids = unique(data.id)
    _data = [DataFrame() for k in 1:length(ids)]
    for k in ids
        _data[k] = data[findall(data.id .== k),Not(:id)]
    end
    _DataCollapsed = unique(_data)

    # find the collapsed dataset for each individual
    inds = map(x -> findfirst(_DataCollapsed .== Ref(x)), _data)

    # tabulate the SamplingWeights
    SamplingWeightsCollapsed = map(x -> sum(SamplingWeights[findall(inds .== x)]), unique(inds))

    # add a fake id variable to the collapsed datasets (for purchasing alcohol)
    for k in 1:length(_DataCollapsed)
        insertcols!(_DataCollapsed[k], :tstart, :id => fill(k, nrow(_DataCollapsed[k])))
    end

    # vcat
    DataCollapsed = reduce(vcat, _DataCollapsed)
       
    return DataCollapsed, SamplingWeightsCollapsed
end

"""
    initialize_surrogate!(model::MultistateSemiMarkovProcess; surrogate_parameters = nothing, surrogate_constraints = nothing, crude_inits = true, verbose = true)

Populate the field for the markov surrogate in semi-Markov models.
"""
function initialize_surrogate!(model::MultistateProcess; surrogate_parameters = nothing, surrogate_constraints = nothing, crude_inits = true, verbose = true)

    # fit surrogate model
    surrogate_fitted = fit_surrogate(model; surrogate_parameters=surrogate_parameters, surrogate_constraints=surrogate_constraints, crude_inits=crude_inits, verbose=verbose)

    # create the surrogate object
    surrogate = MarkovSurrogate(surrogate_fitted.hazards, surrogate_fitted.parameters)

    for i in eachindex(model.markovsurrogate.parameters)
        copyto!(model.markovsurrogate.parameters[i], surrogate_fitted.parameters[i])
    end
end

"""
    make_subjdat(path::SamplePath, subjectdata::SubDataFrame)

Create a DataFrame for a single subject from a SamplePath object and the original data for that subject.
"""
function make_subjdat(path::SamplePath, subjectdata::SubDataFrame) 

    # times when the likelihood needs to be evaluated
    if (ncol(subjectdata) > 6) & (nrow(subjectdata) > 1)
        
        # times when covariates change
        keepvec = findall(map(i -> !isequal(subjectdata[i-1, 7:end], subjectdata[i, 7:end]), 2:nrow(subjectdata))) .+ 1 

        # utimes is the times in the path (which includes first and last times from subjectdata), and tstart when covariates change
        utimes = sort(unique(vcat(path.times, subjectdata.tstart[keepvec])))

    else 
        utimes = path.times
    end

    # get indices in the data object that correspond to the unique times
    datinds = searchsortedlast.(Ref(subjectdata.tstart), utimes)

    # get indices in the path that correspond to the unique times
    pathinds = searchsortedlast.(Ref(path.times), utimes)

    # make subject data
    subjdat_lik = DataFrame(
        tstart = utimes[Not(end)],
        tstop  = utimes[Not(begin)],
        increment = diff(utimes),
        sojourn = 0.0,
        sojournind = pathinds[Not(end)],
        statefrom = path.states[pathinds[Not(end)]],
        stateto   = path.states[pathinds[Not(begin)]])

    # compute sojourns
    subjdat_gdf = groupby(subjdat_lik, :sojournind)
    for g in subjdat_gdf
        g.sojourn .= cumsum([0.0; g.increment[Not(end)]])
    end

    # remove sojournind
    select!(subjdat_lik, Not(:sojournind))

    # tack on covariates if any
    if ncol(subjectdata) > 6
        # get covariates at the relevant times
        covars = subjectdata[datinds[Not(end)], Not([:id, :tstart, :tstop, :statefrom, :stateto, :obstype])]
        # concatenate and return
        subjdat_lik = hcat(subjdat_lik, covars)
    end
    
    # output
    return subjdat_lik
end