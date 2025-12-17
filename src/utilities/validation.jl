# ============================================================================
# Data and Parameter Validation Functions
# ============================================================================
# Functions for validating user-supplied data, weights, and censoring patterns.
# These are called during model construction to ensure data integrity.
# ============================================================================

"""
    check_data!(data::DataFrame, tmat::Matrix, emat::Matrix{<:Real}; verbose = true)

Validate a user-supplied data frame to ensure that it conforms to MultistateModels.jl requirements.
"""
function check_data!(data::DataFrame, tmat::Matrix, emat::Matrix{<:Real}; verbose = true)

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
    emat_ids = Int64.(emat[:,1])
    statespace = sort(vcat(0, collect(1:size(tmat,1)), emat_ids))
    allstates = sort(vcat(unique(data.stateto), unique(data.statefrom)))
    if !all(allstates .∈ Ref(statespace))
        error("Data contains states that are not in the state space.")
    end

    # warn if state labels are not contiguous (e.g., 1,2,4 instead of 1,2,3)
    # Model creation will fail if states are not 1,2,3,...,n
    observed_states = sort(unique(filter(!=(0), allstates)))  # Exclude state 0 (censoring)
    if !isempty(observed_states)
        expected_states = collect(1:maximum(observed_states))
        if observed_states != expected_states
            missing_states = setdiff(expected_states, observed_states)
            if verbose
                @warn "State labels are not contiguous. States $(missing_states) are missing. State labels must be 1, 2, ..., n for model creation to succeed."
            end
        end
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
        emat_id = Int64.(emat[:,1])
        if any(data.obstype .∉ Ref([[1,2]; emat_id]))
            error("obstype should be one of 1, 2, or a censoring id from emat.")
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

"""
    check_SubjectWeights(SubjectWeights::Vector{Float64}, data::DataFrame)

Check that subject-level weights are properly specified.
"""
function check_SubjectWeights(SubjectWeights::Vector{Float64}, data::DataFrame)
    
    # check that the number of subject weights is correct
    if length(SubjectWeights) != length(unique(data.id))
        error("The length of SubjectWeights is not equal to the number of subjects.")
    end

    # check that the subject weights are non-negative
    if any(SubjectWeights .<= 0)
        error("The elements of SubjectWeights should be positive.")
    end
end

"""
    check_ObservationWeights(ObservationWeights::Vector{Float64}, data::DataFrame)

Check that observation-level weights are properly specified.
"""
function check_ObservationWeights(ObservationWeights::Vector{Float64}, data::DataFrame)
    
    # check that the number of observation weights is correct
    if length(ObservationWeights) != nrow(data)
        error("The length of ObservationWeights ($(length(ObservationWeights))) is not equal to the number of observations ($(nrow(data))).")
    end

    # check that the observation weights are non-negative
    if any(ObservationWeights .<= 0)
        error("The elements of ObservationWeights should be positive.")
    end
end

"""
    check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj::Int64)

Check that SubjectWeights and ObservationWeights are mutually exclusive and handle defaults.
Returns (SubjectWeights, ObservationWeights) where at most one is non-nothing.
"""
function check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj::Int64)
    
    # Check mutual exclusivity
    if !isnothing(SubjectWeights) && !isnothing(ObservationWeights)
        error("SubjectWeights and ObservationWeights are mutually exclusive. Specify only one.")
    end
    
    # SubjectWeights must always be a Vector{Float64} for the model struct
    # Set to ones if not provided (whether ObservationWeights is used or not)
    if isnothing(SubjectWeights)
        SubjectWeights = ones(Float64, nsubj)
    end
    
    return (SubjectWeights, ObservationWeights)
end

"""
    check_CensoringPatterns(CensoringPatterns::Matrix, tmat::Matrix)

Validate a user-supplied censoring patterns matrix to ensure it conforms to MultistateModels.jl requirements.
Accepts both Int64 (binary 0/1) and Float64 (emission probabilities in [0,1]).
"""
function check_CensoringPatterns(CensoringPatterns::Matrix{T}, tmat::Matrix) where T <: Real
    
    nrow, ncol = size(CensoringPatterns)

    # check for empty
    if nrow == 0 | ncol < 2
        error("The matrix CensoringPatterns seems to be empty, while there are censored states.")
    end

    # censoring patterns must be labelled as 3, 4, ...
    if !all(CensoringPatterns[:,1] .== 3:(nrow+2))
        error("The first column of the matrix `CensoringPatterns` must be of the form (3, 4, ...) .")
    end

    # check that values are in [0, 1]
    if any(CensoringPatterns[:,2:ncol] .< 0) || any(CensoringPatterns[:,2:ncol] .> 1)
        error("Columns 2, 3, ... of CensoringPatterns must have values in [0, 1].")
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
        if sum(CensoringPatterns[i,2:ncol] .> 0) .== 1
            println("Censoring pattern $i has only one allowed state; if these observations are not censored there is no need to use a censoring pattern.")
        end
    end
end
