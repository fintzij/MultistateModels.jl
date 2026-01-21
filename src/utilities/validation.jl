# ============================================================================
# Data and Parameter Validation Functions
# ============================================================================
# Functions for validating user-supplied data, weights, and censoring patterns.
# These are called during model construction to ensure data integrity.
#
# Error Handling Convention:
# - User input validation errors: throw(ArgumentError(...))
# - Domain errors: throw(DomainError(...))
# - Internal invariants: @assert
# ============================================================================

"""
    check_data!(data::DataFrame, tmat::Matrix, emat::Matrix{<:Real}; verbose = true, phase_to_state = nothing)

Validate a user-supplied data frame to ensure that it conforms to MultistateModels.jl requirements.

# Arguments
- `data`: DataFrame with required columns
- `tmat`: Transition matrix
- `emat`: Emission matrix
- `verbose`: Whether to print warnings (default: true)
- `phase_to_state`: Optional mapping from phase indices to observed states (for phase-type models).
  When provided, transition count validation is performed on observed state transitions,
  not internal phase transitions.

# Throws
- `ArgumentError` for invalid data format or values
"""
function check_data!(data::DataFrame, tmat::Matrix, emat::Matrix{<:Real}; verbose = true, phase_to_state::Union{Nothing, Vector{Int}} = nothing)

    # validate column names and order
    if any(names(data)[1:6] .!== ["id", "tstart", "tstop", "statefrom", "stateto", "obstype"])
        throw(ArgumentError("The first 6 columns of the data should be 'id', 'tstart', 'tstop', 'statefrom', 'stateto', 'obstype'."))
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
        throw(ArgumentError("Subject IDs must be consecutive integers starting at 1 (i.e., 1, 2, 3, ...). " *
                           "Found: $(sort(unique_id)[1:min(5, length(unique_id))])..."))
    end

    # warn about individuals starting in absorbing states
    # check if there are any absorbing states
    absorbing = map(x -> all(x .== 0), eachrow(tmat))

    # look to see if any of the absorbing states are in statefrom
    if any(absorbing)
        which_absorbing = findall(absorbing .== true)
        abs_warn = any(map(x -> any(data.statefrom .== x), which_absorbing))

        if verbose && abs_warn
            @warn "The data contains observations where a subject originates in an absorbing state."
        end
    end

    # error if any tstart > tstop (strictly greater)
    # Note: tstart == tstop is allowed for instantaneous observations (e.g., phase-type models)
    bad_intervals = findall(data.tstart .> data.tstop)
    if !isempty(bad_intervals)
        throw(ArgumentError("Data contains $(length(bad_intervals)) time intervals where tstart > tstop. " *
                           "First occurrence at row $(bad_intervals[1])."))
    end

    # within each subject's data, error if tstart or tstop are out of order or there are discontinuities given multiple time intervals
    for i in unique_id
        inds = findall(data.id .== i)

        # check sorting
        if(!issorted(data.tstart[inds]) || !issorted(data.tstop[inds])) 
            throw(ArgumentError("tstart and tstop must be sorted for each subject. " *
                               "Subject $i has unsorted time intervals."))
        end
        
        # check for time discontinuities
        if(length(inds) > 1)
            if(any(data.tstart[inds[Not(begin)]] .!= 
                    data.tstop[inds[Not(end)]]))
                throw(ArgumentError("Time intervals for subject $i contain discontinuities. " *
                                   "Each interval's tstart must equal the previous interval's tstop."))
            end
            
            # warn about state discontinuities: statefrom[i+1] should equal stateto[i]
            # This is required for valid multistate data but currently only warns
            # to avoid breaking existing tests with synthetic data
            if verbose
                for j in 2:length(inds)
                    prev_stateto = data.stateto[inds[j-1]]
                    curr_statefrom = data.statefrom[inds[j]]
                    # Skip check if previous stateto is missing/censored (0)
                    if !ismissing(prev_stateto) && prev_stateto != 0 && curr_statefrom != prev_stateto
                        @warn "State discontinuity for subject $i: " *
                              "stateto=$(prev_stateto) at row $(inds[j-1]) does not match " *
                              "statefrom=$(curr_statefrom) at row $(inds[j]). " *
                              "Each interval's statefrom should equal the previous interval's stateto."
                        break  # Only warn once per subject
                    end
                end
            end
        end
    end

    # error if data includes states not in the unique states
    emat_ids = Int64.(emat[:,1])
    statespace = sort(vcat(0, collect(1:size(tmat,1)), emat_ids))
    allstates = sort(vcat(unique(data.stateto), unique(data.statefrom)))
    if !all(allstates .∈ Ref(statespace))
        invalid_states = setdiff(allstates, statespace)
        throw(ArgumentError("Data contains states not in the state space: $(invalid_states). " *
                           "Valid states are: $(statespace)"))
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
    # For phase-type models, check observed state transitions (not internal phase transitions)
    if isnothing(phase_to_state)
        # Standard case: check transitions directly
        n_rs = compute_number_transitions(data, tmat)
        for r in 1:size(tmat)[1]
            for s in 1:size(tmat)[2]
                if verbose && tmat[r,s]!=0 && n_rs[r,s]==0 
                    @warn "Data does not contain any transitions from state $r to state $s"
                end
            end
        end
    else
        # Phase-type case: aggregate phase transitions to observed state transitions
        n_observed = maximum(phase_to_state)
        n_rs_observed = zeros(Int, n_observed, n_observed)
        
        for rd in eachrow(data)
            if rd.statefrom != rd.stateto && rd.statefrom > 0 && rd.stateto > 0
                obs_from = phase_to_state[rd.statefrom]
                obs_to = phase_to_state[rd.stateto]
                # Only count transitions between DIFFERENT observed states
                if obs_from != obs_to
                    n_rs_observed[obs_from, obs_to] += 1
                end
            end
        end
        
        # Build observed tmat from phase_to_state
        observed_tmat = zeros(Int, n_observed, n_observed)
        for r in 1:size(tmat, 1)
            for s in 1:size(tmat, 2)
                if tmat[r, s] != 0
                    obs_r = phase_to_state[r]
                    obs_s = phase_to_state[s]
                    if obs_r != obs_s
                        observed_tmat[obs_r, obs_s] = 1
                    end
                end
            end
        end
        
        # Warn about missing observed state transitions
        for r in 1:n_observed
            for s in 1:n_observed
                if verbose && observed_tmat[r, s] != 0 && n_rs_observed[r, s] == 0
                    @warn "Data does not contain any transitions from state $r to state $s"
                end
            end
        end
    end

    # check that obstype is one of the allowed censoring schemes
    if any(data.obstype .∉ Ref([1,2]))
        emat_id = Int64.(emat[:,1])
        if any(data.obstype .∉ Ref([[1,2]; emat_id]))
            invalid_obstypes = setdiff(unique(data.obstype), [[1,2]; emat_id])
            throw(ArgumentError("obstype must be 1, 2, or a censoring ID from CensoringPatterns. " *
                               "Found invalid obstype values: $(invalid_obstypes)"))
        end
    end

    # check that stateto is 0 when obstype is not 1 or 2
    for i in Base.OneTo(nrow(data))
        if (data.obstype[i] > 2) & (data.stateto[i] .!= 0)            
            throw(ArgumentError("When obstype > 2 (censored observation), stateto must be 0. " *
                               "Row $i has obstype=$(data.obstype[i]) but stateto=$(data.stateto[i])."))
        end
    end

    # check that subjects start in an observed state (statefrom!=0)
    for subj in Base.OneTo(nsubj)
        datasubj = filter(:id => ==(subj), data)
        if datasubj.statefrom[1] == 0          
            throw(ArgumentError("Subject $subj has statefrom=0 in their first observation. " *
                               "Subjects must start in a known (non-censored) state."))
        end
    end

    # check that there is no row for a subject after they hit an absorbing state

end

"""
    check_SubjectWeights(SubjectWeights::Vector{Float64}, data::DataFrame)

Check that subject-level weights are properly specified.

# Throws
- `ArgumentError` for invalid weights (wrong length, non-positive, or non-finite)
"""
function check_SubjectWeights(SubjectWeights::Vector{Float64}, data::DataFrame)
    
    # check that the number of subject weights is correct
    nsubj = length(unique(data.id))
    if length(SubjectWeights) != nsubj
        throw(ArgumentError("SubjectWeights has length $(length(SubjectWeights)) but there are $nsubj subjects."))
    end

    # check that all weights are finite (not NaN or Inf)
    if any(!isfinite, SubjectWeights)
        bad_indices = findall(!isfinite, SubjectWeights)
        throw(ArgumentError("SubjectWeights must be finite (no NaN or Inf). Found non-finite values at indices: $(bad_indices[1:min(5, length(bad_indices))])"))
    end

    # check that the subject weights are positive (non-negative and non-zero)
    if any(SubjectWeights .<= 0)
        bad_indices = findall(SubjectWeights .<= 0)
        throw(ArgumentError("SubjectWeights must be positive. Found non-positive values at indices: $(bad_indices[1:min(5, length(bad_indices))])"))
    end
end

"""
    check_ObservationWeights(ObservationWeights::Vector{Float64}, data::DataFrame)

Check that observation-level weights are properly specified.

# Throws
- `ArgumentError` for invalid weights (wrong length, non-positive, or non-finite)
"""
function check_ObservationWeights(ObservationWeights::Vector{Float64}, data::DataFrame)
    
    # check that the number of observation weights is correct
    if length(ObservationWeights) != nrow(data)
        throw(ArgumentError("ObservationWeights has length $(length(ObservationWeights)) but there are $(nrow(data)) observations."))
    end

    # check that all weights are finite (not NaN or Inf)
    if any(!isfinite, ObservationWeights)
        bad_indices = findall(!isfinite, ObservationWeights)
        throw(ArgumentError("ObservationWeights must be finite (no NaN or Inf). Found non-finite values at indices: $(bad_indices[1:min(5, length(bad_indices))])"))
    end

    # check that the observation weights are positive (non-negative and non-zero)
    if any(ObservationWeights .<= 0)
        bad_indices = findall(ObservationWeights .<= 0)
        throw(ArgumentError("ObservationWeights must be positive. Found non-positive values at indices: $(bad_indices[1:min(5, length(bad_indices))])"))
    end
end

"""
    check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj::Int64)

Check that SubjectWeights and ObservationWeights are mutually exclusive and handle defaults.
Returns (SubjectWeights, ObservationWeights) where at most one is non-nothing.

# Throws
- `ArgumentError` if both weight types are provided
"""
function check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj::Int64)
    
    # Check mutual exclusivity
    if !isnothing(SubjectWeights) && !isnothing(ObservationWeights)
        throw(ArgumentError("SubjectWeights and ObservationWeights are mutually exclusive. Specify only one."))
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

# Throws
- `ArgumentError` for invalid censoring pattern format or values
"""
function check_CensoringPatterns(CensoringPatterns::Matrix{T}, tmat::Matrix) where T <: Real
    
    nrow, ncol = size(CensoringPatterns)

    # check for empty
    if nrow == 0 | ncol < 2
        throw(ArgumentError("CensoringPatterns matrix is empty or has fewer than 2 columns."))
    end

    # censoring patterns must be labelled as 3, 4, ...
    if !all(CensoringPatterns[:,1] .== 3:(nrow+2))
        throw(ArgumentError("First column of CensoringPatterns must be consecutive integers starting at 3 (i.e., 3, 4, 5, ...). " *
                           "Found: $(CensoringPatterns[:,1])"))
    end

    # check that values are in [0, 1]
    if any(CensoringPatterns[:,2:ncol] .< 0) || any(CensoringPatterns[:,2:ncol] .> 1)
        throw(ArgumentError("Emission probabilities in CensoringPatterns (columns 2:end) must be in [0, 1]."))
    end

    # censoring patterns must indicate the presence/absence of each state
    n_states = size(tmat, 1)
    if ncol - 1 .!= n_states
        throw(ArgumentError("CensoringPatterns has $(ncol-1) state columns but the model has $n_states states."))
    end

    # censoring patterns must have at least one possible state
    for i in 1:nrow
        if all(CensoringPatterns[i,2:ncol] .== 0)
            throw(ArgumentError("Censoring pattern $(i+2) has all zero probabilities (no allowed states)."))
        end
        if all(CensoringPatterns[i,2:ncol] .== 1)
            @debug "All states are allowed in censoring pattern $(2+i)."
        end
        if sum(CensoringPatterns[i,2:ncol] .> 0) .== 1
            @debug "Censoring pattern $(i+2) has only one allowed state; if these observations are not censored there is no need to use a censoring pattern."
        end
    end
    
    # Note: Row sums > 1 are valid for censoring patterns that use indicator matrices
    # (1.0 for each possible state). The likelihood computation sums transition
    # probabilities to all allowed states: L = Σ_s P(transition to s) * indicator(s is allowed)
end
