# =============================================================================
# User-Facing Hazard API Functions
# =============================================================================
#
# Public API functions for computing hazards, cumulative hazards, and
# cumulative incidence.
#
# =============================================================================

"""
    _make_covar_namedtuple(covar_names, covar_dict)

Create a NamedTuple from covariate names and a dictionary of values.
Returns an empty NamedTuple if no covariates are present.
"""
function _make_covar_namedtuple(covar_names::Vector{Symbol}, covar_dict::Dict{Symbol, Float64})
    if isempty(covar_names)
        return NamedTuple()
    else
        keys = Tuple(covar_names)
        vals = Tuple(get(covar_dict, cn, 0.0) for cn in covar_names)
        return NamedTuple{keys}(vals)
    end
end

"""
    cumulative_incidence(t, model::MultistateProcess, subj::Int64=1)

Compute the cumulative incidence for each possible transition as a function of time since state entry. Assumes the subject starts their observation period at risk and saves cumulative incidence at the supplied vector of times, t.
"""
function cumulative_incidence(t, model::MultistateProcess, subj::Int64=1)

    # grab parameters, hazards and total hazards
    parameters   = model.parameters
    hazards      = model.hazards
    totalhazards = model.totalhazards

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # merge times with left endpoints of subject observation intervals
    subj_times = sort(unique([0.0; t]))

    # identify transient states
    transients = findall(isa.(totalhazards, _TotalHazardTransient))

    # identify which transient state to grab for each hazard (as transients[trans_inds[h]])
    trans_inds  = reduce(vcat, [i * ones(Int64, length(totalhazards[transients[i]].components)) for i in eachindex(transients)])

    # initialize cumulative incidence
    n_intervals = length(subj_times) - 1
    incidences  = zeros(Float64, n_intervals, length(hazards))
    survprobs   = ones(Float64, n_intervals, length(transients))

    # indices for starting cumulative incidence increments
    interval_inds = map(x -> searchsortedlast(subj_dat.tstart .- minimum(subj_dat.tstart), x), subj_times[Not(end)])

    # compute the survival probabilities to start each interval
    if n_intervals > 1
        for s in eachindex(transients)
            # initialize sprob and identify origin state
            sprob = 1.0
            statefrom = transients[s]

            # compute survival probabilities
            for i in 2:n_intervals
                survprobs[i,s] = sprob * survprob(subj_times[i-1], subj_times[i], parameters, subj_inds[interval_inds[i-1]], totalhazards[statefrom], hazards; give_log = false)
                sprob = survprobs[i,s]
            end
        end
    end
    
    # compute the cumulative incidence for each transition type
    for h in eachindex(hazards)
        # identify origin state
        statefrom = transients[trans_inds[h]]
        hazard = hazards[h]

        # compute incidences
        for r in 1:n_intervals
            subjdat_row = subj_dat[interval_inds[r], :]
            covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
            incidences[r,h] = 
                survprobs[r,trans_inds[h]] * 
                quadgk(t -> (
                        eval_hazard(hazard, t, parameters[hazard.hazname], covars) * 
                        survprob(subj_times[r], t, parameters, subjdat_row, totalhazards[statefrom], hazards; give_log = false)), 
                        subj_times[r], subj_times[r + 1])[1]
        end        
    end

    # return cumulative incidences
    return cumsum(incidences; dims = 1)
end

"""
    cumulative_incidence(t, model::MultistateProcess, parameters, statefrom, subj::Int64=1)

Compute the cumulative incidence for each possible transition originating in `statefrom` as a function of time since state entry. Assumes the subject starts their observation period at risk and saves cumulative incidence at the supplied vector of times since state entry. This function is used internally.
"""
function cumulative_incidence(t, model::MultistateProcess, parameters, statefrom, subj::Int64=1)

    # get hazards
    hazards = model.hazards

    # get total hazards
    totalhazards = model.totalhazards

    # return zero if starting from absorbing state
    if isa(totalhazards[statefrom], _TotalHazardAbsorbing)
        return zeros(length(t))
    end

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # merge times with left endpoints of subject observation intervals
    subj_times = sort(unique([0.0; t]))

    # initialize cumulative incidence
    n_intervals = length(subj_times) - 1
    hazinds     = totalhazards[statefrom].components
    incidences  = zeros(Float64, n_intervals, length(hazinds))
    survprobs   = ones(Float64, n_intervals)

    # indices for starting cumulative incidence increments
    interval_inds = map(x -> searchsortedlast(subj_dat.tstart .- minimum(subj_dat.tstart), x), subj_times[Not(end)])

    # compute the survival probabilities to start each interval
    if n_intervals > 1

        # initialize sprob
        sprob = 1.0

        # compute survival probabilities
        for i in 2:n_intervals
            survprobs[i] = sprob * survprob(subj_times[i-1], subj_times[i], parameters, subj_inds[interval_inds[i-1]], totalhazards[statefrom], hazards; give_log = false)
            sprob = survprobs[i]
        end
    end
    
    # compute the cumulative incidence for each transition type
    for h in eachindex(hazinds)
        for r in 1:n_intervals
            subjdat_row = subj_dat[interval_inds[r], :]
            hazard = hazards[hazinds[h]]
            covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
            incidences[r,h] = 
                survprobs[r] * 
                quadgk(t -> (
                        eval_hazard(hazard, t, parameters[hazinds[h]], covars) * 
                        survprob(subj_times[r], t, parameters, subjdat_row, totalhazards[statefrom], hazards; give_log = false)), 
                        subj_times[r], subj_times[r + 1])[1]
        end        
    end

    # return cumulative incidences
    return cumsum(incidences; dims = 1)
end

"""
    compute_hazard(t, model::MultistateProcess, hazard::Symbol, subj::Int64=1)

Compute the hazard at times t. 

# Arguments
- t: time or vector of times. 
- model: MultistateProcess object. 
- hazard: Symbol specifying the hazard, e.g., :h12 for the hazard for transitioning from state 1 to state 2. 
- subj: subject id. 
"""
function compute_hazard(t, model::MultistateProcess, hazard::Symbol, subj::Int64 = 1)

    # get hazard index
    hazind = model.hazkeys[hazard]
    
    # get natural-scale parameters for this hazard (required by eval_hazard)
    hazard_params = get_hazard_params(model.parameters, model.hazards)[hazind]
    haz = model.hazards[hazind]

    # compute hazards
    hazards = zeros(Float64, length(t))
    for s in eachindex(t)
        # get row
        rowind = findlast((model.data.id .== subj) .& (model.data.tstart .<= t[s]))
        subjdat_row = model.data[rowind, :]
        covars = extract_covariates_fast(subjdat_row, haz.covar_names)

        # compute hazard
        hazards[s] = eval_hazard(haz, t[s], hazard_params, covars)
    end

    # return hazards
    return hazards
end

"""
    compute_cumulative_hazard(tstart, tstop, model::MultistateProcess, hazard::Symbol, subj::Int64=1)

Compute the cumulative hazard over [tstart,tstop]. 

# Arguments
- tstart: starting times
- tstop: stopping times
- model: MultistateProcess object. 
- hazard: Symbol specifying the hazard, e.g., :h12 for the hazard for transitioning from state 1 to state 2. 
- subj: subject id. 
"""
function compute_cumulative_hazard(tstart, tstop, model::MultistateProcess, hazard::Symbol, subj::Int64 = 1)

    # check bounds
    if (length(tstart) == length(tstop))
        # nothing to do
    elseif (length(tstart) == 1) & (length(tstop) != 1)
        tstart = rep(tstart, length(tstart))
    elseif (length(tstart) != 1) & (length(tstop) == 1)
        tstop = rep(tstop, length(tstart))
    else
        throw(ArgumentError("Lengths of tstart ($(length(tstart))) and tstop ($(length(tstop))) are not compatible."))
    end

    # get hazard index
    hazind = model.hazkeys[hazard]
    
    # get natural-scale parameters for this hazard (required by eval_cumhaz)
    hazard_params = get_hazard_params(model.parameters, model.hazards)[hazind]
    haz = model.hazards[hazind]

    # compute hazards
    cumulative_hazards = zeros(Float64, length(tstart))
    for s in eachindex(tstart)

        # find times between tstart and tstop
        times = [tstart[s]; model.data.tstart[findall((model.data.id .== subj) .& (model.data.tstart .> tstart[s]) .& (model.data.tstart .< tstop[s]))]; tstop[s]]

        # initialize cumulative hazard
        chaz = 0.0

        # accumulate
        for i in 1:(length(times) - 1)
            # get row
            rowind = findlast((model.data.id .== subj) .& (model.data.tstart .<= times[i]))
            subjdat_row = model.data[rowind, :]
            covars = extract_covariates_fast(subjdat_row, haz.covar_names)

            # compute cumulative hazard
            chaz += eval_cumhaz(haz, times[i], times[i+1], hazard_params, covars)
        end

        # save
        cumulative_hazards[s] = chaz
    end

    # return cumulative hazards
    return cumulative_hazards
end
# =============================================================================
# Cumulative Incidence with Explicit Covariate Specification
# =============================================================================

"""
    cumulative_incidence(t, model::MultistateProcess, newdata::NamedTuple; statefrom::Int)

Compute the cumulative incidence at specified covariate values.

This method allows computing cumulative incidence without requiring a subject in the
data with matching covariate values. Useful for:
- Computing cumulative incidence at reference level (all covariates = 0)
- Evaluating cumulative incidence at specific covariate combinations
- Knot calibration for spline hazards

# Arguments
- `t`: Vector of times at which to evaluate cumulative incidence
- `model`: MultistateProcess object
- `newdata`: NamedTuple specifying covariate values, e.g., `(age=50.0, treatment=1.0)`
- `statefrom`: Origin state for cumulative incidence calculation

# Returns
Matrix of cumulative incidences (length(t) × n_transitions_from_statefrom).
Each column corresponds to a transition from `statefrom` to a different destination.

# Examples
```julia
# Compute cumulative incidence at reference level (all covariates = 0)
ci = cumulative_incidence([0.5, 1.0, 2.0], model, (age=0.0, treatment=0.0); statefrom=1)

# Compute at specific covariate values
ci = cumulative_incidence([0.5, 1.0, 2.0], model, (age=50.0, treatment=1.0); statefrom=1)
```

See also: [`cumulative_incidence_at_reference`](@ref)
"""
function cumulative_incidence(t, model::MultistateProcess, newdata::NamedTuple; statefrom::Int)
    # Get all covariate names used by hazards from this origin state
    totalhazards = model.totalhazards
    hazards = model.hazards
    
    # Return zero if starting from absorbing state
    if isa(totalhazards[statefrom], _TotalHazardAbsorbing)
        return zeros(length(t), 0)
    end
    
    # Get hazard indices for this origin state
    hazinds = totalhazards[statefrom].components
    
    # Collect all covariate names from hazards originating from statefrom
    all_covar_names = Symbol[]
    for hidx in hazinds
        append!(all_covar_names, hazards[hidx].covar_names)
    end
    unique!(all_covar_names)
    
    # Build covariate dictionary with defaults of 0.0
    covar_dict = Dict{Symbol, Float64}()
    for name in all_covar_names
        covar_dict[name] = haskey(newdata, name) ? Float64(newdata[name]) : 0.0
    end
    
    # Get parameters (use model.parameters.nested which contains per-hazard structure)
    parameters = model.parameters.nested
    
    # Time grid
    times = sort(unique([0.0; t]))
    n_intervals = length(times) - 1
    
    # Initialize output
    incidences = zeros(Float64, n_intervals, length(hazinds))
    survprobs = ones(Float64, n_intervals)
    
    # Compute survival probabilities for each interval
    if n_intervals > 1
        sprob = 1.0
        for i in 2:n_intervals
            # Compute total hazard integral over interval using covariate values
            total_cumhaz = 0.0
            for hidx in hazinds
                haz = hazards[hidx]
                haz_pars = parameters[haz.hazname]
                covars = _make_covar_namedtuple(haz.covar_names, covar_dict)
                total_cumhaz += eval_cumhaz(haz, times[i-1], times[i], haz_pars, covars)
            end
            survprobs[i] = sprob * exp(-total_cumhaz)
            sprob = survprobs[i]
        end
    end
    
    # Compute cumulative incidence for each transition type
    for (h_idx, hidx) in enumerate(hazinds)
        haz = hazards[hidx]
        haz_pars = parameters[haz.hazname]
        covars = _make_covar_namedtuple(haz.covar_names, covar_dict)
        
        for r in 1:n_intervals
            # Compute cause-specific incidence over this interval
            # CI_j(t) = ∫₀ᵗ S(u) h_j(u) du
            integrand = u -> begin
                # Survival to time u
                S_u = survprobs[r]
                if u > times[r]
                    # Additional survival from interval start to u
                    cumhaz_extra = 0.0
                    for kidx in hazinds
                        khaz = hazards[kidx]
                        khaz_pars = parameters[khaz.hazname]
                        kcovars = _make_covar_namedtuple(khaz.covar_names, covar_dict)
                        cumhaz_extra += eval_cumhaz(khaz, times[r], u, khaz_pars, kcovars)
                    end
                    S_u *= exp(-cumhaz_extra)
                end
                # Cause-specific hazard at u
                h_j_u = eval_hazard(haz, u, haz_pars, covars)
                return S_u * h_j_u
            end
            
            incidences[r, h_idx], _ = quadgk(integrand, times[r], times[r + 1])
        end
    end
    
    # Return cumulative sums
    return cumsum(incidences; dims = 1)
end

"""
    cumulative_incidence(t, model::MultistateProcess, newdata::DataFrameRow; statefrom::Int)

Compute the cumulative incidence at covariate values specified by a DataFrameRow.

Converts the DataFrameRow to a NamedTuple and calls the NamedTuple method.

# Arguments
- `t`: Vector of times at which to evaluate cumulative incidence
- `model`: MultistateProcess object  
- `newdata`: DataFrameRow specifying covariate values
- `statefrom`: Origin state for cumulative incidence calculation

# Returns
Matrix of cumulative incidences (length(t) × n_transitions_from_statefrom).

# Examples
```julia
# Use first row of a DataFrame as covariate specification
df_row = DataFrame(age=50.0, treatment=1.0)[1, :]
ci = cumulative_incidence([0.5, 1.0, 2.0], model, df_row; statefrom=1)
```

See also: [`cumulative_incidence_at_reference`](@ref)
"""
function cumulative_incidence(t, model::MultistateProcess, newdata::DataFrameRow; statefrom::Int)
    # Convert DataFrameRow to NamedTuple
    nt = NamedTuple{Tuple(propertynames(newdata))}(Tuple(newdata))
    return cumulative_incidence(t, model, nt; statefrom=statefrom)
end

"""
    cumulative_incidence_at_reference(t, model::MultistateProcess; statefrom::Int)

Compute the cumulative incidence at reference covariate values (all covariates = 0).

This is a convenience function for computing cumulative incidence at the reference
level, which is useful for:
- Baseline hazard interpretation
- Spline knot calibration (placing knots at quantiles of exit time distribution)
- Computing reference-level survival curves

# Arguments
- `t`: Vector of times at which to evaluate cumulative incidence
- `model`: MultistateProcess object
- `statefrom`: Origin state for cumulative incidence calculation

# Returns
Matrix of cumulative incidences (length(t) × n_transitions_from_statefrom).
Each column corresponds to a transition from `statefrom` to a different destination.

# Examples
```julia
# Compute cumulative incidence at reference level
ci_ref = cumulative_incidence_at_reference([0.5, 1.0, 2.0, 5.0], model; statefrom=1)

# Total exit probability at reference level (sum across destinations)
total_exit = sum(ci_ref, dims=2)

# Invert to get median exit time (50th percentile)
t_grid = range(0.0, 10.0, length=1000)
ci_grid = cumulative_incidence_at_reference(collect(t_grid), model; statefrom=1)
total_exit = vec(sum(ci_grid, dims=2))
idx_median = searchsortedfirst(total_exit, 0.5)
t_median = t_grid[idx_median]
```

See also: [`cumulative_incidence`](@ref)
"""
function cumulative_incidence_at_reference(t, model::MultistateProcess; statefrom::Int)
    # Get all covariate names from hazards originating from this state
    totalhazards = model.totalhazards
    hazards = model.hazards
    
    # Return zero if starting from absorbing state  
    if isa(totalhazards[statefrom], _TotalHazardAbsorbing)
        return zeros(length(t), 0)
    end
    
    # Get hazard indices for this origin state
    hazinds = totalhazards[statefrom].components
    
    # Collect all covariate names
    all_covar_names = Symbol[]
    for hidx in hazinds
        append!(all_covar_names, hazards[hidx].covar_names)
    end
    unique!(all_covar_names)
    
    # Create reference NamedTuple with all zeros
    if isempty(all_covar_names)
        ref_data = NamedTuple()
    else
        ref_data = NamedTuple{Tuple(all_covar_names)}(zeros(length(all_covar_names)))
    end
    
    return cumulative_incidence(t, model, ref_data; statefrom=statefrom)
end