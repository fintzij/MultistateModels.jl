#=============================================================================
PHASE 2: Runtime-Generated Hazard Functions

PARAMETER SCALE CONVENTION (Phase 2.5):
- All baseline/shape/scale parameters are stored and passed on LOG SCALE
- This matches the model.parameters storage convention (legacy compatibility)
- Covariate coefficients (β) remain on NATURAL SCALE
- Functions internally apply exp() transformations to return natural scale hazards
- ParameterHandling.jl integration (Phase 3) will handle transformations explicitly

Example:
  model.parameters = [[0.8], [log(2.5), log(1.5)]]  # Log scale storage
  hazard_fn(t, [0.8], []) returns exp(0.8) = 2.225...  # Natural scale output
=============================================================================# 

"""
    extract_covar_names(parnames::Vector{Symbol})

Extract covariate names from parameter names by removing hazard prefix and excluding Intercept/shape/scale.

# Example
```julia
parnames = [:h12_Intercept, :h12_age, :h12_trt]
extract_covar_names(parnames)  # Returns [:age, :trt]

parnames_wei = [:h12_shape, :h12_scale, :h12_age]
extract_covar_names(parnames_wei)  # Returns [:age]
```
"""
function extract_covar_names(parnames::Vector{Symbol})
    covar_names = Symbol[]
    for pname in parnames
        pname_str = String(pname)
        # Skip baseline parameters (not covariates)
        # Exponential: "Intercept", Weibull/Gompertz: "shape" and "scale"
        if occursin("Intercept", pname_str) || occursin("shape", pname_str) || occursin("scale", pname_str)
            continue
        end
        # Remove hazard prefix (e.g., "h12_age" -> "age")
        covar_name = replace(pname_str, r"^h\d+_" => "")
        push!(covar_names, Symbol(covar_name))
    end
    return covar_names
end

"""
    extract_covariates(subjdat::DataFrameRow, parnames::Vector{Symbol})

Extract covariates from a DataFrame row as a NamedTuple, using parameter names to determine which columns to extract.

# Arguments
- `subjdat`: A DataFrameRow containing covariate values
- `parnames`: Vector of parameter names (e.g., [:h12_Intercept, :h12_age, :h12_trt])

# Returns
- Empty NamedTuple() if no covariates
- NamedTuple with covariate values otherwise (e.g., (age=50, trt=1))

# Example
```julia
row = DataFrame(id=1, tstart=0.0, tstop=10.0, statefrom=1, stateto=2, obstype=1, age=50, trt=1)[1, :]
parnames = [:h12_Intercept, :h12_age, :h12_trt]
extract_covariates(row, parnames)  # Returns (age=50, trt=1)
```
"""
function extract_covariates(subjdat::Union{DataFrameRow,DataFrame}, parnames::Vector{Symbol})
    covar_names = extract_covar_names(parnames)
    
    if isempty(covar_names)
        return NamedTuple()
    end
    
    # Extract values from subjdat
    # Handle both DataFrameRow and DataFrame (for single-row DataFrame)
    if subjdat isa DataFrame
        @assert nrow(subjdat) == 1 "DataFrame must have exactly one row"
        subjdat_row = subjdat[1, :]
    else
        subjdat_row = subjdat
    end
    
    values = Tuple(subjdat_row[cname] for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(values)
end

"""
    generate_exponential_hazard(has_covariates::Bool, parnames::Vector{Symbol})

Generate runtime function for exponential hazard with name-based covariate matching.

# Arguments
- `has_covariates`: Whether covariates are present
- `parnames`: Vector of parameter names for name-based covariate access

# Returns
- `hazard_fn(t, pars, covars)`: Returns hazard rate on natural scale
- `cumhaz_fn(lb, ub, pars, covars)`: Returns cumulative hazard over [lb, ub]

# Parameters (LOG scale for baseline, natural scale for covariate effects)
- Without covariates: `pars = [log_rate]`, `covars = NamedTuple()`
  - hazard = exp(log_rate)
- With covariates: `pars = [log_baseline, coef1, coef2, ...]`, `covars = (age=x1, trt=x2, ...)`
  - hazard = exp(log_baseline + coef1*age + coef2*trt + ...)
"""
function generate_exponential_hazard(has_covariates::Bool, parnames::Vector{Symbol})
    if !has_covariates
        # No covariates: exp(log_rate) gives constant hazard rate
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                return exp(pars[1])  # exp(log_rate) = rate
            end
        ))
        
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                return exp(pars[1]) * (ub - lb)  # rate * duration
            end
        ))
    else
        # With covariates, name-based (uses NamedTuple)
        covar_names = extract_covar_names(parnames)
        
        # Build expressions for name-based access
        # e.g., pars[2] * covars.age + pars[3] * covars.trt
        linear_pred_terms = [:(pars[$(i+1)] * covars.$(covar_names[i])) 
                             for i in 1:length(covar_names)]
        linear_pred_expr = if isempty(linear_pred_terms)
            :(zero(eltype(pars)))
        else
            Expr(:call, :+, linear_pred_terms...)
        end
        
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_baseline = pars[1]
                linear_pred = $linear_pred_expr
                return exp(log_baseline + linear_pred)
            end
        ))
        
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_baseline = pars[1]
                linear_pred = $linear_pred_expr
                return exp(log_baseline + linear_pred) * (ub - lb)
            end
        ))
    end
    
    return hazard_fn, cumhaz_fn
end

"""
    generate_weibull_hazard(has_covariates::Bool, parnames::Vector{Symbol})

Generate runtime function for Weibull hazard with name-based covariate matching.

# Arguments
- `has_covariates`: Whether covariates are present
- `parnames`: Vector of parameter names for name-based covariate access

# Returns
- `hazard_fn(t, pars, covars)`: Returns hazard rate at time t
- `cumhaz_fn(lb, ub, pars, covars)`: Returns cumulative hazard over [lb, ub]

# Parameters (LOG scale for shape and scale, natural scale for covariate effects)
- Without covariates: `pars = [log_shape, log_scale]`, `covars = NamedTuple()`
  - h(t) = shape * scale * t^(shape-1), where shape=exp(log_shape), scale=exp(log_scale)
- With covariates: `pars = [log_shape, log_scale, coef1, coef2, ...]`, `covars = (age=x1, trt=x2, ...)`
  - Proportional hazards on scale: h(t) = shape * t^(shape-1) * scale * exp(β'X)
"""
function generate_weibull_hazard(has_covariates::Bool, parnames::Vector{Symbol})
    if !has_covariates
        # No covariates: h(t) = shape * scale * t^(shape-1)
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                return exp(log_shape + log_scale + expm1(log_shape) * log(t))
            end
        ))
        
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                scale = exp(log_scale)
                return scale * (ub^shape - lb^shape)
            end
        ))
    else
        # With covariates, name-based (uses NamedTuple)
        covar_names = extract_covar_names(parnames)
        
        # Build expressions for name-based access
        linear_pred_terms = [:(pars[$(i+2)] * covars.$(covar_names[i])) 
                             for i in 1:length(covar_names)]
        linear_pred_expr = if isempty(linear_pred_terms)
            :(zero(eltype(pars)))
        else
            Expr(:call, :+, linear_pred_terms...)
        end
        
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                linear_pred = $linear_pred_expr
                return exp(log_shape + expm1(log_shape) * log(t) + log_scale + linear_pred)
            end
        ))
        
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                scale = exp(log_scale)
                linear_pred = $linear_pred_expr
                return scale * exp(linear_pred) * (ub^shape - lb^shape)
            end
        ))
    end
    
    return hazard_fn, cumhaz_fn
end

"""
    generate_gompertz_hazard(has_covariates::Bool, parnames::Vector{Symbol})

Generate runtime function for Gompertz hazard with name-based covariate matching.

# Arguments
- `has_covariates`: Whether covariates are present
- `parnames`: Vector of parameter names for name-based covariate access

# Returns
- `hazard_fn(t, pars, covars)`: Returns hazard rate at time t
- `cumhaz_fn(lb, ub, pars, covars)`: Returns cumulative hazard over [lb, ub]

# Parameters (LOG scale for shape and scale, natural scale for covariate effects)
- Without covariates: `pars = [log_shape, log_scale]`, `covars = NamedTuple()`
  - h(t) = scale * shape * exp(shape*t), where shape=exp(log_shape), scale=exp(log_scale)
- With covariates: `pars = [log_shape, log_scale, coef1, coef2, ...]`, `covars = (age=x1, trt=x2, ...)`
  - h(t) = scale * shape * exp(shape*t + β'X)
"""
function generate_gompertz_hazard(has_covariates::Bool, parnames::Vector{Symbol})
    if !has_covariates
        # No covariates: h(t) = scale * shape * exp(shape*t)
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                return exp(log_scale + log_shape + shape * t)
            end
        ))
        
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                scale = exp(log_scale)
                if abs(shape) < 1e-10
                    return scale * (ub - lb)
                else
                    return scale * (exp(shape * ub) - exp(shape * lb))
                end
            end
        ))
    else
        # With covariates, name-based (uses NamedTuple)
        covar_names = extract_covar_names(parnames)
        
        # Build expressions for name-based access
        linear_pred_terms = [:(pars[$(i+2)] * covars.$(covar_names[i])) 
                             for i in 1:length(covar_names)]
        linear_pred_expr = if isempty(linear_pred_terms)
            :(zero(eltype(pars)))
        else
            Expr(:call, :+, linear_pred_terms...)
        end
        
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                linear_pred = $linear_pred_expr
                return exp(log_scale + log_shape + shape * t + linear_pred)
            end
        ))
        
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                scale = exp(log_scale)
                if abs(shape) < 1e-10
                    baseline_cumhaz = scale * (ub - lb)
                else
                    baseline_cumhaz = scale * (exp(shape * ub) - exp(shape * lb))
                end
                linear_pred = $linear_pred_expr
                return baseline_cumhaz * exp(linear_pred)
            end
        ))
    end
    
    return hazard_fn, cumhaz_fn
end

#=============================================================================
PHASE 2: Callable Hazard Interface
=============================================================================# 

"""
    (hazard::MarkovHazard)(t, pars, covars=Float64[])

Make MarkovHazard directly callable for hazard evaluation.
Returns hazard rate at time t (time parameter ignored for Markov processes).
"""
function (hazard::MarkovHazard)(t::Real, pars::AbstractVector, covars::Union{AbstractVector,NamedTuple}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    (hazard::SemiMarkovHazard)(t, pars, covars=Float64[])

Make SemiMarkovHazard directly callable for hazard evaluation.
Returns hazard rate at time t.
"""
function (hazard::SemiMarkovHazard)(t::Real, pars::AbstractVector, covars::Union{AbstractVector,NamedTuple}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    (hazard::SplineHazard)(t, pars, covars=Float64[])

Make SplineHazard directly callable for hazard evaluation.
Returns hazard rate at time t.
"""
function (hazard::SplineHazard)(t::Real, pars::AbstractVector, covars::Union{AbstractVector,NamedTuple}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    cumulative_hazard(hazard::Union{MarkovHazard,SemiMarkovHazard,SplineHazard}, lb, ub, pars, covars=Float64[])

Compute cumulative hazard over interval [lb, ub].
"""
function cumulative_hazard(hazard::Union{MarkovHazard,SemiMarkovHazard,SplineHazard}, 
                          lb::Real, ub::Real, 
                          pars::AbstractVector, 
                          covars::Union{AbstractVector,NamedTuple}=Float64[])
    return hazard.cumhaz_fn(lb, ub, pars, covars)
end

#=============================================================================
PHASE 2: Backward Compatibility Layer
=============================================================================# 

"""
Backward compatibility: Make new hazard types work with old call_haz() dispatch.

For new hazard types (MarkovHazard, SemiMarkovHazard, SplineHazard), we extract covariates
by name using the parnames field and the provided subjdat row.

# Note on Interface:
- Old hazard types: Expect `rowind` and have `.data` field
- New hazard types: Expect `subjdat::DataFrameRow` and use `parnames` to extract covariates
- Both interfaces supported for backward compatibility
"""

# MarkovHazard backward compatibility (with subjdat)
function call_haz(t, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::MarkovHazard; give_log = true)
    covars = extract_covariates(subjdat, hazard.parnames)
    haz = hazard(t, parameters, covars)
    give_log ? log(haz) : haz
end

function call_cumulhaz(lb, ub, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::MarkovHazard; give_log = true)
    covars = extract_covariates(subjdat, hazard.parnames)
    cumhaz = cumulative_hazard(hazard, lb, ub, parameters, covars)
    give_log ? log(cumhaz) : cumhaz
end

# SemiMarkovHazard backward compatibility (with subjdat)
function call_haz(t, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::SemiMarkovHazard; give_log = true)
    covars = extract_covariates(subjdat, hazard.parnames)
    haz = hazard(t, parameters, covars)
    give_log ? log(haz) : haz
end

function call_cumulhaz(lb, ub, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::SemiMarkovHazard; give_log = true)
    covars = extract_covariates(subjdat, hazard.parnames)
    cumhaz = cumulative_hazard(hazard, lb, ub, parameters, covars)
    give_log ? log(cumhaz) : cumhaz
end

# SplineHazard backward compatibility (with subjdat)
function call_haz(t, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::SplineHazard; give_log = true)
    covars = extract_covariates(subjdat, hazard.parnames)
    haz = hazard(t, parameters, covars)
    give_log ? log(haz) : haz
end

function call_cumulhaz(lb, ub, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::SplineHazard; give_log = true)
    covars = extract_covariates(subjdat, hazard.parnames)
    cumhaz = cumulative_hazard(hazard, lb, ub, parameters, covars)
    give_log ? log(cumhaz) : cumhaz
end

#=============================================================================
OLD DISPATCH-BASED FUNCTIONS (Will be deprecated after Phase 2)
=============================================================================#

"""
    survprob(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards; give_log = true) 

Return the survival probability over the interval [lb, ub].
"""
function survprob(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards; give_log = true) 

    # log total cumulative hazard
    log_survprob = -total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard, _hazards; give_log = false)

    # return survival probability or not
    give_log ? log_survprob : exp(log_survprob)
end

"""
    total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards; give_log = true) 

Return the log-total cumulative hazard out of a transient state over the interval [lb, ub].
"""
function total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards; give_log = true) 

    # log total cumulative hazard
    tot_haz = 0.0

    for x in _totalhazard.components
        tot_haz += call_cumulhaz(
                    lb,
                    ub, 
                    parameters[x],
                    subjdat_row,  # Pass the DataFrameRow directly
                    _hazards[x];
                    give_log = false)
    end
    
    # return the log, or not
    give_log ? log(tot_haz) : tot_haz
end

"""
    total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardAbsorbing, _hazards; give_log = true) 

Return zero log-total cumulative hazard over the interval [lb, ub] as the current state is absorbing.
"""
function total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardAbsorbing, _hazards; give_log = true) 

    # return 0 cumulative hazard
    give_log ? -Inf : 0

end

"""
    next_state_probs(t, scur, subjdat_row, parameters, hazards, totalhazards, tmat)

Return a vector ns_probs with probabilities of transitioning to each state based on hazards from current state. 

# Arguments 
- t: time at which hazards should be calculated
- scur: current state
- subjdat_row: DataFrame row containing subject covariates
- parameters: vector of vectors of model parameters
- hazards: vector of cause-specific hazards
- totalhazards: vector of total hazards
- tmat: transition matrix
"""
function next_state_probs(t, scur, subjdat_row, parameters, hazards, totalhazards, tmat)

    # initialize vector of next state transition probabilities
    ns_probs = zeros(size(tmat, 2))

    # indices for possible destination states
    trans_inds = findall(tmat[scur,:] .!= 0.0)

    if length(trans_inds) == 1
        ns_probs[trans_inds] .= 1.0
    else
        ns_probs[trans_inds] = softmax(map(x -> call_haz(t, parameters[x], subjdat_row, hazards[x]), totalhazards[scur].components))
    end

    # catch for numerical instabilities (weird edge case)
    if all(isnan.(ns_probs[trans_inds]))
        ns_probs[trans_inds] .= 1 / length(trans_inds)
    elseif any(isnan.(ns_probs[trans_inds]))
        pisnan = findall(isnan.(ns_probs[trans_inds]))
        if length(pisnan) == 1
            ns_probs[trans_inds][pisnan] = 1 - sum(ns_probs[trans_inds][Not(pisnan)])
        else
            ns_probs[trans_inds][pisnan] .= 1 - sum(ns_probs[trans_inds][Not(pisnan)])/length(pisnan)
        end        
    end

    # return the next state probabilities
    return ns_probs
end

########################################################
############# multistate markov process ################
###### transition intensities and probabilities ########
########################################################

"""
    compute_hazmat!(Q, parameters, hazards::Vector{T}, tpm_index::DataFrame, model_data::DataFrame) where T <: _Hazard

Fill in a matrix of transition intensities for a multistate Markov model.
"""
function compute_hazmat!(Q, parameters, hazards::Vector{T}, tpm_index::DataFrame, model_data::DataFrame) where T <: _Hazard

    # Get the DataFrameRow for covariate extraction
    subjdat_row = model_data[tpm_index.datind[1], :]
    
    # compute transition intensities
    for h in eachindex(hazards) 
        Q[hazards[h].statefrom, hazards[h].stateto] = 
            call_haz(
                tpm_index.tstart[1], 
                parameters[h],
                subjdat_row,
                hazards[h]; 
                give_log = false)
    end

    # set diagonal elements equal to the sum of off-diags
    Q[diagind(Q)] = -sum(Q, dims = 2)
end

"""
    compute_tmat!(P, Q, tpm_index::DataFrame, cache)

Calculate transition probability matrices for a multistate Markov process. 
"""
function compute_tmat!(P, Q, tpm_index::DataFrame, cache)

    for t in eachindex(P)
        copyto!(P[t], exponential!(Q * tpm_index.tstop[t], ExpMethodGeneric(), cache))
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

        # compute incidences
        for r in 1:n_intervals
            subjdat_row = subj_dat[interval_inds[r], :]
            incidences[r,h] = 
                survprobs[r,trans_inds[h]] * 
                quadgk(t -> (
                        call_haz(t, parameters[h], subjdat_row, hazards[h]; give_log = false) * 
                        survprob(subj_times[r], t, parameters, subjdat_row, totalhazards[statefrom], hazards; give_log = false)), 
                        subj_times[r], subj_times[r + 1])[1]
        end        
    end

    # return cumulative incidences
    return cumsum(incidences; dims = 1)
end

"""
    cumulative_incidence(t, model::MultistateProcess, statefrom, subj::Int64=1)

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
            incidences[r,h] = 
                survprobs[r] * 
                quadgk(t -> (
                        call_haz(t, parameters[hazinds[h]], subj_inds[interval_inds[r]], hazards[hazinds[h]]; give_log = false) * 
                        survprob(subj_times[r], t, parameters, subj_inds[interval_inds[r]], totalhazards[statefrom], hazards; give_log = false)), 
                        subj_times[r], subj_times[r + 1])[1]
        end        
    end

    # return cumulative incidences
    return cumsum(incidences; dims = 1)
end

"""
    compute_hazard(t, model::MultistateProcess, hazard::Symbol)

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

    # compute hazards
    hazards = zeros(Float64, length(t))
    for s in eachindex(t)
        # get row index
        rowind = findlast((model.data.id .== subj) .& (model.data.tstart .<= t[s]))

        # compute hazard
        hazards[s] = call_haz(t[s], model.parameters[hazind], rowind, model.hazards[hazind]; give_log = false)
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
        error("Lengths of tstart and tstop are not compatible.")
    end

    # get hazard index
    hazind = model.hazkeys[hazard]

    # compute hazards
    cumulative_hazards = zeros(Float64, length(tstart))
    for s in eachindex(tstart)

        # find times between tstart and tstop
        times = [tstart[s]; model.data.tstart[findall((model.data.id .== subj) .& (model.data.tstart .> tstart[s]) .& (model.data.tstart .< tstop[s]))]; tstop[s]]

        # initialize cumulative hazard
        chaz = 0.0

        # accumulate
        for i in 1:(length(times) - 1)
            # get row index
            rowind = findlast((model.data.id .== subj) .& (model.data.tstart .<= times[i]))

            # compute hazard
            chaz += call_cumulhaz(times[i], times[i+1], model.parameters[hazind], rowind, model.hazards[hazind]; give_log = false)
        end

        # save
        cumulative_hazards[s] = chaz
    end

    # return cumulative hazards
    return cumulative_hazards
end
