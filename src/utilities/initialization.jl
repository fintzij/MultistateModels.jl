# =============================================================================
# Parameter Initialization
# =============================================================================
# This module provides functions for initializing parameters in multistate models.
# The main entry points are:
#   - initialize_parameters!(): in-place parameter initialization
#   - initialize_parameters(): returns new model with initialized parameters
#   - set_crude_init!(): sets parameters to crude rates from data
#
# Initialization methods:
#   - :crude - Use crude log rates from observed data
#   - :markov - Fit Markov surrogate, use its log rates (semi-Markov only)
#   - :surrogate - Draw paths from surrogate, fit to exact data, use those parameters
#   - :auto - Select based on model type (:crude for Markov, :surrogate for semi-Markov)
# =============================================================================

"""
    set_crude_init!(model::MultistateProcess; constraints = nothing) -> Bool

Set parameter values to crude rate estimates from observed data.

Crude rates are computed as (number of transitions) / (time at risk) for each
transition type, then log-transformed for use as initial parameter values.

# Arguments
- `model::MultistateProcess`: Model to initialize
- `constraints`: Optional parameter constraints. If provided and the crude 
  initialization violates them, a warning is issued but parameters are still set.

# Returns
- `true` if initialization was applied and constraints (if any) are satisfied
- `false` if initialization was applied but violates constraints (warning issued)
"""
function set_crude_init!(model::MultistateProcess; constraints = nothing)
    
    crude_par = calculate_crude(model)

    for i in model.hazards
        # v0.3.0+: Pass crude rate directly (natural scale), not log-transformed
        set_par_to = init_par(i, crude_par[i.statefrom, i.stateto])
        set_parameters!(model, NamedTuple{(i.hazname,)}((set_par_to,)))
    end
    
    # Check constraints if provided
    if !isnothing(constraints)
        if !_check_constraints_satisfied(model, constraints)
            @warn "Crude initialization violates constraints. Parameters initialized but may need manual adjustment before fitting."
            return false
        end
    end
    
    return true
end

"""
    _check_constraints_satisfied(model, constraints) -> Bool

Check if current model parameters satisfy the given constraints.
"""
function _check_constraints_satisfied(model::MultistateProcess, constraints)
    _constraints = deepcopy(constraints)
    consfun = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_check)
    params = get_parameters_flat(model)
    cons_values = consfun(zeros(length(constraints.cons)), params, nothing)
    badcons = findall(cons_values .< constraints.lcons .|| cons_values .> constraints.ucons)
    return isempty(badcons)
end

# =============================================================================
# Helper Functions for Initialization
# =============================================================================

"""
    _is_degenerate_data(model::MultistateProcess) -> Bool

Check if model data has no observed transitions (all statefrom == stateto).

This detects "template" datasets used for simulation where all rows have
dummy states. Fitting surrogate MLEs on such data produces meaningless
parameters (often extreme rates).

Returns `true` if the data is degenerate and should not be used for 
surrogate-based initialization.
"""
function _is_degenerate_data(model::MultistateProcess)
    data = model.data
    # Check if any row has an actual transition (statefrom != stateto)
    return !any(data.statefrom .!= data.stateto)
end

"""
    _is_semimarkov(model::MultistateProcess) -> Bool

Check if model has any non-Markov (semi-Markov) hazards.
"""
function _is_semimarkov(model::MultistateProcess)
    !all(isa.(model.hazards, _MarkovHazard))
end

"""
    _interpolate_covariates!(exact_data::DataFrame, original_data::DataFrame)

Interpolate covariates from original panel data into exact transition data.

Uses piecewise constant interpolation: for each row in exact_data, finds the
original data row where id matches and tstart ≤ exact_row.tstart < tstop,
then copies all covariate columns.

# Arguments
- `exact_data::DataFrame`: Exact transition data (will be modified in place)
- `original_data::DataFrame`: Original panel data with covariates

# Returns
- The modified `exact_data` DataFrame with covariate columns added
"""
function _interpolate_covariates!(exact_data::DataFrame, original_data::DataFrame)
    # Identify covariate columns (all columns beyond standard 6)
    standard_cols = ["id", "tstart", "tstop", "statefrom", "stateto", "obstype"]
    covariate_cols = setdiff(names(original_data), standard_cols)
    
    if isempty(covariate_cols)
        return exact_data  # No covariates to interpolate
    end
    
    # Add covariate columns to exact_data
    for col in covariate_cols
        exact_data[!, col] = Vector{eltype(original_data[!, col])}(undef, nrow(exact_data))
    end
    
    # Group original data by subject for efficient lookup
    orig_grouped = groupby(original_data, :id)
    
    # Interpolate for each row in exact_data
    for i in 1:nrow(exact_data)
        subj_id = exact_data.id[i]
        t = exact_data.tstart[i]
        
        # Find original data for this subject
        subj_orig = orig_grouped[(id = subj_id,)]
        
        # Find interval containing t (tstart ≤ t < tstop, or last interval if t == final tstop)
        row_idx = findfirst(r -> subj_orig.tstart[r] <= t < subj_orig.tstop[r], 1:nrow(subj_orig))
        if isnothing(row_idx)
            # Edge case: t equals final tstop, use last interval
            row_idx = findlast(r -> subj_orig.tstop[r] >= t, 1:nrow(subj_orig))
        end
        
        @assert !isnothing(row_idx) "Could not find interval for subject $subj_id at time $t"
        
        # Copy covariates
        for col in covariate_cols
            exact_data[i, col] = subj_orig[row_idx, col]
        end
    end
    
    return exact_data
end

"""
    _interpolate_covariates_from_paths!(exact_data, original_data, samplepaths)

Interpolate covariates from original panel data into exact transition data when
using pseudo-subject IDs from multiple paths per subject.

Maps pseudo-subject IDs back to original subject IDs using the structure of
samplepaths, then performs piecewise constant covariate interpolation.

# Arguments
- `exact_data::DataFrame`: Exact transition data with pseudo-subject IDs (modified in place)
- `original_data::DataFrame`: Original panel data with covariates
- `samplepaths::Vector{Vector{SamplePath}}`: Original paths structure (subjects × paths)
  Used to map pseudo-IDs back to original subject IDs.

# Returns
- The modified `exact_data` DataFrame with covariate columns added
"""
function _interpolate_covariates_from_paths!(exact_data::DataFrame, 
                                              original_data::DataFrame,
                                              samplepaths::Vector{Vector{SamplePath}})
    # Identify covariate columns (all columns beyond standard 6)
    standard_cols = ["id", "tstart", "tstop", "statefrom", "stateto", "obstype"]
    covariate_cols = setdiff(names(original_data), standard_cols)
    
    if isempty(covariate_cols)
        return exact_data  # No covariates to interpolate
    end
    
    # Build mapping from pseudo-subject ID to original subject ID
    # Pseudo IDs were assigned sequentially: for subject i, path j, pseudo_id = (i-1)*npaths + j
    # where npaths may vary per subject, so we reconstruct from samplepaths
    pseudo_to_orig = Dict{Int, Int}()
    pseudo_id = 0
    for i in eachindex(samplepaths)
        orig_subj_id = samplepaths[i][1].subj  # Original subject ID from the first path
        for _ in samplepaths[i]
            pseudo_id += 1
            pseudo_to_orig[pseudo_id] = orig_subj_id
        end
    end
    
    # Add covariate columns to exact_data
    for col in covariate_cols
        exact_data[!, col] = Vector{eltype(original_data[!, col])}(undef, nrow(exact_data))
    end
    
    # Group original data by subject for efficient lookup
    orig_grouped = groupby(original_data, :id)
    
    # Interpolate for each row in exact_data
    for i in 1:nrow(exact_data)
        pseudo_subj_id = exact_data.id[i]
        orig_subj_id = pseudo_to_orig[pseudo_subj_id]
        t = exact_data.tstart[i]
        
        # Find original data for this subject
        subj_orig = orig_grouped[(id = orig_subj_id,)]
        
        # Find interval containing t (tstart ≤ t < tstop, or last interval if t == final tstop)
        row_idx = findfirst(r -> subj_orig.tstart[r] <= t < subj_orig.tstop[r], 1:nrow(subj_orig))
        if isnothing(row_idx)
            # Edge case: t equals final tstop, use last interval
            row_idx = findlast(r -> subj_orig.tstop[r] >= t, 1:nrow(subj_orig))
        end
        
        @assert !isnothing(row_idx) "Could not find interval for original subject $orig_subj_id (pseudo $pseudo_subj_id) at time $t"
        
        # Copy covariates
        for col in covariate_cols
            exact_data[i, col] = subj_orig[row_idx, col]
        end
    end
    
    return exact_data
end

"""
    _transfer_parameters!(target_model, source_model)

Transfer parameters from source model to target model by hazard name.

Uses the nested parameter structure to match hazards by name, avoiding
fragile index-based matching.

# Arguments
- `target_model::MultistateProcess`: Model to receive parameters
- `source_model`: Model to copy parameters from (fitted or unfitted)
"""
function _transfer_parameters!(target_model::MultistateProcess, source_model)
    source_nested = source_model.parameters.nested
    
    # Transfer parameters by hazard name (robust to ordering differences)
    for hazard in target_model.hazards
        hazname = hazard.hazname
        
        # Get source parameters for this hazard
        source_haz_params = source_nested[hazname]
        
        # Extract as flat vector: baseline first, then covariates
        baseline_vals = collect(values(source_haz_params.baseline))
        if haskey(source_haz_params, :covariates)
            covar_vals = collect(values(source_haz_params.covariates))
            param_vec = vcat(baseline_vals, covar_vals)
        else
            param_vec = baseline_vals
        end
        
        # Verify parameter count matches
        @assert length(param_vec) == hazard.npar_total "Parameter count mismatch for $hazname: source has $(length(param_vec)), target expects $(hazard.npar_total)"
        
        # Set parameters for this hazard
        set_parameters!(target_model, NamedTuple{(hazname,)}((param_vec,)))
    end
end

"""
    _init_from_surrogate_rates!(model; surrogate_constraints, surrogate_parameters)

Initialize parameters using rates from fitted Markov surrogate.

This is the implementation for `method = :markov`.

If the model already has a fitted Markov surrogate (created during model generation
or via `initialize_surrogate!`), it will be used directly. Otherwise, a new surrogate
will be fitted.

# Arguments
- `model::MultistateProcess`: Model to initialize
- `surrogate_constraints`: Constraints for surrogate fitting (only used if new fit needed)
- `surrogate_parameters`: Fixed surrogate parameters (optional)

# Note
As of v0.3.0, parameters are on natural scale (rates, not log rates).
"""
function _init_from_surrogate_rates!(model::MultistateProcess;
                                      surrogate_constraints = nothing,
                                      surrogate_parameters = nothing)
    # Check if model already has a fitted surrogate
    if !isnothing(model.markovsurrogate) && model.markovsurrogate.fitted
        surrog = model.markovsurrogate
    else
        # Fit Markov surrogate
        surrog = fit_surrogate(model; surrogate_constraints = surrogate_constraints, 
                               surrogate_parameters = surrogate_parameters, verbose = false)
    end

    for i in eachindex(model.hazards)
        hazard = model.hazards[i]
        hazname = hazard.hazname
        
        # v0.3.0+: Get natural-scale baseline parameter from surrogate
        # Surrogate always has exponential hazards with one baseline param (rate)
        surrog_nested = surrog.parameters.nested[hazname]
        surrog_rate = first(values(surrog_nested.baseline))
        
        # Initialize baseline params using the surrogate rate
        set_par_to = init_par(hazard, surrog_rate)

        # Copy covariate effects if there are any
        if hazard.has_covariates && haskey(surrog_nested, :covariates)
            surrog_covar_values = collect(values(surrog_nested.covariates))
            ncovar = length(surrog_covar_values)
            nbaseline = hazard.npar_baseline
            # Covariate params are at the end of set_par_to
            set_par_to[(nbaseline + 1):(nbaseline + ncovar)] .= surrog_covar_values
        end
        
        set_parameters!(model, NamedTuple{(hazname,)}((set_par_to,)))
    end
end

"""
    _init_from_surrogate_paths!(model, npaths; constraints, surrogate_constraints)

Initialize semi-Markov model parameters by simulating from surrogate and fitting to exact data.

# Algorithm
1. Draw `npaths` paths per subject from the Markov surrogate
2. Convert all paths to exact-observation data with pseudo-subject IDs
3. Interpolate covariates from original data
4. Create new model with exact data and same hazard specifications
5. Initialize and fit the exact-data model
6. Transfer fitted parameters to original model

# Arguments
- `model`: Semi-Markov model to initialize
- `npaths::Int`: Number of paths to draw per subject
- `constraints`: Parameter constraints for exact-data fitting
- `surrogate_constraints`: Constraints for surrogate fitting

# Note
Unlike MCEM, initialization does not use importance weights. We simply draw paths
from the surrogate and fit to those paths with uniform weights. The importance 
weights from draw_paths are computed but ignored for initialization.
"""
function _init_from_surrogate_paths!(model::MultistateProcess,
                                      npaths::Int;
                                      constraints = nothing,
                                      surrogate_constraints = nothing)
    # Step 1: Draw paths from surrogate (will fit surrogate if needed)
    # Note: draw_paths computes importance weights, but we ignore them for initialization
    result = draw_paths(model; npaths = npaths, paretosmooth = false)
    
    # Step 2: Convert all paths to exact data with pseudo-subject IDs
    # result.samplepaths is Vector{Vector{SamplePath}} (subjects × paths per subject)
    nsubj = length(result.samplepaths)
    
    # Flatten paths and create pseudo-subject IDs (sequential 1:n)
    all_paths = SamplePath[]
    pseudo_id = 0
    
    for i in 1:nsubj
        subj_paths = result.samplepaths[i]
        for j in eachindex(subj_paths)
            pseudo_id += 1
            path = subj_paths[j]
            push!(all_paths, SamplePath(pseudo_id, path.times, path.states))
        end
    end
    
    # Convert to exact data
    exact_data = paths_to_dataset(all_paths)
    
    # Check if we have any transitions
    if nrow(exact_data) == 0
        @warn "No transitions in simulated paths; falling back to :markov initialization"
        _init_from_surrogate_rates!(model; surrogate_constraints = surrogate_constraints)
        return
    end
    
    # Step 3: Interpolate covariates from original data
    # Map pseudo-subject IDs back to original subject IDs for covariate interpolation
    _interpolate_covariates_from_paths!(exact_data, model.data, result.samplepaths)
    
    # Step 4: Create model with exact data (same hazard specs)
    # IMPORTANT: initialize=false to avoid recursive initialization
    # All paths have uniform weight (no importance weighting for initialization)
    exact_model = multistatemodel(model.modelcall.hazards...; 
                                   data = exact_data,
                                   initialize = false,
                                   SubjectWeights = nothing,  # Uniform weights
                                   CensoringPatterns = nothing)
    
    # Step 5: Initialize exact model with :crude method (quick starting point)
    set_crude_init!(exact_model)
    
    # Step 6: Fit exact model (fast - exact likelihood, no MCEM)
    # Use select_lambda=:none to skip automatic λ selection during initialization
    # (λ selection on small simulated samples is unreliable and expensive)
    exact_fitted = fit(exact_model; constraints = constraints, 
                       compute_vcov = false, compute_ij_vcov = false, 
                       compute_jk_vcov = false, verbose = false,
                       select_lambda = :none)
    
    # Step 7: Transfer parameters to original model
    _transfer_parameters!(model, exact_fitted)
end

# =============================================================================
# Main Entry Points
# =============================================================================

"""
    initialize_parameters!(model::MultistateProcess; method=:auto, npaths=50, ...)

Initialize model parameters using the specified method.

# Arguments
- `model::MultistateProcess`: Model to initialize (modified in place)
- `constraints`: Parameter constraints for fitting (used with :surrogate method)
- `surrogate_constraints`: Constraints for surrogate fitting
- `surrogate_parameters`: Fixed surrogate parameters (for :markov method)
- `method::Symbol = :auto`: Initialization method
  - `:auto` - Select based on model type (:crude for Markov, :surrogate for semi-Markov)
  - `:crude` - Use crude rates from observed data
  - `:markov` - Fit Markov surrogate, use its log rates
  - `:surrogate` - Draw paths from surrogate, fit to exact data, use those parameters
- `npaths::Int = 50`: Number of paths per subject for :surrogate method

# Examples
```julia
# Auto-select method based on model type
initialize_parameters!(model)

# Force crude initialization
initialize_parameters!(model; method = :crude)

# Use surrogate simulation with more paths
initialize_parameters!(model; method = :surrogate, npaths = 50)
```

See also: [`initialize_parameters`](@ref), [`set_crude_init!`](@ref)
"""
function initialize_parameters!(model::MultistateProcess;
                                constraints = nothing,
                                surrogate_constraints = nothing,
                                surrogate_parameters = nothing,
                                method::Symbol = :auto,
                                npaths::Int = 10)
    
    # Validate npaths
    npaths > 0 || throw(ArgumentError("npaths must be positive, got $npaths"))
    
    # Check for degenerate data (no transitions observed)
    # This detects "template" data where all statefrom == stateto, which would cause
    # surrogate MLE fitting to produce meaningless parameters
    if _is_degenerate_data(model)
        @warn "No transitions observed in data (all statefrom == stateto). " *
              "Falling back to :crude initialization. If this is template data for " *
              "simulation, consider using `initialize=false` when creating the model." maxlog=1
        set_crude_init!(model; constraints = constraints)
        return nothing
    end
    
    # Check if model is phase-type (has phasetype_expansion)
    is_phasetype = has_phasetype_expansion(model)
    
    # Resolve :auto method based on model type
    # Phase-type models are Markov models, so they use :crude like other Markov models
    actual_method = if method == :auto
        if is_phasetype || !_is_semimarkov(model)
            :crude  # Markov and phase-type models default to :crude
        else
            :surrogate  # Semi-Markov models default to :surrogate
        end
    else
        method
    end
    
    # Validate method
    if is_phasetype
        # Phase-type models only support :crude (they are Markov)
        actual_method == :crude ||
            throw(ArgumentError("Phase-type models only support :crude initialization (they are Markov models), got :$method"))
    else
        actual_method in (:crude, :markov, :surrogate) || 
            throw(ArgumentError("method must be :auto, :crude, :markov, or :surrogate, got :$method"))
    end
    
    # Check constraints compatibility
    if actual_method == :markov && !isnothing(constraints) && isnothing(surrogate_constraints)
        @warn "Constraints provided but surrogate_constraints not specified for :markov method"
    end
    
    # Dispatch on method
    if actual_method == :crude
        set_crude_init!(model; constraints = constraints)  # Will warn if constraints violated
        
    elseif actual_method == :markov
        _init_from_surrogate_rates!(model; 
                                     surrogate_constraints = surrogate_constraints,
                                     surrogate_parameters = surrogate_parameters)
        
    elseif actual_method == :surrogate
        # Check if model supports surrogate path initialization:
        # 1. Must have semi-Markov hazards
        # 2. Must be a model type that has panel/censored observations (not exact data)
        supports_surrogate = _is_semimarkov(model) && 
                             isa(model, MultistateProcess)
        
        if !supports_surrogate
            if !_is_semimarkov(model)
                @warn "Using :surrogate method on Markov model; this is equivalent to :crude"
            else
                @warn "Using :surrogate method on exact-data model; falling back to :crude"
            end
            set_crude_init!(model; constraints = constraints)
        else
            _init_from_surrogate_paths!(model, npaths; 
                                         constraints = constraints,
                                         surrogate_constraints = surrogate_constraints)
        end
    end
    
    return nothing
end

"""
    initialize_parameters(model::MultistateProcess; method=:auto, npaths=10, ...) -> MultistateProcess

Return a new model with initialized parameters (non-mutating version).

See [`initialize_parameters!`](@ref) for argument descriptions.
"""
function initialize_parameters(model::MultistateProcess; 
                               constraints = nothing, 
                               surrogate_constraints = nothing, 
                               surrogate_parameters = nothing,
                               method::Symbol = :auto,
                               npaths::Int = 10)
    model_copy = deepcopy(model)
    initialize_parameters!(model_copy; 
                           constraints = constraints,
                           surrogate_constraints = surrogate_constraints,
                           surrogate_parameters = surrogate_parameters,
                           method = method,
                           npaths = npaths)
    return model_copy
end

"""
    compute_suff_stats(dat, tmat, SubjectWeights)

Return a matrix in same format as tmat with observed transition counts, and a vector of time spent in each state. Used for checking data and calculating crude initialization rates.
"""
function compute_suff_stats(data, tmat, SubjectWeights)
    # matrix to store number of transitions from state r to state s, stored using same convention as model.tmat
    n_rs = zeros(size(tmat))

    # vector of length equal to number of states to store time spent in state
    T_r = zeros(size(tmat)[1])

    for rd in eachrow(data)
        # loop through data rows
        # track how much time has been spent in state
        #T_r_accumulator += r.tstop - r.tstart
        if rd.statefrom>0
            T_r[rd.statefrom] += (rd.tstop - rd.tstart) * SubjectWeights[rd.id]
        end        
        # if a state transition happens then increment transition by 1 in n_rs
        if rd.statefrom != rd.stateto
            if rd.statefrom>0 && rd.stateto>0
                n_rs[rd.statefrom, rd.stateto] += SubjectWeights[rd.id]
            end
        end
    end

    return n_rs, T_r
end

"""
    compute_number_transitions(data, tmat)

Return a matrix in same format as tmat with observed transition counts, and a vector of time spent in each state. Used for checking data and calculating crude initialization rates.
"""
function compute_number_transitions(data, tmat)
    # matrix to store number of transitions from state r to state s, stored using same convention as model.tmat
    n_rs = zeros(size(tmat))

    for rd in eachrow(data)
        # loop through data rows
        if rd.statefrom != rd.stateto
            # if a state transition happens then increment transition by 1 in n_rs
            if rd.statefrom>0 && rd.stateto>0
                n_rs[rd.statefrom, rd.stateto] += 1
            end
        end
    end

    return n_rs
end

"""
    calculate_crude(model::MultistateProcess)

Return a matrix with same dimensions as model.tmat, but row i column j entry is number of transitions from state i to state j divided by time spent in state i, then log transformed. In other words, a faster version of log exponential rates that fit_exact would return.

Accept a MultistateProcess object.
"""
function calculate_crude(model::MultistateProcess)
    # n_rs is matrix like tmat except each entry is number of transitions from state r to s
    # T_r is vector of length number of states
    n_rs, T_r = compute_suff_stats(model.data, model.tmat, model.SubjectWeights)

    # crude fix to avoid taking the log of zero (for pairs of states with no observed transitions) by turning zeros to 0.5. Also check_data!() should have thrown an error during model generation if this is the case.
    n_rs = max.(n_rs, 0.5)
    
    # give a reasonable sojourn time to states never visited
    T_r[T_r .== 0] .= mean(model.data.tstop - model.data.tstart)

# UNALLOWED TRANSITIONS SHOULD BE ZERO
# ALLOWED TRANSITIONS WITH ZERO SHOULD BE 0.5
# DIAGONALS SHOULD BE ROW SUMS, WHICH INCLUDE ANY 0.5

    # return log of the rate
    crude_mat = n_rs ./ T_r
    
    crude_mat[findall(model.tmat .== 0)] .= 0

    for i in 1:length(T_r)
        crude_mat[i, i] =  -sum(crude_mat[i, Not(i)])
    end
    
    return crude_mat
    #give_log ? log.(crude_mat) : crude_mat 
end

"""
Initialize parameters for new hazard types (MarkovHazard, SemiMarkovHazard, RuntimeSplineHazard).

Dispatches based on family and has_covariates to determine parameter initialization.

Note: As of v0.3.0, parameters are initialized on NATURAL scale (not log-transformed).
The input `crude_rate` should be a positive rate value (not log-transformed).
"""
function init_par(hazard::Union{MarkovHazard,SemiMarkovHazard,_SplineHazard}, crude_rate=1.0)
    family = hazard.family
    has_covs = hazard.has_covariates
    ncovar = hazard.npar_total - hazard.npar_baseline
    
    if family == :exp
        # Exponential: [rate] or [rate, β1, β2, ...]
        return has_covs ? vcat(crude_rate, zeros(ncovar)) : [crude_rate]
        
    elseif family == :wei
        # Weibull: [shape, scale] or [shape, scale, β1, β2, ...]
        # Initialize shape=1 to start as exponential-like
        baseline = [1.0, crude_rate]  # shape=1, scale
        return has_covs ? vcat(baseline, zeros(ncovar)) : baseline
        
    elseif family == :gom
        # Gompertz: [shape, rate] or [shape, rate, β1, β2, ...]
        # shape is unconstrained (can be positive, negative, or zero)
        # rate is positive
        # Initialize shape=0 so hazard starts as exponential: h(t) = rate * exp(0*t) = rate
        baseline = [0.0, crude_rate]  # shape=0, rate
        return has_covs ? vcat(baseline, zeros(ncovar)) : baseline
        
    elseif family == :sp
        # Spline: all coefficients non-negative (natural scale)
        # Initialize all to same value so hazard starts as approximately constant
        nbasis = hazard.npar_baseline
        baseline = fill(crude_rate, nbasis)  # Natural scale
        return has_covs ? vcat(baseline, zeros(ncovar)) : baseline
        
    else
        throw(ArgumentError("Unknown hazard family: $family. Supported: :exp, :wei, :gom, :sp"))
    end
end
