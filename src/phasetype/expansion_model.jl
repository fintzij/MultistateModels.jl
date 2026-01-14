# =============================================================================
# Phase-Type Model Building
# =============================================================================
#
# Build MultistateModel with phase-type expansion from user hazard specs.
# Called by multistatemodel() when any hazard is :pt.
#
# Contents:
# - _build_phasetype_model_from_hazards: Main model building function
# - _compute_ordering_reference: Compute reference covariate values for ordering constraints
# - _build_expanded_parameters: Build parameters for expanded hazards
# - _build_original_parameters: Extract initial parameters from user hazards
# - _extract_original_natural_vector: Extract parameter vector from hazard
# - _count_covariates: Count covariates in formula
# - _count_hazard_parameters: Count parameters in hazard specification
# - _merge_censoring_patterns_with_shift: Merge user and phase censoring patterns
#
# =============================================================================

using Statistics: mean, median

"""
    _compute_ordering_reference(ordering_at, data, covariate_names) -> Dict{Symbol, Float64}

Compute reference covariate values for eigenvalue ordering constraints.

# Arguments
- `ordering_at::Union{Symbol, NamedTuple}`: How to compute reference values
  - `:reference`: returns empty Dict (linear constraints at x=0)
  - `:mean`: compute mean of each covariate from data
  - `:median`: compute median of each covariate from data
  - `NamedTuple`: use explicit values provided by user
- `data::DataFrame`: Data to compute means/medians from
- `covariate_names::Vector{Symbol}`: Names of covariates in the model

# Returns
- `Dict{Symbol, Float64}`: Reference values for each covariate. Empty dict signals
  reference (x=0) ordering, which produces linear constraints.

# Example
```julia
_compute_ordering_reference(:reference, data, [:age, :trt])  # Dict()
_compute_ordering_reference(:mean, data, [:age, :trt])      # Dict(:age => 50.0, :trt => 0.4)
_compute_ordering_reference((age=50.0,), data, [:age])      # Dict(:age => 50.0)
```
"""
function _compute_ordering_reference(ordering_at::Union{Symbol, NamedTuple},
                                      data::DataFrame,
                                      covariate_names::Vector{Symbol})::Dict{Symbol, Float64}
    if ordering_at === :reference
        return Dict{Symbol, Float64}()
    elseif ordering_at === :mean
        result = Dict{Symbol, Float64}()
        for cov in covariate_names
            if hasproperty(data, cov)
                result[cov] = mean(skipmissing(data[!, cov]))
            else
                @warn "Covariate $cov not found in data, using 0.0 for ordering reference"
                result[cov] = 0.0
            end
        end
        return result
    elseif ordering_at === :median
        result = Dict{Symbol, Float64}()
        for cov in covariate_names
            if hasproperty(data, cov)
                result[cov] = median(skipmissing(data[!, cov]))
            else
                @warn "Covariate $cov not found in data, using 0.0 for ordering reference"
                result[cov] = 0.0
            end
        end
        return result
    elseif ordering_at isa NamedTuple
        result = Dict{Symbol, Float64}()
        nt_keys = keys(ordering_at)
        for cov in covariate_names
            if cov in nt_keys
                result[cov] = ordering_at[cov]
            elseif cov in [:Intercept, Symbol("(Intercept)")]
                # Skip intercept - it's not a covariate
                continue
            else
                throw(ArgumentError("ordering_at NamedTuple missing covariate $cov. " *
                    "Provide values for all model covariates: $covariate_names"))
            end
        end
        return result
    else
        throw(ArgumentError("ordering_at must be :reference, :mean, :median, or NamedTuple, got $ordering_at"))
    end
end

"""
    _extract_covariate_names(hazards) -> Vector{Symbol}

Extract unique covariate names from a collection of hazards.
Excludes the intercept term.
"""
function _extract_covariate_names(hazards::Vector{<:HazardFunction})::Vector{Symbol}
    covar_names = Set{Symbol}()
    for h in hazards
        if hasproperty(h, :hazard) && h.hazard isa StatsModels.FormulaTerm
            formula = h.hazard
            # Get the RHS - this can be various StatsModels types
            rhs = formula.rhs
            _extract_terms_recursive!(covar_names, rhs)
        end
    end
    return collect(covar_names)
end

"""
    _extract_terms_recursive!(names::Set{Symbol}, term)

Recursively extract covariate symbol names from a StatsModels term structure.
"""
function _extract_terms_recursive!(names::Set{Symbol}, term)
    if term isa StatsModels.Term
        sym = term.sym
        if sym ∉ (:1, Symbol("(Intercept)"))
            push!(names, sym)
        end
    elseif term isa StatsModels.InteractionTerm
        for subterm in term.terms
            _extract_terms_recursive!(names, subterm)
        end
    elseif term isa Tuple
        for subterm in term
            _extract_terms_recursive!(names, subterm)
        end
    elseif term isa StatsModels.InterceptTerm || term isa StatsModels.ConstantTerm
        # Skip intercept/constant terms
    elseif term isa StatsModels.FunctionTerm
        # FunctionTerm wraps a function call like s(x) or te(x,y)
        # The args contain the actual terms
        for arg in term.args
            _extract_terms_recursive!(names, arg)
        end
    # Skip other term types (ConstantTerm, MatrixTerm, etc.)
    end
end

"""
    _build_phasetype_model_from_hazards(hazards, data; kwargs...) -> MultistateModel

Build a MultistateModel with phase-type expansion from user hazard specs.

Called by `multistatemodel()` when any hazard is `:pt`. Builds expanded state space,
expands hazards and data, and stores metadata in `phasetype_expansion` field.

# Arguments
- `hazards::Tuple`: User hazard specifications (some :pt)
- `data::DataFrame`: Observation data
- `n_phases::Union{Nothing,Dict{Int,Int}}`: Phases per state (default: 2)
- `coxian_structure::Symbol`: `:unstructured` or `:sctp`
- `ordering_at::Union{Symbol, NamedTuple}`: Where to enforce eigenvalue ordering
- Other kwargs: `constraints`, `SubjectWeights`, `CensoringPatterns`, etc.

# Returns
`MultistateModel` on expanded state space with `phasetype_expansion` metadata.

See also: [`PhaseTypeExpansion`](@ref), [`build_phasetype_mappings`](@ref)
"""
function _build_phasetype_model_from_hazards(hazards::Tuple{Vararg{HazardFunction}};
                                              data::DataFrame,
                                              constraints = nothing,
                                              initialize::Bool = true,
                                              n_phases::Union{Nothing, Dict{Int,Int}} = nothing,
                                              coxian_structure::Symbol = :unstructured,
                                              ordering_at::Union{Symbol, NamedTuple} = :reference,
                                              SubjectWeights::Union{Nothing,Vector{Float64}} = nothing,
                                              ObservationWeights::Union{Nothing,Vector{Float64}} = nothing,
                                              CensoringPatterns::Union{Nothing,Matrix{<:Real}} = nothing,
                                              EmissionMatrix::Union{Nothing,Matrix{Float64}} = nothing,
                                              verbose::Bool = false)
    
    hazards_vec = collect(hazards)
    
    # Step 1: Enumerate hazards and build original tmat
    hazinfo = enumerate_hazards(hazards...)
    hazards_ordered = hazards_vec[hazinfo.order]
    select!(hazinfo, Not(:order))
    tmat_original = create_tmat(hazinfo)
    n_states = size(tmat_original, 1)
    
    if verbose
        println("Building phase-type model...")
        println("  Original states: $n_states")
    end
    
    # Step 1b: Validate and process n_phases
    # Identify states with :pt hazards
    pt_states = Set{Int}()
    for h in hazards_ordered
        if h isa PhaseTypeHazard
            push!(pt_states, h.statefrom)
        end
    end
    
    # Build n_phases_per_state from model-level n_phases dict
    n_phases_per_state = ones(Int, n_states)  # Default: 1 phase for non-pt states
    
    if isnothing(n_phases) || isempty(n_phases)
        # Legacy path: use n_phases from PhaseTypeHazard
        for h in hazards_ordered
            if h isa PhaseTypeHazard
                s = h.statefrom
                n_phases_per_state[s] = max(n_phases_per_state[s], h.n_phases)
            end
        end
    else
        # New path: use model-level n_phases dict
        # Validate: all pt states must be in dict
        for s in pt_states
            if !haskey(n_phases, s)
                throw(ArgumentError("State $s has :pt hazards but is not in n_phases dict. " *
                      "Specify n_phases = Dict($s => k) where k is the number of phases."))
            end
        end
        
        # Validate: dict shouldn't have states without :pt hazards
        for (s, k) in n_phases
            if s ∉ pt_states
                throw(ArgumentError("n_phases specifies $k phases for state $s, but state $s has no :pt hazards."))
            end
            if k < 1
                throw(ArgumentError("n_phases[$s] must be ≥ 1, got $k"))
            end
            # Coerce n_phases = 1 to exponential (no warning, just do it)
            n_phases_per_state[s] = k
        end
    end
    
    # Step 1c: Check for mixed hazard types from states with :pt hazards
    for s in pt_states
        outgoing_hazards = filter(h -> h.statefrom == s, hazards_ordered)
        non_pt_hazards = filter(h -> !(h isa PhaseTypeHazard), outgoing_hazards)
        if !isempty(non_pt_hazards)
            non_pt_names = [Symbol("h$(h.statefrom)$(h.stateto)") for h in non_pt_hazards]
            @warn "State $s has :pt hazards but also has non-:pt hazards: $non_pt_names. " *
                  "All outgoing hazards from state $s will be treated as phase-type."
        end
    end
    
    if verbose
        println("  Phases per state: $n_phases_per_state")
        println("  Coxian structure: $coxian_structure")
    end
    
    # Step 2: Build mappings from n_phases_per_state
    mappings = build_phasetype_mappings(hazards_ordered, tmat_original, n_phases_per_state)
    
    if verbose
        println("  Expanded states: $(mappings.n_expanded)")
        println("  Phases per state: $(mappings.n_phases_per_state)")
    end
    
    # Step 3: Expand hazards to runtime representation
    # hazard_to_params_idx maps each hazard index to its parameter index
    # (shared hazards point to the same parameter index as their base)
    expanded_hazards, expanded_params_list, hazard_to_params_idx = expand_hazards_for_phasetype(
        hazards_ordered, mappings, data
    )
    
    if verbose
        println("  Expanded hazards: $(length(expanded_hazards))")
    end
    
    # Step 4: Expand data for phase uncertainty
    expanded_data, phase_censoring_patterns, original_row_map = expand_data_for_phasetype_fitting(data, mappings)
    
    # Step 5: Build expanded hazkeys
    # Use hazard_to_params_idx to map to parameter indices (handles shared hazards)
    expanded_hazkeys = Dict{Symbol, Int64}()
    for (idx, h) in enumerate(expanded_hazards)
        # Use the parameter index, not the hazard index
        # For shared hazards, this will be the base hazard's parameter index
        expanded_hazkeys[h.hazname] = hazard_to_params_idx[idx]
    end
    
    # Step 6: Build parameters structure
    expanded_parameters = _build_expanded_parameters(expanded_params_list, expanded_hazkeys, expanded_hazards, hazard_to_params_idx)
    
    # Step 7: Get subject indices on expanded data
    subjinds, nsubj = get_subjinds(expanded_data)
    
    # Step 8: Handle weights
    SubjectWeights, ObservationWeights = check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj)
    
    # Step 9: Prepare censoring patterns for expanded space
    if CensoringPatterns === nothing
        CensoringPatterns_expanded = phase_censoring_patterns
    else
        # Merge user patterns with phase uncertainty patterns
        # Pass original data to identify which obstypes are user-supplied
        CensoringPatterns_expanded, expanded_data = _merge_censoring_patterns_with_shift(
            CensoringPatterns, phase_censoring_patterns, mappings, expanded_data, 
            data, original_row_map
        )
    end
    
    # Validate inputs (pass phase_to_state to avoid spurious warnings about missing phase transitions)
    CensoringPatterns_final = _prepare_censoring_patterns(CensoringPatterns_expanded, mappings.n_expanded)
    _validate_inputs!(expanded_data, mappings.expanded_tmat, CensoringPatterns_final, SubjectWeights, ObservationWeights; 
                      verbose = verbose, phase_to_state = mappings.phase_to_state)
    
    # Step 10: Build emission matrix
    if !isnothing(EmissionMatrix)
        # Expand user-supplied emission matrix to expanded state space
        expanded_emat = expand_emission_matrix(EmissionMatrix, original_row_map, mappings)
        emat = build_emat(expanded_data, CensoringPatterns_final, expanded_emat, mappings.expanded_tmat)
    else
        emat = build_emat(expanded_data, CensoringPatterns_final, nothing, mappings.expanded_tmat)
    end
    
    # Step 11: Build total hazards on expanded space
    expanded_totalhazards = build_totalhazards(expanded_hazards, mappings.expanded_tmat)
    
    # Step 11b: Generate auto constraints (SCTP, ordering, C1)
    final_constraints = constraints
    
    # SCTP constraints: ensure P(dest | leaving state) is constant across phases
    # Applied for :sctp, :sctp_increasing, and :sctp_decreasing
    if coxian_structure in (:sctp, :sctp_increasing, :sctp_decreasing)
        sctp_constraints = _generate_sctp_constraints(expanded_hazards, mappings, pt_states)
        if !isnothing(sctp_constraints)
            if verbose
                println("  Generated $(length(sctp_constraints.cons)) SCTP constraints")
            end
            final_constraints = _merge_constraints(final_constraints, sctp_constraints)
        end
    end
    
    # Ordering constraints: enforce eigenvalue ordering
    # Applied for :sctp_decreasing and :sctp_increasing
    if coxian_structure in (:sctp_decreasing, :sctp_increasing)
        # Determine ordering direction
        # :sctp_increasing → ν₁ ≤ ν₂ ≤ ... ≤ νₙ (late exits more likely)
        # :sctp_decreasing → ν₁ ≥ ν₂ ≥ ... ≥ νₙ (early exits more likely)
        ordering_direction = coxian_structure === :sctp_increasing ? :increasing : :decreasing
        
        # Compute reference covariate values for ordering constraints
        covariate_names = _extract_covariate_names(hazards_ordered)
        ordering_reference = _compute_ordering_reference(ordering_at, data, covariate_names)
        
        if verbose && !isempty(ordering_reference)
            println("  Ordering constraints at reference: $ordering_reference")
        end
        
        ordering_constraints = _generate_ordering_constraints(
            expanded_hazards, mappings, pt_states, hazards_ordered, ordering_reference, ordering_direction
        )
        if !isnothing(ordering_constraints)
            if verbose
                direction_str = ordering_direction === :increasing ? "increasing" : "decreasing"
                println("  Generated $(length(ordering_constraints.cons)) eigenvalue ordering constraints ($direction_str)")
            end
            final_constraints = _merge_constraints(final_constraints, ordering_constraints)
        end
    end
    
    # C1 constraints: equality constraints for shared covariate effects
    c1_constraints = _generate_c1_constraints(expanded_hazards, hazards_ordered)
    if !isnothing(c1_constraints)
        if verbose
            println("  Generated $(length(c1_constraints.cons)) C1 covariate equality constraints")
        end
        final_constraints = _merge_constraints(final_constraints, c1_constraints)
    end
    
    # Step 12: Store modelcall
    modelcall = (
        hazards = hazards, 
        data = data, 
        constraints = final_constraints, 
        SubjectWeights = SubjectWeights, 
        ObservationWeights = ObservationWeights, 
        CensoringPatterns = CensoringPatterns, 
        EmissionMatrix = EmissionMatrix,
        is_phasetype = true  # Mark this as a phase-type model for fitted model accessors
    )
    
    # Step 13: Build original parameters structure (for user-facing API)
    original_parameters = _build_original_parameters(hazards_ordered, data)
    
    # Step 14: Create PhaseTypeExpansion metadata
    phasetype_expansion = PhaseTypeExpansion(
        mappings,
        data,              # original data
        tmat_original,     # original tmat
        hazards_ordered,   # original hazard specs
        original_parameters
    )
    
    # Step 15: Assemble the MultistateModel on expanded space using _assemble_model
    # This ensures bounds are generated correctly, matching the pattern for regular models
    components = (
        data = expanded_data,
        parameters = expanded_parameters,
        hazards = expanded_hazards,
        totalhazards = expanded_totalhazards,
        tmat = mappings.expanded_tmat,
        emat = emat,
        hazkeys = expanded_hazkeys,
        subjinds = subjinds,
        SubjectWeights = SubjectWeights,
        ObservationWeights = ObservationWeights,
        CensoringPatterns = CensoringPatterns_final,
    )
    
    # Phase-type models are always Markov in expanded space with panel data
    model = _assemble_model(:panel, :markov, components, nothing, modelcall;
                            phasetype_expansion = phasetype_expansion)
    
    # Step 16: Initialize parameters (phase-type models use :crude like other Markov models)
    if initialize
        initialize_parameters!(model; constraints = final_constraints)
    end
    
    if verbose
        println("  MultistateModel (phase-type) created successfully")
    end
    
    return model
end

"""
    _build_expanded_parameters(params_list, hazkeys, hazards, hazard_to_params_idx)

Build ParameterHandling-compatible parameter structure for expanded hazards.

# Arguments
- `params_list`: Vector of parameter vectors, one per unique parameter set
- `hazkeys`: Dict mapping hazard name to parameter index
- `hazards`: Vector of all expanded hazards (may have more entries than params_list for shared hazards)
- `hazard_to_params_idx`: Maps each hazard index to its parameter index (default: identity mapping)

# Note on C1 Covariate Constraints
When using `covariate_constraints=:homogeneous`, multiple hazards will have the same covariate parameter
name (e.g., `h12_x` for both `h12_a` and `h12_b`). The current implementation stores these
as separate entries in the nested structure. During optimization, equality constraints should
be used to enforce that these parameters remain equal. Use `set_parameters!` to ensure
consistent values when setting parameters manually.
"""
function _build_expanded_parameters(params_list::Vector{Vector{Float64}}, 
                                     hazkeys::Dict{Symbol, Int64}, 
                                     hazards::Vector{<:_Hazard},
                                     hazard_to_params_idx::Vector{Int} = collect(1:length(hazards)))
    # Build a reverse mapping: for each parameter index, find a hazard that uses it
    # (for shared hazards, any of them will have the same parnames/npar_baseline)
    params_idx_to_hazard_idx = Dict{Int, Int}()
    for (haz_idx, params_idx) in enumerate(hazard_to_params_idx)
        if !haskey(params_idx_to_hazard_idx, params_idx)
            params_idx_to_hazard_idx[params_idx] = haz_idx
        end
    end
    
    # Build nested parameters structure per unique parameter set
    params_nested_pairs = [
        let haz_idx = params_idx_to_hazard_idx[params_idx]
            hazname => build_hazard_params(params_list[params_idx], hazards[haz_idx].parnames, hazards[haz_idx].npar_baseline, length(params_list[params_idx]))
        end
        for (hazname, params_idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_nested = NamedTuple(params_nested_pairs)
    
    # Create ReConstructor for AD-compatible flatten/unflatten
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    return (
        flat = params_flat,
        nested = params_nested,
        reconstructor = reconstructor
    )
end

"""
    _build_original_parameters(hazards, data)

Build parameter structure for the original (user-facing) hazard specifications.

This is used for storing and reporting parameters in terms of the user's
specification, not the expanded internal representation.

Returns a structure matching the standard model parameters format:
- `flat`: Vector of log-scale baseline params + covariate coefficients
- `nested`: NamedTuple per hazard with `baseline` and `covariates` NamedTuples
- `natural`: NamedTuple per hazard with simple Vector{Float64} of natural-scale values
- `reconstructor`: ReConstructor for flatten/unflatten operations
"""
function _build_original_parameters(hazards::Vector{<:HazardFunction}, data::DataFrame)
    hazkeys = Dict{Symbol, Int64}()
    params_nested_pairs = Vector{Pair{Symbol, NamedTuple}}()
    
    for (idx, h) in enumerate(hazards)
        hazname = Symbol("h$(h.statefrom)$(h.stateto)")
        hazkeys[hazname] = idx
        
        if h isa PhaseTypeHazard
            # PT hazard: 2n-1 baseline params (λ rates + μ exit rates) + covariates
            n = h.n_phases
            npar_baseline = 2 * n - 1
            npar_covar = _count_covariates(h.hazard, data)
            
            # Build baseline parameter names: λ₁, λ₂, ..., μ₁, μ₂, ..., μₙ
            baseline_names = Symbol[]
            for i in 1:(n-1)
                push!(baseline_names, Symbol("$(hazname)_λ$i"))
            end
            for i in 1:n
                push!(baseline_names, Symbol("$(hazname)_μ$i"))
            end
            baseline_nt = NamedTuple{Tuple(baseline_names)}(zeros(npar_baseline))
            
            # Build covariate parameter names if any
            if npar_covar > 0
                covar_names = [Symbol("$(hazname)_covar$i") for i in 1:npar_covar]
                covar_nt = NamedTuple{Tuple(covar_names)}(zeros(npar_covar))
                push!(params_nested_pairs, hazname => (baseline = baseline_nt, covariates = covar_nt))
            else
                push!(params_nested_pairs, hazname => (baseline = baseline_nt,))
            end
        else
            # Non-PT hazard: use standard parameter structure
            npar = _count_hazard_parameters(h, data)
            npar_covar = _count_covariates(h.hazard, data)
            npar_baseline = npar - npar_covar
            
            baseline_names = [Symbol("$(hazname)_baseline$i") for i in 1:npar_baseline]
            baseline_nt = NamedTuple{Tuple(baseline_names)}(zeros(npar_baseline))
            
            if npar_covar > 0
                covar_names = [Symbol("$(hazname)_covar$i") for i in 1:npar_covar]
                covar_nt = NamedTuple{Tuple(covar_names)}(zeros(npar_covar))
                push!(params_nested_pairs, hazname => (baseline = baseline_nt, covariates = covar_nt))
            else
                push!(params_nested_pairs, hazname => (baseline = baseline_nt,))
            end
        end
    end
    
    # Sort by hazard index and build nested structure
    sorted_pairs = sort(params_nested_pairs, by = p -> hazkeys[p.first])
    params_nested = NamedTuple(sorted_pairs)
    
    # Build reconstructor and flat vector
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    # Build natural-scale parameters (simple vectors per hazard)
    # For phase-type, all baseline params are rates (positive), so exp transform
    params_natural_pairs = [
        hazname => _extract_original_natural_vector(params_nested[hazname])
        for hazname in keys(params_nested)
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    return (
        flat = params_flat,
        nested = params_nested,
        natural = params_natural,
        reconstructor = reconstructor
    )
end

"""
    _extract_original_natural_vector(hazard_params::NamedTuple)

Extract parameter vector from original phase-type hazard params.
As of v0.3.0, all parameters are already on natural scale (no exp transform needed).
Covariate coefficients are unconstrained (kept as-is).
"""
function _extract_original_natural_vector(hazard_params::NamedTuple)
    baseline_vals = collect(values(hazard_params.baseline))
    # v0.3.0+: Parameters already on natural scale, no exp() needed
    baseline_natural = baseline_vals
    
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_natural, covar_vals)
    else
        return baseline_natural
    end
end

"""
    _count_covariates(formula, data)

Count the number of covariate parameters in a hazard formula.
"""
function _count_covariates(formula::FormulaTerm, data::DataFrame)
    schema = StatsModels.schema(formula, data)
    hazschema = apply_schema(formula, schema)
    rhs_names = _hazard_rhs_names(hazschema)
    # Subtract 1 for intercept
    return max(0, length(rhs_names) - 1)
end

"""
    _count_hazard_parameters(h, data)

Count the total number of parameters for a hazard specification.
"""
function _count_hazard_parameters(h::HazardFunction, data::DataFrame)
    ncovar = _count_covariates(h.hazard, data)
    family = h.family isa Symbol ? h.family : Symbol(lowercase(String(h.family)))
    
    if family == :exp
        return 1 + ncovar
    elseif family in (:wei, :gom)
        return 2 + ncovar
    elseif family == :sp
        # Splines have variable number of basis functions
        # This is an approximation; actual count determined during build
        return 5 + ncovar
    else
        return 1 + ncovar  # Default
    end
end

"""
    _merge_censoring_patterns_with_shift(user_patterns, phase_patterns, mappings, 
                                          expanded_data, original_data, original_row_map)

Merge user-provided censoring patterns with phase uncertainty patterns.

User patterns are kept at their original codes (expanded to phase space).
Auto-generated phase patterns are shifted to start after the user's maximum code.

This ensures user's obstype values remain unchanged while auto-generated phase
uncertainty patterns get new codes that don't conflict.

# Arguments
- `user_patterns`: User-supplied censoring patterns (n_user × n_observed+1)
- `phase_patterns`: Auto-generated phase uncertainty patterns (n_phase × n_expanded+1)
- `mappings`: PhaseTypeMappings
- `expanded_data`: Data after phase-type expansion
- `original_data`: Original data (before expansion) to identify user obstypes
- `original_row_map`: Maps expanded row indices to original row indices

Returns the combined censoring patterns AND the updated expanded_data with shifted phase obstypes.
"""
function _merge_censoring_patterns_with_shift(user_patterns::Matrix{<:Real}, 
                                               phase_patterns::Matrix{Float64},
                                               mappings::PhaseTypeMappings,
                                               expanded_data::DataFrame,
                                               original_data::DataFrame,
                                               original_row_map::Vector{Int})
    n_observed = mappings.n_observed
    n_expanded = mappings.n_expanded
    n_user = size(user_patterns, 1)
    n_phase = size(phase_patterns, 1)
    
    # Identify user codes from the user_patterns matrix
    user_codes = Set(Int.(user_patterns[:, 1]))
    
    # User patterns keep their original codes (starting at 3)
    # Phase patterns are shifted to start after user's max code
    user_max_code = n_user > 0 ? maximum(user_codes) : 2
    phase_shift = user_max_code - 2  # How much to shift phase codes (originally 3, 4, ...)
    
    # Create expanded user patterns (keep original codes, expand to phase space)
    expanded_user = zeros(Float64, n_user, n_expanded + 1)
    
    for p in 1:n_user
        original_code = Int(user_patterns[p, 1])
        expanded_user[p, 1] = original_code  # Keep original code
        
        # Expand state probabilities to phase probabilities
        for s_obs in 1:n_observed
            obs_prob = user_patterns[p, s_obs + 1]
            phases = mappings.state_to_phases[s_obs]
            for phase in phases
                # Equal probability across phases of each state
                expanded_user[p, phase + 1] = obs_prob
            end
        end
    end
    
    # Create shifted phase patterns
    shifted_phase = copy(phase_patterns)
    for p in 1:n_phase
        shifted_phase[p, 1] = phase_patterns[p, 1] + phase_shift
    end
    
    # Combine: user patterns first, then shifted phase patterns
    combined = vcat(expanded_user, shifted_phase)
    
    # Sort by code to ensure consecutive ordering
    sort_order = sortperm(combined[:, 1])
    combined = combined[sort_order, :]
    
    # Update expanded_data.obstype: 
    # - User codes in original data stay the same
    # - Phase-generated codes (from expand_data_for_phasetype) need to be shifted
    updated_data = copy(expanded_data)
    
    # Build mapping from original phase codes to shifted codes
    # Phase codes are 3, 4, ... for multi-phase states
    multi_phase_states = [s for s in 1:n_observed if length(mappings.state_to_phases[s]) > 1]
    phase_codes_original = Set(2 + s for s in multi_phase_states)
    
    code_shift_map = Dict{Int, Int}()
    for s in multi_phase_states
        original_code = 2 + s
        code_shift_map[original_code] = original_code + phase_shift
    end
    
    for i in 1:nrow(updated_data)
        ot = updated_data.obstype[i]
        orig_row = original_row_map[i]
        orig_obstype = original_data.obstype[orig_row]
        
        if ot >= 3
            # Is this obstype from the original data (user code) or generated (phase code)?
            if orig_obstype in user_codes
                # This is a user code - keep it unchanged
                # (it may have been preserved through expansion)
                updated_data.obstype[i] = orig_obstype
            elseif haskey(code_shift_map, ot)
                # This is a phase-generated code - shift it
                updated_data.obstype[i] = code_shift_map[ot]
            end
        end
    end
    
    return combined, updated_data
end

