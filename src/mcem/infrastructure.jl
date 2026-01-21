# =============================================================================
# MCEM Infrastructure
# =============================================================================
#
# This file defines the MCEMInfrastructure struct and builder functions for
# surrogate-agnostic MCEM implementation.
#
# The infrastructure is built once at MCEM start and contains all precomputed
# objects needed for importance sampling. Surrogate-specific differences are
# handled via dispatch on build_mcem_infrastructure().
#
# =============================================================================

"""
    MCEMInfrastructure{S<:AbstractSurrogate}

All precomputed infrastructure needed for MCEM importance sampling.
Parameterized by surrogate type to enable dispatch-based specialization.

Built once at MCEM start via `build_mcem_infrastructure(model, surrogate)`.

# Fields

**Core infrastructure (used by both surrogate types):**
- `tpm_book::Vector{Vector{Matrix{Float64}}}`: TPMs [covariate combo][time interval]
- `hazmat_book::Vector{Matrix{Float64}}`: Intensity matrices (Q) per covariate combo
- `fbmats::Union{Nothing, Vector}`: Forward-backward matrices per subject
- `emat::Matrix{Float64}`: Emission matrix for censored observations
- `books::Tuple`: (tpm_times, tpm_map) from `build_tpm_mapping`
- `exp_cache`: Preallocated cache for matrix exponential computation

**Data view (may be expanded for phase-type with exact observations):**
- `data::DataFrame`: Working data (original or expanded)
- `subjectindices::Vector{Vector{Int}}`: Row indices per subject

**Surrogate reference:**
- `surrogate::S`: The surrogate model (for dispatch and parameter access)

**Markov surrogate infrastructure:**
- `tpm_book_markov::Vector{Vector{Matrix{Float64}}}`: TPMs on original state space
- `hazmat_book_markov::Vector{Matrix{Float64}}`: Intensity matrices on original

**Surrogate-specific extras (Nothing for Markov):**
- `schur_cache::Union{Nothing, Vector{CachedSchurDecomposition}}`: PhaseType only
- `original_row_map::Union{Nothing, Vector{Int}}`: PhaseType data expansion map
- `expanded_tpm_map::Union{Nothing, Matrix{Int}}`: PhaseType expanded tpm_map

**Model reference:**
- `absorbingstates::Vector{Int}`: Indices of absorbing states
- `nsubj::Int`: Number of subjects

See also: [`build_mcem_infrastructure`](@ref), [`AbstractSurrogate`](@ref)
"""
struct MCEMInfrastructure{S<:AbstractSurrogate}
    # Core infrastructure (state space depends on surrogate type)
    # tpm_book is Vector{Vector{Matrix}} - outer for covariate combos, inner for time intervals
    tpm_book::Vector{Vector{Matrix{Float64}}}
    hazmat_book::Vector{Matrix{Float64}}
    fbmats::Union{Nothing, Vector}
    emat::Matrix{Float64}
    books::Tuple{Vector, Matrix{Int}}
    exp_cache::Any  # ExponentialUtilities cache
    
    # Data view
    data::DataFrame
    subjectindices::Vector{Vector{Int}}
    
    # Surrogate reference
    surrogate::S
    
    # Markov surrogate infrastructure (always built, used for target loglik eval)
    # For Markov surrogate: same as tpm_book/hazmat_book
    # For PhaseType surrogate: separate infrastructure on original state space
    tpm_book_markov::Vector{Vector{Matrix{Float64}}}
    hazmat_book_markov::Vector{Matrix{Float64}}
    
    # PhaseType-specific (Nothing for Markov)
    schur_cache::Union{Nothing, Vector{CachedSchurDecomposition}}
    original_row_map::Union{Nothing, Vector{Int}}
    expanded_tpm_map::Union{Nothing, Matrix{Int}}
    
    # Model info
    absorbingstates::Vector{Int}
    nsubj::Int
end

# =============================================================================
# Builder: MarkovSurrogate
# =============================================================================

"""
    build_mcem_infrastructure(model::MultistateModel, surrogate::MarkovSurrogate; verbose=false)

Build MCEM infrastructure for Markov surrogate importance sampling.

The Markov surrogate operates on the original state space. TPMs are computed
via matrix exponential of the intensity matrix.

# Returns
`MCEMInfrastructure{MarkovSurrogate}` with all precomputed objects.
"""
function build_mcem_infrastructure(model::MultistateModel, surrogate::MarkovSurrogate; verbose::Bool=false)
    # Identify absorbing states
    absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))
    nsubj = length(model.subjectindices)
    
    # Build TPM bookkeeping
    books = build_tpm_mapping(model.data)
    
    # Allocate TPM and hazmat books on original state space
    hazmat_book = build_hazmat_book(Float64, model.tmat, books[1])
    tpm_book = build_tpm_book(Float64, model.tmat, books[1])
    
    # Allocate cache for matrix exponential
    exp_cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())
    
    # Solve Kolmogorov equations: compute TPMs from intensity matrices
    surrogate_pars = get_hazard_params(surrogate.parameters, surrogate.hazards)
    for t in eachindex(books[1])
        compute_hazmat!(hazmat_book[t], surrogate_pars, surrogate.hazards, books[1][t], model.data)
        compute_tmat!(tpm_book[t], hazmat_book[t], books[1][t], exp_cache)
    end
    
    # Build forward-backward matrices if censored observations exist
    fbmats = any(model.data.obstype .> 2) ? build_fbmats(model) : nothing
    
    if verbose
        println("Built Markov surrogate infrastructure:")
        println("  TPM combos: $(length(tpm_book))")
        println("  Subjects: $nsubj")
        println("  FB matrices: $(isnothing(fbmats) ? "not needed" : "built")")
    end
    
    MCEMInfrastructure(
        tpm_book,           # Primary TPM book (same as Markov)
        hazmat_book,        # Primary hazmat book (same as Markov)
        fbmats,
        model.emat,
        books,
        exp_cache,
        model.data,
        model.subjectindices,
        surrogate,
        tpm_book,           # Markov TPM (same reference)
        hazmat_book,        # Markov hazmat (same reference)
        nothing,            # No Schur cache
        nothing,            # No row map
        nothing,            # No expanded tpm_map
        absorbingstates,
        nsubj
    )
end

# =============================================================================
# Builder: PhaseTypeSurrogate
# =============================================================================

"""
    build_mcem_infrastructure(model::MultistateModel, surrogate::PhaseTypeSurrogate; verbose=false)

Build MCEM infrastructure for phase-type surrogate importance sampling.

The phase-type surrogate operates on an expanded state space (phases within
each observed state). Data may be expanded to express phase uncertainty for
exact observations.

# Returns
`MCEMInfrastructure{PhaseTypeSurrogate}` with all precomputed objects.
"""
function build_mcem_infrastructure(model::MultistateModel, surrogate::PhaseTypeSurrogate; verbose::Bool=false)
    # Identify absorbing states
    absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))
    nsubj = length(model.subjectindices)
    
    # =========================================================================
    # Build Markov infrastructure on ORIGINAL state space
    # (needed for sampling via FFBS on original state space)
    # =========================================================================
    books_markov = build_tpm_mapping(model.data)
    hazmat_book_markov = build_hazmat_book(Float64, model.tmat, books_markov[1])
    tpm_book_markov = build_tpm_book(Float64, model.tmat, books_markov[1])
    exp_cache = ExponentialUtilities.alloc_mem(similar(hazmat_book_markov[1]), ExpMethodGeneric())
    
    # Solve Kolmogorov equations for Markov surrogate
    # Use the underlying MarkovSurrogate rates from PhaseTypeSurrogate
    markov_pars = get_hazard_params(surrogate.parameters, surrogate.hazards)
    for t in eachindex(books_markov[1])
        compute_hazmat!(hazmat_book_markov[t], markov_pars, surrogate.hazards, books_markov[1][t], model.data)
        compute_tmat!(tpm_book_markov[t], hazmat_book_markov[t], books_markov[1][t], exp_cache)
    end
    
    # =========================================================================
    # Handle data expansion for exact observations
    # =========================================================================
    expanded_data = nothing
    expanded_subjectindices = nothing
    original_row_map = nothing
    expanded_tpm_map = nothing
    censoring_patterns = nothing
    
    if needs_data_expansion_for_phasetype(model.data)
        n_states = size(model.tmat, 1)
        expansion_result = expand_data_for_phasetype(model.data, n_states)
        expanded_data = expansion_result.expanded_data
        censoring_patterns = expansion_result.censoring_patterns
        original_row_map = expansion_result.original_row_map
        expanded_subjectindices = compute_expanded_subject_indices(expanded_data)
        
        if verbose
            n_orig = nrow(model.data)
            n_exp = nrow(expanded_data)
            println("  Expanded data for phase-type: $n_orig → $n_exp rows")
        end
    end
    
    # Working data and indices (expanded if needed)
    working_data = isnothing(expanded_data) ? model.data : expanded_data
    working_subjectindices = isnothing(expanded_subjectindices) ? model.subjectindices : expanded_subjectindices
    
    # =========================================================================
    # Build PhaseType infrastructure on EXPANDED state space
    # =========================================================================
    if !isnothing(expanded_data)
        books_ph = build_tpm_mapping(expanded_data)
        expanded_tpm_map = books_ph[2]
        tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(surrogate, books_ph, expanded_data)
    else
        books_ph = books_markov  # Use same books if no expansion
        tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(surrogate, books_markov, model.data)
    end
    
    # Pre-compute Schur decompositions for efficient TPM computation at arbitrary times
    schur_cache = [CachedSchurDecomposition(Q) for Q in hazmat_book_ph]
    
    # Build forward-backward matrices for expanded state space
    fbmats_ph = build_fbmats_phasetype_with_indices(working_subjectindices, surrogate)
    
    # Build emission matrix for expanded state space
    emat_ph = build_phasetype_emat_expanded(model, surrogate;
                                             expanded_data = expanded_data,
                                             censoring_patterns = censoring_patterns)
    
    if verbose
        println("Built phase-type surrogate infrastructure:")
        println("  Original states: $(size(model.tmat, 1))")
        println("  Expanded states: $(surrogate.n_expanded_states)")
        println("  TPM combos: $(length(tpm_book_ph))")
        println("  Subjects: $nsubj")
        println("  Data expanded: $(!isnothing(expanded_data))")
    end
    
    MCEMInfrastructure(
        tpm_book_ph,          # Primary TPM book (expanded)
        hazmat_book_ph,       # Primary hazmat book (expanded)
        fbmats_ph,
        emat_ph,
        books_ph,
        exp_cache,
        working_data,
        working_subjectindices,
        surrogate,
        tpm_book_markov,      # Markov TPM (original space)
        hazmat_book_markov,   # Markov hazmat (original space)
        schur_cache,
        original_row_map,
        expanded_tpm_map,
        absorbingstates,
        nsubj
    )
end

# =============================================================================
# Helper: Check if infrastructure is for PhaseType
# =============================================================================

"""
    is_phasetype_infrastructure(infra::MCEMInfrastructure) -> Bool

Check if infrastructure is for phase-type surrogate.
"""
is_phasetype_infrastructure(infra::MCEMInfrastructure{MarkovSurrogate}) = false
is_phasetype_infrastructure(infra::MCEMInfrastructure{PhaseTypeSurrogate}) = true

"""
    get_surrogate_type(infra::MCEMInfrastructure) -> Symbol

Get the surrogate type as a symbol (:markov or :phasetype).
"""
get_surrogate_type(::MCEMInfrastructure{MarkovSurrogate}) = :markov
get_surrogate_type(::MCEMInfrastructure{PhaseTypeSurrogate}) = :phasetype
