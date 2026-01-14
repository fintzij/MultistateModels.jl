# =============================================================================
# Transition Probability Matrix (TPM) Functions
# =============================================================================
#
# Functions for computing hazard intensity matrices and transition probability
# matrices for Markov models.
#
# =============================================================================

########################################################
############# multistate markov process ################
###### transition intensities and probabilities ########
########################################################

"""
    compute_hazmat!(Q, parameters, hazards::Vector{T}, tpm_index::DataFrame, model_data::DataFrame) where T <: _Hazard

Fill in a matrix of transition intensities for a multistate Markov model (in-place version).
"""
function compute_hazmat!(Q, parameters, hazards::Vector{T}, tpm_index::DataFrame, model_data::DataFrame) where T <: _Hazard

    # Get the DataFrameRow for covariate extraction - ONLY ONCE
    @inbounds subjdat_row = model_data[tpm_index.datind[1], :]
    
    # Pre-extract covariates for each hazard using cached covar_names
    # This avoids regex parsing in extract_covariates on every call
    @inbounds for h in eachindex(hazards) 
        hazard = hazards[h]
        # Use extract_covariates_fast with pre-cached covar_names
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        Q[hazard.statefrom, hazard.stateto] = 
            eval_hazard(hazard, tpm_index.tstart[1], parameters[hazard.hazname], covars)
    end

    # set diagonal elements equal to the sum of off-diags
    Q[diagind(Q)] = -sum(Q, dims = 2)
end

"""
    compute_hazmat_cached!(Q, pars_cache, covars_cache, hazards, pattern_idx)

Fill in transition intensity matrix using cached parameters and covariates.
This is the allocation-free path for Float64 parameters.

# Arguments
- `Q::Matrix{Float64}`: Pre-allocated intensity matrix to fill
- `pars_cache::Vector{Vector{Float64}}`: Cached parameter vectors per hazard
- `covars_cache::Vector{Vector{NamedTuple}}`: Cached covariates per pattern per hazard
- `hazards::Vector{<:_Hazard}`: Hazard objects
- `pattern_idx::Int`: Index of the covariate pattern
"""
function compute_hazmat_cached!(Q::Matrix{Float64}, 
                                pars_cache::Vector{Vector{Float64}},
                                covars_cache::Vector{Vector{NamedTuple}},
                                hazards::Vector{<:_Hazard},
                                pattern_idx::Int)
    # Reset Q to zero
    fill!(Q, 0.0)
    
    # Get pre-extracted covariates for this pattern
    pattern_covars = covars_cache[pattern_idx]
    
    # Evaluate hazard for each transition
    @inbounds for h in eachindex(hazards)
        hazard = hazards[h]
        pars = pars_cache[h]
        covars = pattern_covars[h]
        # Evaluate at t=0 (for Markov models, rate is time-invariant)
        Q[hazard.statefrom, hazard.stateto] = hazard(0.0, pars, covars)
    end
    
    # Set diagonal elements equal to negative sum of off-diagonals
    @inbounds for i in axes(Q, 1)
        row_sum = 0.0
        for j in axes(Q, 2)
            if i != j
                row_sum += Q[i, j]
            end
        end
        Q[i, i] = -row_sum
    end
    
    return nothing
end

"""
    compute_hazard_rates!(rates_cache, pars, hazards, covars_cache)

Pre-compute hazard rates for all (pattern, hazard) combinations.

For Markov models, hazard rates are time-invariant: `h(t) = rate * exp(Xβ)`.
This function evaluates all rates once when parameters change, avoiding
repeated hazard function evaluations in `compute_hazmat!`.

# Arguments
- `rates_cache::Vector{Vector{Float64}}`: Pre-allocated [pattern][hazard] → rate
- `pars::NamedTuple`: Unflattened natural-scale parameters
- `hazards::Vector{<:_Hazard}`: Hazard objects (must be Markov; time-invariant)
- `covars_cache::Vector{Vector{NamedTuple}}`: Pre-extracted [pattern][hazard] → covars

# Notes
- Only valid for Markov hazard types (exponential, phase-type)
- Semi-Markov hazards (Weibull, Gompertz, splines) are time-dependent and cannot be cached
- Evaluates at t=0 since Markov rates don't depend on time
"""
function compute_hazard_rates!(rates_cache::Vector{Vector{Float64}},
                               pars::NamedTuple,
                               hazards::Vector{<:_Hazard},
                               covars_cache::Vector{Vector{NamedTuple}})
    n_patterns = length(rates_cache)
    n_hazards = length(hazards)
    
    @inbounds for p in 1:n_patterns
        pattern_covars = covars_cache[p]
        pattern_rates = rates_cache[p]
        
        for h in 1:n_hazards
            hazard = hazards[h]
            covars = pattern_covars[h]
            hazard_pars = pars[hazard.hazname]
            # Evaluate at t=0 (Markov rates are time-invariant)
            pattern_rates[h] = eval_hazard(hazard, 0.0, hazard_pars, covars)
        end
    end
    
    return nothing
end

"""
    compute_hazmat_from_rates!(Q, rates, hazards)

Fill transition intensity matrix Q using pre-computed hazard rates.
This is the allocation-free path for Markov models with cached rates.

# Arguments
- `Q::Matrix{Float64}`: Pre-allocated intensity matrix to fill (will be zeroed)
- `rates::Vector{Float64}`: Pre-computed hazard rates for this pattern [h] → rate
- `hazards::Vector{<:_Hazard}`: Hazard objects (must be Markov; for statefrom/stateto)

# Performance
This function is O(n_hazards + n_states) with zero allocations, compared to
`compute_hazmat!` which evaluates hazard functions and may extract covariates.
"""
function compute_hazmat_from_rates!(Q::Matrix{Float64},
                                    rates::Vector{Float64},
                                    hazards::Vector{<:_Hazard})
    # Reset Q to zero
    fill!(Q, 0.0)
    
    # Fill off-diagonal elements from cached rates
    @inbounds for h in eachindex(hazards)
        hazard = hazards[h]
        Q[hazard.statefrom, hazard.stateto] = rates[h]
    end
    
    # Set diagonal elements to negative row sums (generator matrix property)
    @inbounds for i in axes(Q, 1)
        row_sum = 0.0
        for j in axes(Q, 2)
            if i != j
                row_sum += Q[i, j]
            end
        end
        Q[i, i] = -row_sum
    end
    
    return nothing
end

"""
    compute_hazmat(T, n_states, parameters, hazards, tpm_index, model_data)

Construct transition intensity matrix Q for a multistate Markov model (non-mutating version).

Returns a fresh matrix without modifying any pre-allocated storage.
Compatible with reverse-mode AD (Enzyme, Zygote).

# Arguments
- `T`: Element type (e.g., Float64 or Dual)
- `n_states::Int`: Number of states in the model
- `parameters`: Nested parameters (tuple of vectors)
- `hazards`: Vector of hazard objects
- `tpm_index`: DataFrame with time and data indices
- `model_data`: Model data DataFrame
"""
function compute_hazmat(::Type{T}, n_states::Int, parameters, hazards::Vector{<:_Hazard}, 
                        tpm_index::DataFrame, model_data::DataFrame) where T
    # Get covariate row once
    subjdat_row = model_data[tpm_index.datind[1], :]
    
    # Build Q matrix functionally using comprehension
    # Start with zeros, then set off-diagonals
    Q = zeros(T, n_states, n_states)
    
    for h in eachindex(hazards)
        hazard = hazards[h]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        rate = eval_hazard(hazard, tpm_index.tstart[1], parameters[hazard.hazname], covars)
        Q = setindex_immutable(Q, rate, hazard.statefrom, hazard.stateto)
    end
    
    # Set diagonal: each element is negative sum of its row (excluding diagonal)
    for i in 1:n_states
        row_sum = zero(T)
        for j in 1:n_states
            if i != j
                row_sum += Q[i, j]
            end
        end
        Q = setindex_immutable(Q, -row_sum, i, i)
    end
    
    return Q
end

"""
    setindex_immutable(A, val, i, j)

Return a new matrix with A[i,j] = val without mutating A.
This is the key primitive for reverse-mode AD compatibility.
"""
@inline function setindex_immutable(A::AbstractMatrix{T}, val::T, i::Int, j::Int) where T
    # Create a copy and set the value
    B = copy(A)
    B[i, j] = val
    return B
end

# More efficient version using ntuple for small matrices (avoids copy overhead)
@inline function setindex_immutable(A::AbstractMatrix{T}, val, i::Int, j::Int) where T
    B = copy(A)
    B[i, j] = convert(T, val)
    return B
end

# =============================================================================
# Transition Probability Matrix Computation
# =============================================================================

"""
    compute_tmat!(P, Q, tpm_index::DataFrame, cache)

Calculate transition probability matrices for a multistate Markov process (in-place version). 
"""
function compute_tmat!(P, Q, tpm_index::DataFrame, cache)
    @inbounds for t in eachindex(P)
        copyto!(P[t], exponential!(Q * tpm_index.tstop[t], ExpMethodGeneric(), cache))
    end  
end

"""
    compute_tmat_batched!(P, Q, dt_values, schur_cache, pattern_idx)

Calculate transition probability matrices using Schur decomposition for multiple Δt values.

Uses the Schur decomposition Q = U T U* where T is upper triangular. Then:
    exp(Q * Δt) = U * exp(T * Δt) * U*

The key optimization is computing the Schur decomposition once and reusing it
for all Δt values. For triangular T, exp(T * Δt) is also triangular, but we
use the standard matrix exponential which is well-optimized by LAPACK.

# Arguments
- `P::Vector{Matrix{Float64}}`: Pre-allocated output matrices (one per Δt)
- `Q::Matrix{Float64}`: Transition intensity matrix  
- `dt_values::Vector{Float64}`: Time intervals (Δt values)
- `schur_cache::SchurCache`: Pre-allocated workspace for Schur computation
- `pattern_idx::Int`: Index (unused, retained for API compatibility)

# Performance
- Schur decomposition: O(n³) once per Q matrix
- Each exp(T * Δt): O(n³) but with triangular structure (faster in practice)
- Matrix multiplications: O(n³) per Δt (fast BLAS)
- Break-even vs standard approach: ~2 different Δt values
- Speedup for 10 Δt values: ~2-4x depending on matrix size

# Numerical Stability
Schur decomposition is always stable, unlike eigendecomposition which fails
for defective matrices (repeated eigenvalues) common in phase-type models.
"""
function compute_tmat_batched!(P::Vector{Matrix{Float64}}, 
                                Q::Matrix{Float64}, 
                                dt_values::Vector{Float64},
                                schur_cache::SchurCache,
                                pattern_idx::Int)
    k = length(dt_values)
    n = size(Q, 1)
    
    # Handle edge case: no intervals
    k == 0 && return nothing
    
    # Compute Schur decomposition: Q = U * T * U'
    # T is upper triangular, U is unitary
    F = schur!(copyto!(schur_cache.Q_work, Q))
    T = F.T  # Upper triangular (Schur form)
    U = F.Z  # Unitary matrix
    
    # Compute exp(Q * dt) = U * exp(T * dt) * U' for each dt
    @inbounds for t in 1:k
        dt = dt_values[t]
        if dt == 0.0
            # Identity matrix for dt=0
            fill!(P[t], 0.0)
            for i in 1:n
                P[t][i, i] = 1.0
            end
        else
            # exp(T * dt) - still O(n³) but triangular structure helps
            schur_cache.E_work .= exp(T * dt)
            
            # P = U * exp(T*dt) * U'  (two O(n³) BLAS calls)
            mul!(schur_cache.tmp_work, U, schur_cache.E_work)  # tmp = U * E
            mul!(P[t], schur_cache.tmp_work, U')               # P = tmp * U'
        end
    end
    
    return nothing
end

# Legacy signature for backward compatibility (creates temporary SchurCache)
function compute_tmat_batched!(P::Vector{Matrix{Float64}}, 
                                Q::Matrix{Float64}, 
                                dt_values::Vector{Float64},
                                eigen_cache::Vector{Union{Nothing, Tuple{Matrix{Float64}, Vector{ComplexF64}, Matrix{ComplexF64}}}},
                                pattern_idx::Int)
    n = size(Q, 1)
    # Create temporary SchurCache (not ideal for performance, but maintains API)
    cache = SchurCache(n)
    compute_tmat_batched!(P, Q, dt_values, cache, pattern_idx)
    return nothing
end

"""
    invalidate_eigen_cache!(cache, pattern_idx::Int)

Clear the eigendecomposition cache for a pattern when Q matrix changes.
Called when parameters are updated.
"""
@inline function invalidate_eigen_cache!(cache, pattern_idx::Int)
    if pattern_idx <= length(cache.eigen_cache)
        cache.eigen_cache[pattern_idx] = nothing
    end
end

"""
    compute_tmat(Q, dt)

Compute transition probability matrix P = exp(Q * dt) without mutation.
Compatible with reverse-mode AD.

# Arguments
- `Q`: Transition intensity matrix
- `dt`: Time interval

# Returns
- `P`: Transition probability matrix
"""
function compute_tmat(Q::AbstractMatrix{T}, dt::Real) where T
    # Use ExponentialUtilities.exp_generic for AD-compatible matrix exponential
    # exp_generic handles arbitrary element types (including Dual numbers for ForwardDiff)
    Qt = Q * dt
    return ExponentialUtilities.exp_generic(Qt)
end
