# =============================================================================
# Phase-Type Correctness Tests
# =============================================================================
#
# This test file validates the mathematical correctness of phase-type
# distribution implementations by comparing against analytic formulas.
#
# Coverage:
#   1. Emission Matrix: Correct obstype codes, censoring patterns, phase mappings
#   2. Data Expansion: Correct splitting of exact observations
#   3. Coxian Hazard/Survival: Match known analytic formulas
#   4. Loglik: Match analytic path densities
#
# Analytic Reference Formulas for Coxian Phase-Type:
# --------------------------------------------------
# For a p-phase Coxian with sub-intensity S and absorption rates a:
#
# Density:     f(t) = π' exp(St) a
# Survival:    S(t) = π' exp(St) 𝟙
# Hazard:      h(t) = f(t) / S(t) = [π' exp(St) a] / [π' exp(St) 𝟙]
# Cumulative:  H(t) = -log(S(t))
#
# For 2-phase Coxian starting in phase 1 with:
#   S = [-(r₁+a₁)   r₁    ]     a = [a₁]     π = [1]
#       [   0    -(a₂)]             [a₂]         [0]
#
# The matrix exponential exp(St) can be computed analytically.
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using LinearAlgebra
using Random
using QuadGK

import MultistateModels: PhaseTypeConfig, PhaseTypeDistribution, PhaseTypeSurrogate,
    build_phasetype_surrogate, build_phasetype_emat, build_phasetype_emat_expanded,
    expand_data_for_phasetype, needs_data_expansion_for_phasetype,
    compute_expanded_subject_indices, build_coxian_intensity, subintensity,
    absorption_rates, progression_rates, loglik_phasetype_expanded, SamplePath,
    collapse_phasetype_path

# =============================================================================
# Helper: Analytic Matrix Exponential for 2-Phase Coxian
# =============================================================================

#=
Compute exp(St) analytically for a 2-phase Coxian.

For S = [-(r+a1)   r  ]  with eigenvalues -lam1 = -(r+a1), -lam2 = -a2
        [  0    -a2  ]

For an upper triangular matrix, the matrix exponential has:
  exp(S*t)[1,1] = exp(-lam1*t)
  exp(S*t)[2,2] = exp(-lam2*t)
  exp(S*t)[1,2] = r * (exp(-lam1*t) - exp(-lam2*t)) / (lam2 - lam1)  if lam1 ≠ lam2
                = r * t * exp(-lam1*t)                                if lam1 = lam2
=#
function coxian2_matrix_exp(t::Float64, r::Float64, a1::Float64, a2::Float64)
    λ1 = r + a1  # Total rate out of phase 1
    λ2 = a2      # Rate out of phase 2
    
    M11 = exp(-λ1 * t)
    M22 = exp(-λ2 * t)
    M21 = 0.0
    
    if abs(λ1 - λ2) < 1e-12
        # Degenerate case: use L'Hôpital limiting formula
        M12 = r * t * exp(-λ1 * t)
    else
        M12 = r * (exp(-λ1 * t) - exp(-λ2 * t)) / (λ2 - λ1)
    end
    
    return [M11 M12; M21 M22]
end

#=
Analytic survival function for 2-phase Coxian starting in phase 1.
S(t) = pi' exp(St) 1 where pi = [1, 0]
=#
function coxian2_survival(t::Float64, r::Float64, a1::Float64, a2::Float64)
    M = coxian2_matrix_exp(t, r, a1, a2)
    # π = [1, 0], so S(t) = M[1,1] + M[1,2] (sum of first row)
    return M[1, 1] + M[1, 2]
end

"""
Analytic density for 2-phase Coxian starting in phase 1.
f(t) = π' exp(St) a where a = [a1, a2]
"""
function coxian2_density(t::Float64, r::Float64, a1::Float64, a2::Float64)
    M = coxian2_matrix_exp(t, r, a1, a2)
    # f(t) = M[1,1]*a1 + M[1,2]*a2
    return M[1, 1] * a1 + M[1, 2] * a2
end

"""
Analytic hazard for 2-phase Coxian starting in phase 1.
h(t) = f(t) / S(t)
"""
function coxian2_hazard(t::Float64, r::Float64, a1::Float64, a2::Float64)
    return coxian2_density(t, r, a1, a2) / coxian2_survival(t, r, a1, a2)
end

"""
Analytic cumulative hazard for 2-phase Coxian starting in phase 1.
H(t) = -log(S(t))
"""
function coxian2_cumul_hazard(t::Float64, r::Float64, a1::Float64, a2::Float64)
    return -log(coxian2_survival(t, r, a1, a2))
end

# =============================================================================
# TEST SECTION 1: EMISSION MATRIX CORRECTNESS
# =============================================================================

@testset "Emission Matrix Correctness" begin
    
    @testset "build_phasetype_emat obstype handling" begin
        # Create a simple 3-state model: 1 → 2 → 3 (absorbing)
        tmat = [0 1 0; 0 0 1; 0 0 0]
        config = PhaseTypeConfig(n_phases=[2, 2, 1])  # 2 phases for states 1 & 2, 1 for absorbing
        surrogate = build_phasetype_surrogate(tmat, config)
        
        # Verify surrogate structure
        @test surrogate.n_observed_states == 3
        @test surrogate.n_expanded_states == 5  # 2 + 2 + 1
        @test surrogate.state_to_phases[1] == 1:2
        @test surrogate.state_to_phases[2] == 3:4
        @test surrogate.state_to_phases[3] == 5:5
        
        # Create test data with different obstypes
        data = DataFrame(
            id = [1, 1, 1, 1],
            tstart = [0.0, 1.0, 2.0, 3.0],
            tstop = [1.0, 2.0, 3.0, 4.0],
            statefrom = [1, 1, 2, 2],
            stateto = [1, 2, 2, 3],
            obstype = [2, 1, 2, 1]  # Panel, Exact, Panel, Exact
        )
        
        # Censoring patterns: 3 states, so we need patterns for obstype > 2
        # For this test, we use trivial patterns (not used for obstype 1, 2)
        CensoringPatterns = zeros(Float64, 1, 4)  # Minimal placeholder
        
        emat = build_phasetype_emat(data, surrogate, CensoringPatterns)
        
        # Check dimensions
        @test size(emat) == (4, 5)  # 4 rows of data, 5 expanded states
        
        # Row 1: obstype=2, stateto=1 → phases 1,2 should be 1.0
        @test emat[1, 1] == 1.0
        @test emat[1, 2] == 1.0
        @test emat[1, 3] == 0.0
        @test emat[1, 4] == 0.0
        @test emat[1, 5] == 0.0
        
        # Row 2: obstype=1, stateto=2 → phases 3,4 should be 1.0
        @test emat[2, 1] == 0.0
        @test emat[2, 2] == 0.0
        @test emat[2, 3] == 1.0
        @test emat[2, 4] == 1.0
        @test emat[2, 5] == 0.0
        
        # Row 3: obstype=2, stateto=2 → phases 3,4 should be 1.0
        @test emat[3, 3] == 1.0
        @test emat[3, 4] == 1.0
        
        # Row 4: obstype=1, stateto=3 → phase 5 should be 1.0
        @test emat[4, 5] == 1.0
        @test emat[4, 1:4] == zeros(4)
    end
    
    @testset "build_phasetype_emat obstype=0 (fully censored)" begin
        tmat = [0 1 0; 0 0 1; 0 0 0]
        config = PhaseTypeConfig(n_phases=[2, 2, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        
        data = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [0],  # Unknown
            obstype = [0]   # Fully censored
        )
        
        CensoringPatterns = zeros(Float64, 1, 4)
        emat = build_phasetype_emat(data, surrogate, CensoringPatterns)
        
        # All phases should be possible (1.0) for fully censored
        @test all(emat[1, :] .== 1.0)
    end
    
    @testset "build_phasetype_emat with CensoringPatterns" begin
        tmat = [0 1 1; 0 0 1; 0 0 0]  # States 1→2, 1→3, 2→3
        config = PhaseTypeConfig(n_phases=[2, 2, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        
        # Create censoring pattern: obstype=3 means "could be state 1 or 2"
        # Column format: [code, state1_prob, state2_prob, state3_prob]
        CensoringPatterns = [3.0 1.0 1.0 0.0]  # obstype=3 → states 1 or 2 possible
        
        data = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [0],
            obstype = [3]  # Partial censoring
        )
        
        emat = build_phasetype_emat(data, surrogate, CensoringPatterns)
        
        # Phases of state 1 (1,2) and state 2 (3,4) should have prob 1.0
        @test emat[1, 1] == 1.0  # state 1, phase 1
        @test emat[1, 2] == 1.0  # state 1, phase 2
        @test emat[1, 3] == 1.0  # state 2, phase 1
        @test emat[1, 4] == 1.0  # state 2, phase 2
        @test emat[1, 5] == 0.0  # state 3 (absorbing) not possible
    end
    
    @testset "Phase mappings are consistent" begin
        # Verify phase_to_state inverts state_to_phases
        tmat = [0 1 1; 0 0 1; 0 0 0]
        config = PhaseTypeConfig(n_phases=[3, 2, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        
        # Check that phase_to_state correctly maps back
        for s in 1:3
            for p in surrogate.state_to_phases[s]
                @test surrogate.phase_to_state[p] == s
            end
        end
        
        # Total phases should match n_expanded_states
        @test length(surrogate.phase_to_state) == surrogate.n_expanded_states
        @test surrogate.n_expanded_states == 3 + 2 + 1
    end
end

# =============================================================================
# TEST SECTION 2: DATA EXPANSION CORRECTNESS
# =============================================================================

@testset "Data Expansion for Phase-Type" begin
    
    @testset "expand_data_for_phasetype splits exact observations" begin
        # Simple test: one exact observation
        data = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [2],
            obstype = [1]  # Exact
        )
        
        result = expand_data_for_phasetype(data, 3)  # 3 states
        expanded = result.expanded_data
        
        # Should have 2 rows
        @test nrow(expanded) == 2
        
        # Row 1: sojourn interval with censoring code = 2 + statefrom = 3
        @test expanded.id[1] == 1
        @test expanded.tstart[1] == 0.0
        @test expanded.tstop[1] ≈ 1.0 - sqrt(eps()) atol=1e-10
        @test expanded.statefrom[1] == 1
        @test expanded.stateto[1] == 0  # Censored at end
        @test expanded.obstype[1] == 3  # 2 + statefrom = 2 + 1 = 3
        
        # Row 2: exact observation of transition
        @test expanded.id[2] == 1
        @test expanded.tstart[2] ≈ 1.0 - sqrt(eps()) atol=1e-10
        @test expanded.tstop[2] == 1.0
        @test expanded.statefrom[2] == 0  # Coming from censored
        @test expanded.stateto[2] == 2
        @test expanded.obstype[2] == 1  # Exact
    end
    
    @testset "expand_data_for_phasetype preserves panel observations" begin
        data = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [2],
            obstype = [2]  # Panel - should not be split
        )
        
        result = expand_data_for_phasetype(data, 3)
        expanded = result.expanded_data
        
        # Should still have 1 row
        @test nrow(expanded) == 1
        @test expanded.obstype[1] == 2
        @test expanded.statefrom[1] == 1
        @test expanded.stateto[1] == 2
    end
    
    @testset "expand_data_for_phasetype censoring patterns are correct" begin
        data = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 2],
            stateto = [2, 3],
            obstype = [1, 1]
        )
        
        result = expand_data_for_phasetype(data, 3)
        censor_pat = result.censoring_patterns
        
        # Should have 3 rows (one per state)
        @test size(censor_pat, 1) == 3
        
        # Column 1 is the code, columns 2:4 are state indicators
        # Row 1 (obstype=3): state 1 possible
        @test censor_pat[1, 1] == 3.0  # code
        @test censor_pat[1, 2] == 1.0  # state 1
        @test censor_pat[1, 3] == 0.0  # state 2
        @test censor_pat[1, 4] == 0.0  # state 3
        
        # Row 2 (obstype=4): state 2 possible
        @test censor_pat[2, 1] == 4.0
        @test censor_pat[2, 2] == 0.0
        @test censor_pat[2, 3] == 1.0
        @test censor_pat[2, 4] == 0.0
        
        # Row 3 (obstype=5): state 3 possible
        @test censor_pat[3, 1] == 5.0
        @test censor_pat[3, 2] == 0.0
        @test censor_pat[3, 3] == 0.0
        @test censor_pat[3, 4] == 1.0
    end
    
    @testset "expand_data_for_phasetype preserves covariates" begin
        data = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [2],
            obstype = [1],  # Exact - will be split
            x = [0.5],      # Covariate
            treatment = ["A"]
        )
        
        result = expand_data_for_phasetype(data, 2)
        expanded = result.expanded_data
        
        # Both rows should have the covariate values
        @test nrow(expanded) == 2
        @test expanded.x[1] == 0.5
        @test expanded.x[2] == 0.5
        @test expanded.treatment[1] == "A"
        @test expanded.treatment[2] == "A"
    end
    
    @testset "needs_data_expansion_for_phasetype detection" begin
        # Data with exact observations
        data_exact = DataFrame(
            id = [1], tstart = [0.0], tstop = [1.0],
            statefrom = [1], stateto = [2], obstype = [1]
        )
        @test needs_data_expansion_for_phasetype(data_exact) == true
        
        # Data with only panel observations
        data_panel = DataFrame(
            id = [1], tstart = [0.0], tstop = [1.0],
            statefrom = [1], stateto = [2], obstype = [2]
        )
        @test needs_data_expansion_for_phasetype(data_panel) == false
        
        # Mixed data
        data_mixed = DataFrame(
            id = [1, 1], tstart = [0.0, 1.0], tstop = [1.0, 2.0],
            statefrom = [1, 2], stateto = [2, 3], obstype = [2, 1]
        )
        @test needs_data_expansion_for_phasetype(data_mixed) == true
    end
    
    @testset "compute_expanded_subject_indices" begin
        expanded_data = DataFrame(
            id = [1, 1, 1, 2, 2],
            tstart = [0.0, 0.5, 1.0, 0.0, 1.0],
            tstop = [0.5, 1.0, 2.0, 1.0, 2.0],
            statefrom = [1, 1, 2, 1, 2],
            stateto = [0, 2, 3, 0, 3],
            obstype = [3, 1, 1, 3, 1]
        )
        
        subj_inds = compute_expanded_subject_indices(expanded_data)
        
        @test length(subj_inds) == 2
        @test subj_inds[1] == 1:3
        @test subj_inds[2] == 4:5
    end
end

# =============================================================================
# TEST SECTION 3: COXIAN HAZARD/SURVIVAL ANALYTIC TESTS
# =============================================================================

@testset "Coxian Phase-Type Analytic Correctness" begin
    
    @testset "build_coxian_intensity structure" begin
        # 3-phase Coxian: λ = [2.0, 1.5], μ = [0.3, 0.5, 1.0]
        λ = [2.0, 1.5]
        μ = [0.3, 0.5, 1.0]
        Q = build_coxian_intensity(λ, μ)
        
        # Q should be 4×4 (3 phases + absorbing)
        @test size(Q) == (4, 4)
        
        # Check diagonal elements: -(λᵢ + μᵢ) for i < n, -μₙ for last phase
        @test Q[1, 1] ≈ -(λ[1] + μ[1])  # -(2.0 + 0.3) = -2.3
        @test Q[2, 2] ≈ -(λ[2] + μ[2])  # -(1.5 + 0.5) = -2.0
        @test Q[3, 3] ≈ -μ[3]           # -1.0
        
        # Check progression rates (off-diagonal within transient states)
        @test Q[1, 2] ≈ λ[1]  # 2.0
        @test Q[2, 3] ≈ λ[2]  # 1.5
        
        # Check absorption rates (transition to absorbing state)
        @test Q[1, 4] ≈ μ[1]  # 0.3
        @test Q[2, 4] ≈ μ[2]  # 0.5
        @test Q[3, 4] ≈ μ[3]  # 1.0
        
        # Absorbing state row should be all zeros
        @test all(Q[4, :] .== 0.0)
        
        # Row sums should be zero (intensity matrix property)
        for i in 1:4
            @test isapprox(sum(Q[i, :]), 0.0; atol=1e-12)
        end
    end
    
    @testset "PhaseTypeDistribution accessors" begin
        # 2-phase Coxian
        λ = [1.5]
        μ = [0.5, 1.0]
        Q = build_coxian_intensity(λ, μ)
        initial = [1.0, 0.0]  # Start in phase 1
        
        ph = PhaseTypeDistribution(2, Q, initial)
        
        # Check subintensity
        S = subintensity(ph)
        @test size(S) == (2, 2)
        @test S[1, 1] ≈ -(λ[1] + μ[1])
        @test S[1, 2] ≈ λ[1]
        @test S[2, 1] ≈ 0.0
        @test S[2, 2] ≈ -μ[2]
        
        # Check absorption rates
        abs_rates = absorption_rates(ph)
        @test length(abs_rates) == 2
        @test abs_rates[1] ≈ μ[1]
        @test abs_rates[2] ≈ μ[2]
        
        # Check progression rates
        prog_rates = progression_rates(ph)
        @test length(prog_rates) == 1
        @test prog_rates[1] ≈ λ[1]
    end
    
    @testset "2-phase Coxian survival vs analytic" begin
        # Parameters: r (progression), a1 (absorption from phase 1), a2 (absorption from phase 2)
        r = 2.0
        a1 = 0.5
        a2 = 1.0
        
        # Build using our implementation
        λ = [r]
        μ = [a1, a2]
        Q = build_coxian_intensity(λ, μ)
        
        # Test at multiple time points
        test_times = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        
        for t in test_times
            # Analytic survival
            S_analytic = coxian2_survival(t, r, a1, a2)
            
            # Compute via matrix exponential
            S = Q[1:2, 1:2]  # Sub-intensity
            expSt = exp(S * t)
            S_matrix = sum(expSt[1, :])  # π = [1, 0], sum first row
            
            @test isapprox(S_matrix, S_analytic; rtol=1e-10) ||
                  isapprox(S_matrix, S_analytic; atol=1e-12)
        end
    end
    
    @testset "2-phase Coxian density vs analytic" begin
        r = 1.5
        a1 = 0.3
        a2 = 0.8
        
        λ = [r]
        μ = [a1, a2]
        Q = build_coxian_intensity(λ, μ)
        
        test_times = [0.1, 0.5, 1.0, 2.0, 3.0]
        
        for t in test_times
            # Analytic density
            f_analytic = coxian2_density(t, r, a1, a2)
            
            # Compute via matrix exponential: f(t) = π' exp(St) a
            S = Q[1:2, 1:2]
            a = μ
            expSt = exp(S * t)
            f_matrix = dot(expSt[1, :], a)  # π = [1, 0]
            
            @test isapprox(f_matrix, f_analytic; rtol=1e-10) ||
                  isapprox(f_matrix, f_analytic; atol=1e-12)
        end
    end
    
    @testset "2-phase Coxian hazard vs analytic" begin
        r = 2.0
        a1 = 0.4
        a2 = 1.2
        
        λ = [r]
        μ = [a1, a2]
        Q = build_coxian_intensity(λ, μ)
        
        test_times = [0.1, 0.5, 1.0, 2.0]
        
        for t in test_times
            # Analytic hazard
            h_analytic = coxian2_hazard(t, r, a1, a2)
            
            # Compute via matrix exponential: h(t) = f(t)/S(t)
            S = Q[1:2, 1:2]
            a = μ
            expSt = exp(S * t)
            f_matrix = dot(expSt[1, :], a)
            S_matrix = sum(expSt[1, :])
            h_matrix = f_matrix / S_matrix
            
            @test isapprox(h_matrix, h_analytic; rtol=1e-10) ||
                  isapprox(h_matrix, h_analytic; atol=1e-12)
        end
    end
    
    @testset "2-phase Coxian cumulative hazard vs analytic" begin
        r = 1.0
        a1 = 0.5
        a2 = 0.5
        
        test_times = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for t in test_times
            # Analytic cumulative hazard
            H_analytic = coxian2_cumul_hazard(t, r, a1, a2)
            
            # Via analytic survival
            S = coxian2_survival(t, r, a1, a2)
            H_from_S = -log(S)
            
            @test isapprox(H_analytic, H_from_S; rtol=1e-12)
        end
    end
    
    @testset "Coxian mean matches expected for exponential limit" begin
        # When n_phases = 1, Coxian reduces to exponential with rate = absorption rate
        μ = [2.0]
        Q = build_coxian_intensity(Float64[], μ)
        
        # Mean of exponential with rate 2.0 is 0.5
        expected_mean = 1.0 / μ[1]
        
        # Compute mean via integration: E[T] = ∫₀^∞ S(t) dt
        S = Q[1:1, 1:1]  # Just [-2.0]
        
        # For exponential, S(t) = exp(-2t), ∫S(t)dt = 1/2
        # Use numerical integration
        function survival_fn(t)
            return exp(S[1,1] * t)
        end
        
        using QuadGK
        mean_numerical, _ = quadgk(survival_fn, 0.0, Inf)
        
        @test isapprox(mean_numerical, expected_mean; rtol=1e-6)
    end
end

# =============================================================================
# TEST SECTION 4: LOGLIK ANALYTIC TESTS
# =============================================================================

@testset "Phase-Type Loglik Analytic Correctness" begin
    
    @testset "loglik_phasetype_expanded for exponential path" begin
        # Create a simple 1-phase (exponential) surrogate
        tmat = [0 1; 0 0]  # 1 → 2 (absorbing)
        config = PhaseTypeConfig(n_phases=[1, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        
        # Set rate = 2.0 for transition 1→2
        # Q = [-2  2]
        #     [ 0  0]
        rate = 2.0
        surrogate.expanded_Q[1, 1] = -rate
        surrogate.expanded_Q[1, 2] = rate
        
        # Path: state 1 for time T=0.5, then transition to state 2
        T = 0.5
        path = SamplePath(1, [0.0, T], [1, 2])
        
        # Analytic loglik: log(S(T)) + log(rate) = -rate*T + log(rate)
        expected_ll = -rate * T + log(rate)
        
        computed_ll = loglik_phasetype_expanded(path, surrogate)
        
        @test isapprox(computed_ll, expected_ll; rtol=1e-10)
    end
    
    @testset "loglik_phasetype_expanded for 2-phase path" begin
        # 2-phase Coxian for state 1, then absorbing state 2
        tmat = [0 1; 0 0]
        config = PhaseTypeConfig(n_phases=[2, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        
        # Set Q matrix:
        # Phases 1,2 are state 1; Phase 3 is state 2 (absorbing)
        # Q = [-(r+a1)   r      a1 ]
        #     [   0    -a2      a2 ]
        #     [   0      0       0 ]
        r = 1.5   # progression 1→2
        a1 = 0.3  # absorption from phase 1 to state 2
        a2 = 0.8  # absorption from phase 2 to state 2
        
        Q = zeros(3, 3)
        Q[1, 1] = -(r + a1)
        Q[1, 2] = r
        Q[1, 3] = a1
        Q[2, 2] = -a2
        Q[2, 3] = a2
        surrogate.expanded_Q .= Q
        
        # Path: phase 1 for t1=0.3, phase 2 for t2=0.5, then absorb to state 2
        t1 = 0.3
        t2 = 0.5
        path = SamplePath(1, [0.0, t1, t1 + t2], [1, 2, 3])
        
        # Analytic loglik:
        # Interval 1: phase 1 for t1, then transition to phase 2
        #   log(survival in phase 1) + log(rate 1→2)
        #   = (-|Q[1,1]| * t1) + log(r) = (-(r+a1)*t1) + log(r)
        # Interval 2: phase 2 for t2, then transition to phase 3 (absorbing)
        #   = (-|Q[2,2]| * t2) + log(a2) = (-a2*t2) + log(a2)
        
        expected_ll = -(r + a1) * t1 + log(r) - a2 * t2 + log(a2)
        
        computed_ll = loglik_phasetype_expanded(path, surrogate)
        
        @test isapprox(computed_ll, expected_ll; rtol=1e-10)
    end
    
    @testset "loglik_phasetype_expanded no transition path" begin
        tmat = [0 1; 0 0]
        config = PhaseTypeConfig(n_phases=[1, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        
        # Path with no transitions (just initial state)
        path = SamplePath(1, [0.0], [1])
        
        # No transitions → loglik = 0
        @test loglik_phasetype_expanded(path, surrogate) == 0.0
    end
    
    @testset "collapse_phasetype_path correctness" begin
        tmat = [0 1 0; 0 0 1; 0 0 0]
        config = PhaseTypeConfig(n_phases=[2, 2, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        absorbingstates = [3]
        
        # Expanded path: phase 1 → phase 2 → phase 3 → phase 5
        # Phases 1,2 map to state 1; phases 3,4 map to state 2; phase 5 maps to state 3
        expanded = SamplePath(1, [0.0, 0.5, 1.0, 1.5], [1, 2, 3, 5])
        collapsed = collapse_phasetype_path(expanded, surrogate, absorbingstates)
        
        # Should collapse to: state 1 (at 0.0) → state 2 (at 1.0) → state 3 (at 1.5)
        # Phase 1→2 transition is within state 1, so it's removed
        @test collapsed.states == [1, 2, 3]
        @test collapsed.times == [0.0, 1.0, 1.5]
    end
end

# =============================================================================
# TEST SECTION 5: EXPANDED Q MATRIX STRUCTURE
# =============================================================================

@testset "Expanded Q Matrix Structure" begin
    
    @testset "Row sums are zero" begin
        tmat = [0 1 1; 0 0 1; 0 0 0]  # 1→2, 1→3, 2→3
        config = PhaseTypeConfig(n_phases=[2, 2, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        
        Q = surrogate.expanded_Q
        n = surrogate.n_expanded_states
        
        for i in 1:n
            @test isapprox(sum(Q[i, :]), 0.0; atol=1e-12)
        end
    end
    
    @testset "Absorbing state has no outgoing transitions" begin
        tmat = [0 1; 0 0]
        config = PhaseTypeConfig(n_phases=[2, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        
        Q = surrogate.expanded_Q
        # State 2 (absorbing) maps to phase 3
        absorbing_phase = first(surrogate.state_to_phases[2])
        
        @test all(Q[absorbing_phase, :] .== 0.0)
    end
    
    @testset "Transient state diagonals are negative" begin
        tmat = [0 1 0; 0 0 1; 0 0 0]
        config = PhaseTypeConfig(n_phases=[3, 2, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        
        Q = surrogate.expanded_Q
        
        # All transient phases (not the absorbing state) should have negative diagonal
        for s in 1:2  # States 1 and 2 are transient
            for p in surrogate.state_to_phases[s]
                @test Q[p, p] < 0
            end
        end
    end
    
    @testset "Phase progression structure is Coxian" begin
        # For a Coxian within each state, only forward transitions allowed
        tmat = [0 1; 0 0]
        config = PhaseTypeConfig(n_phases=[3, 1])
        surrogate = build_phasetype_surrogate(tmat, config)
        
        Q = surrogate.expanded_Q
        phases = surrogate.state_to_phases[1]  # 1:3
        
        # Check that phase i only transitions to phase i+1 (within state)
        # or to the destination state
        for i in 1:2  # phases 1 and 2
            p = phases[i]
            # Should have positive rate to next phase
            @test Q[p, p + 1] > 0
            # Should NOT have positive rate to previous phases
            if i > 1
                @test Q[p, p - 1] == 0.0
            end
        end
    end
end

println("\nPhase-Type Correctness Tests Complete")

# =============================================================================
# Phase-Type Importance Sampling Tests (from test_phasetype_is.jl)
# =============================================================================

# Unit tests for phase-type importance sampling
using Test
using MultistateModels
using Random
using DataFrames
using Statistics
using LinearAlgebra

@testset "Phase-Type Importance Sampling" begin

    # -------------------------------------------------------------------------
    # Test 1: IS weights average to 1 when target = proposal (Markov model)
    #
    # When the target model is Markov (exponential hazards) AND the target
    # parameters equal the Markov surrogate MLE, importance weights should
    # average to 1.
    # -------------------------------------------------------------------------
    @testset "IS weights = 1 when target = proposal (Markov)" begin
        Random.seed!(12345)
        
        # Simple 2-state model with exponential hazards (Markov)
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # Panel data - larger dataset for stability
        n_subj = 50
        dat = DataFrame(
            id = repeat(1:n_subj, inner=3),
            tstart = repeat([0.0, 1.0, 2.0], n_subj),
            tstop = repeat([1.0, 2.0, 3.0], n_subj),
            statefrom = repeat([1, 1, 1], n_subj),
            stateto = vcat([[rand() < 0.3 ? 2 : 1, rand() < 0.5 ? 2 : 1, 2] for _ in 1:n_subj]...),
            obstype = repeat([2, 2, 2], n_subj)
        )
        
        model = multistatemodel(h12; data=dat)
        
        # Fit Markov surrogate to get the MLE
        surrogate_fitted = MultistateModels.fit_surrogate(model; verbose=false)
        
        # Set target parameters to EXACTLY match the Markov surrogate MLE
        # This makes target = proposal
        MultistateModels.set_parameters!(model, [collect(surrogate_fitted.parameters[1])])
        
        # Build phase-type with 1 phase per state (equivalent to Markov)
        tmat = model.tmat
        phasetype_config = MultistateModels.PhaseTypeConfig(n_phases=[1, 1])
        surrogate = MultistateModels.build_phasetype_surrogate(tmat, phasetype_config)
        
        # Update phase-type Q matrix to match the fitted Markov surrogate
        # The phase-type rate should equal the Markov rate
        markov_rate = exp(surrogate_fitted.parameters[1][1])
        surrogate.expanded_Q[1, 1] = -markov_rate
        surrogate.expanded_Q[1, 2] = markov_rate
        
        # Build infrastructure
        emat_ph = MultistateModels.build_phasetype_emat_expanded(model, surrogate)
        books = MultistateModels.build_tpm_mapping(model.data)
        absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
        tpm_book_ph, hazmat_book_ph = MultistateModels.build_phasetype_tpm_book(surrogate, books, model.data)
        fbmats_ph = MultistateModels.build_fbmats_phasetype(model, surrogate)
        
        # Compute marginal likelihood under phase-type
        ll_marginal = MultistateModels.compute_phasetype_marginal_loglik(model, surrogate, emat_ph)
        
        # Sample many paths and compute importance weights for subject 1
        n_paths = 500
        log_weights = Float64[]
        
        for _ in 1:n_paths
            path_result = MultistateModels.draw_samplepath_phasetype(
                1, model, tpm_book_ph, hazmat_book_ph, books[2], 
                fbmats_ph, emat_ph, surrogate, absorbingstates)
            
            params = MultistateModels.get_hazard_params(model.parameters)
            ll_target = MultistateModels.loglik(params, path_result.collapsed, model.hazards, model)
            ll_surrog = MultistateModels.loglik_phasetype_expanded(path_result.expanded, surrogate)
            push!(log_weights, ll_target - ll_surrog)
        end
        
        weights = exp.(log_weights)
        mean_weight = mean(weights)

        # IS estimate for subject 1
        ll_is = ll_marginal + log(mean_weight)
        
        # CRITICAL: When target = proposal, each weight must be EXACTLY 1.0
        # This is not a statistical property - it's an algebraic identity.
        # log_weight = ll_target - ll_proposal = 0 when target ≡ proposal
        # Therefore weight = exp(0) = 1.0 exactly.
        @test all(w -> isapprox(w, 1.0; atol=1e-10), weights)
        @test mean_weight ≈ 1.0 atol=1e-10
        
        # IS estimate must equal marginal likelihood exactly (log(1.0) = 0)
        @test ll_is ≈ ll_marginal atol=1e-10
    end

    # -------------------------------------------------------------------------
    # Test 2: collapse_phasetype_path correctness
    # -------------------------------------------------------------------------
    @testset "collapse_phasetype_path" begin
        # Create a simple surrogate for testing
        tmat = [0 1 0; 0 0 1; 0 0 0]
        config = MultistateModels.PhaseTypeConfig(n_phases=[2, 2, 1])
        surrogate = MultistateModels.build_phasetype_surrogate(tmat, config)
        absorbingstates = [3]
        
        # Test 1: Path that stays in one observed state (phases 1→2→1)
        expanded = MultistateModels.SamplePath(1, [0.0, 0.5, 1.0], [1, 2, 1])
        collapsed = MultistateModels.collapse_phasetype_path(expanded, surrogate, absorbingstates)
        @test collapsed.states == [1]  # All phases map to state 1
        @test collapsed.times == [0.0]
        
        # Test 2: Path with transition between observed states
        expanded = MultistateModels.SamplePath(1, [0.0, 0.5, 1.0, 1.5], [1, 2, 3, 5])
        collapsed = MultistateModels.collapse_phasetype_path(expanded, surrogate, absorbingstates)
        @test collapsed.states == [1, 2, 3]
        @test collapsed.times == [0.0, 1.0, 1.5]
        
        # Test 3: Path that ends in absorbing state - should truncate
        expanded = MultistateModels.SamplePath(1, [0.0, 1.0, 2.0], [1, 3, 5])
        collapsed = MultistateModels.collapse_phasetype_path(expanded, surrogate, absorbingstates)
        @test collapsed.states[end] == 3  # Absorbing state
        @test length(collapsed.states) == 3
    end

    # -------------------------------------------------------------------------
    # Test 3: loglik_phasetype_expanded correctness
    # -------------------------------------------------------------------------
    @testset "loglik_phasetype_expanded" begin
        # Create surrogate
        tmat = [0 1 0; 0 0 1; 0 0 0]
        config = MultistateModels.PhaseTypeConfig(n_phases=[1, 1, 1])
        surrogate = MultistateModels.build_phasetype_surrogate(tmat, config)
        
        # Set Q matrix manually for testing
        # Q = [-1  1  0]
        #     [ 0 -2  2]
        #     [ 0  0  0]
        Q = [-1.0 1.0 0.0; 0.0 -2.0 2.0; 0.0 0.0 0.0]
        surrogate.expanded_Q .= Q
        
        # Simple path: state 1 for 0.5 time, then transition to state 2
        path = MultistateModels.SamplePath(1, [0.0, 0.5, 1.0], [1, 2, 3])
        
        # Manual calculation:
        # Interval 1: state 1 for 0.5 time, then transition to 2
        #   survival: exp(-1 * 0.5) → log = -0.5
        #   transition: log(1) = 0
        # Interval 2: state 2 for 0.5 time, then transition to 3
        #   survival: exp(-2 * 0.5) → log = -1.0
        #   transition: log(2)
        # Total: -0.5 + 0 - 1.0 + log(2) = -1.5 + 0.693 = -0.807
        
        ll = MultistateModels.loglik_phasetype_expanded(path, surrogate)
        expected = -0.5 + 0.0 - 1.0 + log(2.0)
        @test ll ≈ expected atol=1e-10
        
        # Path with no transitions
        path_none = MultistateModels.SamplePath(1, [0.0], [1])
        ll_none = MultistateModels.loglik_phasetype_expanded(path_none, surrogate)
        @test ll_none == 0.0  # No transitions = log(1) = 0
    end

    # -------------------------------------------------------------------------
    # Test 4: draw_samplepath_phasetype produces valid paths
    # -------------------------------------------------------------------------
    @testset "draw_samplepath_phasetype validity" begin
        Random.seed!(54321)
        
        # Illness-death model
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        # Mix of exact and panel observations
        dat = DataFrame(
            id = [1, 1, 1, 1],
            tstart = [0.0, 0.5, 1.0, 1.5],
            tstop = [0.5, 1.0, 1.5, 2.0],
            statefrom = [1, 1, 2, 2],
            stateto = [1, 2, 2, 3],
            obstype = [2, 1, 2, 1]  # Panel, Exact, Panel, Exact
        )
        
        model = multistatemodel(h12, h13, h23; data=dat)
        
        # Build phase-type
        tmat = model.tmat
        config = MultistateModels.PhaseTypeConfig(n_phases=[2, 2, 1])
        surrogate = MultistateModels.build_phasetype_surrogate(tmat, config)
        
        emat_ph = MultistateModels.build_phasetype_emat_expanded(model, surrogate)
        books = MultistateModels.build_tpm_mapping(model.data)
        absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
        tpm_book_ph, hazmat_book_ph = MultistateModels.build_phasetype_tpm_book(surrogate, books, model.data)
        fbmats_ph = MultistateModels.build_fbmats_phasetype(model, surrogate)
        
        # Sample paths and verify validity
        for _ in 1:20
            result = MultistateModels.draw_samplepath_phasetype(
                1, model, tpm_book_ph, hazmat_book_ph, books[2],
                fbmats_ph, emat_ph, surrogate, absorbingstates)
            
            # Collapsed path should start at initial state
            @test result.collapsed.states[1] == 1
            
            # Collapsed path times should be monotonically increasing
            @test issorted(result.collapsed.times)
            
            # Expanded path times should be monotonically increasing
            @test issorted(result.expanded.times)
            
            # Expanded states should be valid phase indices
            @test all(1 .<= result.expanded.states .<= surrogate.n_expanded_states)
            
            # Collapsed states should match observed transitions
            # (path must pass through observed states at observed times)
            collapsed = result.collapsed
            
            # For exact observations, the transition should be recorded
            # Check that state 2 appears after state 1
            idx_state2 = findfirst(==(2), collapsed.states)
            if !isnothing(idx_state2)
                @test all(collapsed.states[1:idx_state2-1] .== 1)
            end
        end
    end

    # -------------------------------------------------------------------------
    # Test 5: IS produces negative log-likelihood for semi-Markov model
    #
    # When using phase-type IS for a Weibull (semi-Markov) model, the IS
    # estimate should be a reasonable negative log-likelihood.
    # -------------------------------------------------------------------------
    @testset "IS estimate negative for semi-Markov" begin
        Random.seed!(99999)
        
        # Weibull model (semi-Markov)
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        # Panel data
        n_subj = 30
        dat = DataFrame(
            id = repeat(1:n_subj, inner=2),
            tstart = repeat([0.0, 1.0], n_subj),
            tstop = repeat([1.0, 2.0], n_subj),
            statefrom = repeat([1, 1], n_subj),
            stateto = vcat([[rand() < 0.4 ? 2 : 1, 2] for _ in 1:n_subj]...),
            obstype = repeat([2, 2], n_subj)
        )
        
        model = multistatemodel(h12; data=dat)
        
        # Build phase-type surrogate with 2 phases
        tmat = model.tmat
        config = MultistateModels.PhaseTypeConfig(n_phases=[2, 1])
        surrogate = MultistateModels.build_phasetype_surrogate(tmat, config)
        
        emat_ph = MultistateModels.build_phasetype_emat_expanded(model, surrogate)
        books = MultistateModels.build_tpm_mapping(model.data)
        absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
        tpm_book_ph, hazmat_book_ph = MultistateModels.build_phasetype_tpm_book(surrogate, books, model.data)
        fbmats_ph = MultistateModels.build_fbmats_phasetype(model, surrogate)
        
        ll_marginal = MultistateModels.compute_phasetype_marginal_loglik(model, surrogate, emat_ph)
        
        # Sample paths for first subject
        n_paths = 300
        log_weights = Float64[]
        
        for _ in 1:n_paths
            result = MultistateModels.draw_samplepath_phasetype(
                1, model, tpm_book_ph, hazmat_book_ph, books[2],
                fbmats_ph, emat_ph, surrogate, absorbingstates)
            
            params = MultistateModels.get_hazard_params(model.parameters)
            ll_target = MultistateModels.loglik(params, result.collapsed, model.hazards, model)
            ll_surrog = MultistateModels.loglik_phasetype_expanded(result.expanded, surrogate)
            push!(log_weights, ll_target - ll_surrog)
        end
        
        weights = exp.(log_weights)
        ll_is = ll_marginal + log(mean(weights))
        
        # For semi-Markov models, IS estimate should be finite
        # (not -Inf which would indicate a mismatch between target and proposal support)
        @test isfinite(ll_is)
        
        # The IS correction (difference between ll_is and ll_marginal) should be moderate
        # Large corrections indicate proposal mismatch
        @test abs(ll_is - ll_marginal) < 10.0
    end

    # -------------------------------------------------------------------------
    # Test: Coxian structure options
    #
    # Tests that the three Coxian structure options (:allequal, :prop_to_prog, 
    # :unstructured) produce valid PhaseTypeDistribution objects with correct
    # structure.
    # -------------------------------------------------------------------------
    @testset "Coxian structure options" begin
        total_rate = 1.5
        n_phases = 3
        
        # Test :allequal structure
        @testset ":allequal structure" begin
            ph = MultistateModels._build_coxian_from_rate(n_phases, total_rate; structure=:allequal)
            
            @test ph.n_phases == n_phases
            @test size(ph.Q) == (n_phases + 1, n_phases + 1)  # Q includes absorbing state
            @test ph.initial == [1.0, 0.0, 0.0]  # Start in phase 1
            
            # Get subintensity for checking transient rates
            S = MultistateModels.subintensity(ph)
            
            # Check structure: all progression rates should be equal
            prog_rates = [S[i, i+1] for i in 1:n_phases-1]
            @test all(prog_rates .≈ prog_rates[1])
            
            # Check structure: all absorption rates should be equal
            abs_rates = MultistateModels.absorption_rates(ph)
            @test all(abs_rates .≈ abs_rates[1])
            
            # Diagonal should be negative for transient states
            @test all(diag(S) .< 0)
            
            # Last row of Q (absorbing state) should be zeros
            @test all(ph.Q[end, :] .== 0)
        end
        
        # Test :prop_to_prog structure
        @testset ":prop_to_prog structure" begin
            ph = MultistateModels._build_coxian_from_rate(n_phases, total_rate; structure=:prop_to_prog)
            
            @test ph.n_phases == n_phases
            @test size(ph.Q) == (n_phases + 1, n_phases + 1)
            @test ph.initial == [1.0, 0.0, 0.0]
            
            # Get subintensity for checking transient rates
            S = MultistateModels.subintensity(ph)
            
            # Check structure: absorption rates proportional to progression rates
            # For phases 1 to n-1: a_i = c * r_i where c is constant
            prog_rates = [S[i, i+1] for i in 1:n_phases-1]
            abs_rates = MultistateModels.absorption_rates(ph)
            
            # Compute ratios a_i / r_i for phases 1 to n-1
            ratios = [abs_rates[i] / prog_rates[i] for i in 1:n_phases-1]
            @test all(ratios .≈ ratios[1])  # All ratios should be equal (= c)
            
            # Diagonal should be negative
            @test all(diag(S) .< 0)
        end
        
        # Test :unstructured structure
        @testset ":unstructured structure" begin
            ph = MultistateModels._build_coxian_from_rate(n_phases, total_rate; structure=:unstructured)
            
            @test ph.n_phases == n_phases
            @test size(ph.Q) == (n_phases + 1, n_phases + 1)
            @test ph.initial == [1.0, 0.0, 0.0]
            
            # Get subintensity for checking transient rates
            S = MultistateModels.subintensity(ph)
            
            # Check structure: rates can vary (decreasing progression, increasing absorption)
            prog_rates = [S[i, i+1] for i in 1:n_phases-1]
            abs_rates = MultistateModels.absorption_rates(ph)
            
            # Progression rates should be decreasing
            @test prog_rates[1] > prog_rates[2]
            
            # Absorption rates (for phases 1 to n-1) should be increasing
            @test abs_rates[1] < abs_rates[2]
            
            # Diagonal should be negative
            @test all(diag(S) .< 0)
        end
        
        # Test single phase (all structures should be equivalent)
        @testset "Single phase equivalence" begin
            ph1 = MultistateModels._build_coxian_from_rate(1, total_rate; structure=:allequal)
            ph2 = MultistateModels._build_coxian_from_rate(1, total_rate; structure=:prop_to_prog)
            ph3 = MultistateModels._build_coxian_from_rate(1, total_rate; structure=:unstructured)
            
            @test ph1.Q ≈ ph2.Q ≈ ph3.Q
            @test ph1.Q[1,1] ≈ -total_rate  # Diagonal of transient state
            @test ph1.Q[1,2] ≈ total_rate   # Absorption rate to absorbing state
        end
        
        # Test PhaseTypeConfig structure field
        @testset "PhaseTypeConfig structure field" begin
            config1 = MultistateModels.PhaseTypeConfig(n_phases=3, structure=:allequal)
            @test config1.structure == :allequal
            
            config2 = MultistateModels.PhaseTypeConfig(n_phases=3, structure=:prop_to_prog)
            @test config2.structure == :prop_to_prog
            
            config3 = MultistateModels.PhaseTypeConfig(n_phases=3, structure=:unstructured)
            @test config3.structure == :unstructured
            
            # Default should be :unstructured
            config_default = MultistateModels.PhaseTypeConfig(n_phases=3)
            @test config_default.structure == :unstructured
            
            # Invalid structure should throw
            @test_throws ArgumentError MultistateModels.PhaseTypeConfig(n_phases=3, structure=:invalid)
        end
        
        # Test ProposalConfig structure field
        @testset "ProposalConfig structure field" begin
            config1 = MultistateModels.ProposalConfig(type=:phasetype, structure=:prop_to_prog)
            @test config1.structure == :prop_to_prog
            
            # Default should be :unstructured
            config_default = MultistateModels.ProposalConfig(type=:phasetype)
            @test config_default.structure == :unstructured
            
            # PhaseTypeProposal should forward structure
            config2 = MultistateModels.PhaseTypeProposal(n_phases=3, structure=:allequal)
            @test config2.structure == :allequal
        end
    end

end

# =============================================================================
# Phase-Type Model Fitting and Parameter Handling Tests (from test_phasetype_fitting.jl)
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using LinearAlgebra

@testset "Phase-Type Model Fitting" begin
    Random.seed!(42)
    
    # Create realistic test data for a 1→2→3 progressive model
    n_subjects = 100
    
    # Generate data where subjects transition 1→2→3
    data_rows = []
    for i in 1:n_subjects
        t1 = rand() * 2  # Time in state 1
        t2 = rand() * 2  # Additional time in state 2
        
        push!(data_rows, (id=i, tstart=0.0, tstop=t1, statefrom=1, stateto=2, obstype=1))
        push!(data_rows, (id=i, tstart=t1, tstop=t1+t2, statefrom=2, stateto=3, obstype=1))
    end
    data = DataFrame(data_rows)
    
    @testset "Parameter Accessor Functions" begin
        # Build model with phase-type hazard
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=3, coxian_structure=:unstructured)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data)
        
        # Test get_parameters returns user-facing parameters
        params = get_parameters(model)
        @test haskey(params, :h12)  # Phase-type hazard
        @test haskey(params, :h23)  # Exponential hazard
        
        # Test get_expanded_parameters returns internal parameters
        exp_params = get_expanded_parameters(model)
        @test haskey(exp_params, :h1_prog1)  # λ₁
        @test haskey(exp_params, :h1_prog2)  # λ₂
        @test haskey(exp_params, :h12_exit1) # μ₁
        @test haskey(exp_params, :h12_exit2) # μ₂
        @test haskey(exp_params, :h12_exit3) # μ₃
        @test haskey(exp_params, :h23)       # h23 transition
        
        # Test get_parameters_flat returns flat vector
        flat = get_parameters_flat(model)
        @test flat isa Vector{Float64}
        @test length(flat) == 6  # 5 phase-type params + 1 exp param
        
        # Test get_parameters_nested
        nested = get_parameters_nested(model)
        @test nested isa NamedTuple
        
        # Test get_parameters_natural
        natural = get_parameters_natural(model)
        @test natural isa NamedTuple
        
        # Test get_unflatten_fn
        unflatten = get_unflatten_fn(model)
        @test unflatten isa Function
        
        # Test round-trip: flatten → unflatten
        restored = unflatten(flat)
        for key in keys(nested)
            # Compare baseline parameter values (now NamedTuples with named fields)
            for parname in keys(nested[key].baseline)
                @test isapprox(nested[key].baseline[parname], restored[key].baseline[parname]; atol=1e-10)
            end
            # Compare covariate parameters if present
            if haskey(nested[key], :covariates)
                for parname in keys(nested[key].covariates)
                    @test isapprox(nested[key].covariates[parname], restored[key].covariates[parname]; atol=1e-10)
                end
            end
        end
    end
    
    @testset "set_parameters! with Vector{Vector}" begin
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data)
        initialize_parameters!(model)
        
        # Set new values using Vector{Vector} format
        # h12: [λ₁, μ₁, μ₂] (2n-1 = 3 params for n=2)
        # h23: [rate]
        new_values = [
            [log(0.5), log(0.3), log(0.4)],  # h12
            [log(0.6)]                        # h23
        ]
        
        set_parameters!(model, new_values)
        
        # Verify user-facing parameters were updated
        params = get_parameters(model)
        @test isapprox(params[:h12][1], 0.5; rtol=1e-6)
        @test isapprox(params[:h12][2], 0.3; rtol=1e-6)
        @test isapprox(params[:h12][3], 0.4; rtol=1e-6)
        @test isapprox(params[:h23][1], 0.6; rtol=1e-6)
        
        # Verify expanded parameters were also updated
        exp_params = get_expanded_parameters(model)
        @test isapprox(exp_params[:h1_prog1][1], 0.5; rtol=1e-6)
        @test isapprox(exp_params[:h12_exit1][1], 0.3; rtol=1e-6)
        @test isapprox(exp_params[:h12_exit2][1], 0.4; rtol=1e-6)
    end
    
    @testset "Initialization Respects Structure" begin
        # Test all three structures initialize with uniform rates
        for structure in [:unstructured, :allequal, :prop_to_prog]
            h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=3, coxian_structure=structure)
            h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
            
            model = multistatemodel(h12, h23; data=data)
            initialize_parameters!(model)
            
            exp_params = get_expanded_parameters(model)
            
            # Get all phase-type rates
            λ1 = exp_params[:h1_prog1][1]
            λ2 = exp_params[:h1_prog2][1]
            μ1 = exp_params[:h12_exit1][1]
            μ2 = exp_params[:h12_exit2][1]
            μ3 = exp_params[:h12_exit3][1]
            
            all_rates = [λ1, λ2, μ1, μ2, μ3]
            rate_range = maximum(all_rates) - minimum(all_rates)
            
            # All rates should be equal for all built-in structures
            @test rate_range < 1e-10
        end
    end
    
    @testset "Basic Fitting" begin
        # Simple 2-phase model
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:allequal)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data)
        
        # Fit the model
        fitted = fit(model; verbose=false, compute_vcov=false)
        
        # Check return type - now returns MultistateModelFitted
        @test fitted isa MultistateModels.MultistateModelFitted
        
        # Check it's detected as phase-type fitted
        @test is_phasetype_fitted(fitted)
        
        # Check convergence
        @test get_convergence(fitted) == true
        
        # Check loglikelihood is finite
        @test isfinite(get_loglik(fitted))
        
        # Check parameters are returned (user-facing phase-type params by default)
        params = get_parameters(fitted)
        @test haskey(params, :h12)
        @test haskey(params, :h23)
        
        # Check parameters are positive (natural scale)
        @test all(params[:h12] .> 0)
        @test all(params[:h23] .> 0)
        
        # Check expanded parameters are accessible
        exp_params = get_parameters(fitted; expanded=true)
        @test haskey(exp_params, :h1_prog1)
        @test haskey(exp_params, :h12_exit1)
    end
    
    @testset "Fitting with Variance-Covariance" begin
        # Larger dataset for stable vcov
        n_large = 200
        data_rows_large = []
        for i in 1:n_large
            t1 = rand() * 2
            t2 = rand() * 2
            push!(data_rows_large, (id=i, tstart=0.0, tstop=t1, statefrom=1, stateto=2, obstype=1))
            push!(data_rows_large, (id=i, tstart=t1, tstop=t1+t2, statefrom=2, stateto=3, obstype=1))
        end
        data_large = DataFrame(data_rows_large)
        
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data_large)
        fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=false)
        
        # Check vcov is returned
        vcov = get_vcov(fitted)
        @test !isnothing(vcov)
        @test vcov isa Matrix{Float64}
        
        # Vcov should be symmetric
        @test issymmetric(vcov)
        
        # Diagonal elements should be positive (variances)
        @test all(diag(vcov) .>= 0)
    end
    
    @testset "Access Fitted Expanded Model" begin
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data)
        fitted = fit(model; verbose=false, compute_vcov=false)
        
        # The fitted model IS the expanded model now (no wrapper)
        @test fitted isa MultistateModels.MultistateModelFitted
        
        # The hazards should have the internal hazard structure
        @test length(fitted.hazards) == 4  # 1 prog + 2 exit + 1 h23
        
        # Loglikelihood should be accessible
        @test isfinite(get_loglik(fitted))
        
        # Access mappings
        mappings = get_mappings(fitted)
        @test !isnothing(mappings)
        
        # Access original data
        orig_data = get_original_data(fitted)
        @test orig_data isa DataFrame
        
        # Access original tmat
        orig_tmat = get_original_tmat(fitted)
        @test orig_tmat isa Matrix{Int64}
    end
    
    @testset "Parameter Round-Trip After Fitting" begin
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data)
        fitted = fit(model; verbose=false, compute_vcov=false)
        
        # Get user-facing parameters
        params = get_parameters(fitted)
        
        # Get expanded parameters
        exp_params = get_parameters(fitted; expanded=true)
        
        # The h23 rate should match in both representations
        @test isapprox(params[:h23][1], exp_params[:h23][1]; rtol=1e-6)
        
        # Can also use explicit accessor
        pt_params = get_phasetype_parameters(fitted)
        @test pt_params == params
    end
end

# =============================================================================
# Phase-Type Simulation Tests (from test_phasetype_simulation.jl)
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random

@testset "Phase-Type Simulation" begin
    Random.seed!(42)
    
    # Create test data for a 1→2→3 progressive model
    n_subjects = 20
    data_rows = []
    for i in 1:n_subjects
        t1 = rand() * 2
        t2 = rand() * 2
        push!(data_rows, (id=i, tstart=0.0, tstop=t1, statefrom=1, stateto=2, obstype=1))
        push!(data_rows, (id=i, tstart=t1, tstop=t1+t2, statefrom=2, stateto=3, obstype=1))
    end
    data = DataFrame(data_rows)
    
    # Build phase-type model with 2 phases on 1→2 transition
    h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    model = multistatemodel(h12, h23; data=data)
    initialize_parameters!(model)
    
    @testset "State Space Structure" begin
        # Verify mappings are correct
        @test model.mappings.n_observed == 3
        @test model.mappings.n_expanded == 4  # 2 phases for state 1, 1 for state 2, 1 for state 3
        @test model.mappings.phase_to_state == [1, 1, 2, 3]
        @test model.mappings.n_phases_per_state == [2, 1, 1]
    end
    
    @testset "simulate_path" begin
        @testset "expanded=true returns phase-level states" begin
            Random.seed!(123)
            path = simulate_path(model, 1; expanded=true)
            
            @test path isa MultistateModels.SamplePath
            @test path.subj == 1
            @test length(path.times) == length(path.states)
            @test all(s in 1:4 for s in path.states)  # States should be in expanded space
        end
        
        @testset "expanded=false returns observed states" begin
            Random.seed!(123)
            path = simulate_path(model, 1; expanded=false)
            
            @test path isa MultistateModels.SamplePath
            @test path.subj == 1
            @test all(s in 1:3 for s in path.states)  # States should be in original space
        end
        
        @testset "collapsed path merges phases correctly" begin
            # Find a case where expanded path has multiple phases in same observed state
            found_merge = false
            for seed in 1:100
                Random.seed!(seed)
                path_exp = simulate_path(model, 1; expanded=true)
                
                # Check if any consecutive expanded states map to same observed state
                observed = [model.mappings.phase_to_state[s] for s in path_exp.states]
                for i in 2:length(observed)
                    if observed[i] == observed[i-1]
                        Random.seed!(seed)
                        path_col = simulate_path(model, 1; expanded=false)
                        
                        # Collapsed should have fewer states
                        @test length(path_col.states) < length(path_exp.states)
                        found_merge = true
                        break
                    end
                end
                found_merge && break
            end
            @test found_merge  # We should find at least one case
        end
        
        @testset "subject index validation" begin
            @test_throws ArgumentError simulate_path(model, 0)
            @test_throws ArgumentError simulate_path(model, n_subjects + 1)
        end
    end
    
    @testset "simulate" begin
        @testset "returns Vector{DataFrame} for data only" begin
            datasets = simulate(model; nsim=3, data=true, paths=false)
            
            @test datasets isa Vector{DataFrame}
            @test length(datasets) == 3
            @test all(d isa DataFrame for d in datasets)
        end
        
        @testset "returns tuple for data and paths" begin
            dat, paths = simulate(model; nsim=2, data=true, paths=true)
            
            @test dat isa Vector{DataFrame}
            @test paths isa Vector{Vector{MultistateModels.SamplePath}}
            @test length(dat) == 2
            @test length(paths) == 2
        end
        
        @testset "expanded=true returns expanded state space" begin
            Random.seed!(456)
            dat_exp, paths_exp = simulate(model; nsim=2, data=true, paths=true, expanded=true)
            
            # States should be in expanded space (1-4)
            for path_set in paths_exp
                for path in path_set
                    @test all(s in 1:4 for s in path.states)
                end
            end
        end
        
        @testset "expanded=false returns collapsed state space" begin
            Random.seed!(456)
            dat_col, paths_col = simulate(model; nsim=2, data=true, paths=true, expanded=false)
            
            # States should be in original space (1-3)
            for path_set in paths_col
                for path in path_set
                    @test all(s in 1:3 for s in path.states)
                end
            end
        end
    end
    
    @testset "simulate_data" begin
        datasets = simulate_data(model; nsim=3)
        
        @test datasets isa Vector{DataFrame}
        @test length(datasets) == 3
        
        # Collapsed data should have states in original space
        for df in datasets
            @test all(s in 1:3 for s in df.statefrom)
            @test all(s in 1:3 for s in df.stateto)
        end
        
        # Expanded data should have states in expanded space
        datasets_exp = simulate_data(model; nsim=3, expanded=true)
        for df in datasets_exp
            @test all(s in 1:4 for s in df.statefrom)
            @test all(s in 1:4 for s in df.stateto)
        end
    end
    
    @testset "simulate_paths" begin
        paths = simulate_paths(model; nsim=3)
        
        @test paths isa Vector{Vector{MultistateModels.SamplePath}}
        @test length(paths) == 3
        
        # Each simulation should have paths for all subjects
        for path_set in paths
            @test length(path_set) == n_subjects
        end
    end
    
    @testset "Path collapsing correctness" begin
        # Test that collapsing merges consecutive same-state phases
        phase_to_state = model.mappings.phase_to_state
        
        # Create a synthetic expanded path that should collapse
        expanded_path = MultistateModels.SamplePath(1, [0.0, 0.5, 1.0, 1.5], [1, 2, 3, 4])
        # Phase 1 -> State 1, Phase 2 -> State 1, Phase 3 -> State 2, Phase 4 -> State 3
        # Expected: [1, 2, 3] at times [0.0, 1.0, 1.5]
        
        collapsed = MultistateModels._collapse_path(expanded_path, model.mappings)
        
        @test collapsed.states == [1, 2, 3]
        @test collapsed.times == [0.0, 1.0, 1.5]
    end
    
    @testset "Data collapsing correctness" begin
        # Create a synthetic expanded DataFrame
        expanded_df = DataFrame(
            id = [1, 1, 1, 1],
            tstart = [0.0, 0.5, 1.0, 1.5],
            tstop = [0.5, 1.0, 1.5, 2.0],
            statefrom = [1, 2, 3, 4],
            stateto = [2, 3, 4, 4],
            obstype = [1, 1, 1, 3]  # Last is censored
        )
        
        collapsed = MultistateModels._collapse_data(expanded_df, model.mappings)
        
        # First two rows (phases 1→2, 2→3) both map to state 1 staying in state 1
        # Then state 1 → state 2 (phase 3 maps to state 2)
        # Then state 2 → state 3 (phase 4 maps to state 3)
        
        @test nrow(collapsed) <= nrow(expanded_df)
        @test all(s in 1:3 for s in collapsed.statefrom)
        @test all(s in 1:3 for s in collapsed.stateto)
    end
end

println("\nPhase-Type Tests Complete")
