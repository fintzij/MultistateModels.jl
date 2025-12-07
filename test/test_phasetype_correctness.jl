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
