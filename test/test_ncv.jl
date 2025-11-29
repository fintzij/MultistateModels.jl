# =============================================================================
# NCV (Neighbourhood Cross-Validation) Tests
# =============================================================================
#
# Tests for the NCV framework implementing Wood (2024) "Neighbourhood Cross 
# Validation" (arXiv:2404.16490).
#
# Test categories:
#   1. Cholesky downdate algorithm
#   2. LOO perturbation computation methods
#   3. NCV criterion computation
#   4. Degeneracy detection
#   5. Integration with IJ/JK variance estimation

using LinearAlgebra
using Random

# =============================================================================
# 1. Cholesky Downdate Tests
# =============================================================================

@testset "Cholesky Downdate" begin
    
    @testset "Basic rank-1 downdate" begin
        # Create a positive definite matrix
        Random.seed!(12345)
        n = 5
        A = randn(n, n)
        H = A' * A + 2.0 * I  # Ensure positive definite
        
        # Create a vector for downdate (small enough to keep result PD)
        v = 0.3 * randn(n)
        
        # Expected result: H - v*v'
        H_updated_expected = H - v * v'
        
        # Compute via Cholesky downdate
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        success = cholesky_downdate!(L_copy, v)
        
        @test success == true
        
        # Verify L_copy' * L_copy ≈ H - v*v'
        H_updated = L_copy * L_copy'
        @test H_updated ≈ H_updated_expected atol=1e-10
    end
    
    @testset "Downdate preserves positive definiteness" begin
        Random.seed!(23456)
        n = 4
        A = randn(n, n)
        H = A' * A + 5.0 * I
        
        # Small downdate should succeed
        v_small = 0.1 * randn(n)
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        
        success = cholesky_downdate!(L_copy, v_small)
        @test success == true
        
        # Result should be positive definite
        H_new = L_copy * L_copy'
        eigvals_new = eigvals(Symmetric(H_new))
        @test all(eigvals_new .> 0)
    end
    
    @testset "Downdate detects indefiniteness" begin
        Random.seed!(34567)
        n = 3
        A = randn(n, n)
        H = A' * A + 0.5 * I  # Weakly positive definite
        
        # Large downdate should fail (result would be indefinite)
        v_large = 2.0 * randn(n)
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        
        success = cholesky_downdate!(L_copy, v_large)
        @test success == false
    end
    
    @testset "Non-mutating downdate copy" begin
        Random.seed!(45678)
        n = 4
        A = randn(n, n)
        H = A' * A + 3.0 * I
        v = 0.2 * randn(n)
        
        L = cholesky(Symmetric(H)).L
        L_original = copy(L)
        
        L_new, success = cholesky_downdate_copy(L, v)
        
        # Original should be unchanged
        @test L ≈ L_original
        
        # New should be the downdate
        @test success == true
        @test L_new * L_new' ≈ H - v * v' atol=1e-10
    end
    
    @testset "Multiple sequential downdates" begin
        Random.seed!(56789)
        n = 5
        A = randn(n, n)
        H = A' * A + 10.0 * I
        
        # Multiple small downdates
        vs = [0.1 * randn(n) for _ in 1:3]
        
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        
        H_expected = copy(H)
        all_success = true
        for v in vs
            success = cholesky_downdate!(L_copy, v)
            all_success &= success
            H_expected -= v * v'
        end
        
        @test all_success == true
        @test L_copy * L_copy' ≈ H_expected atol=1e-9
    end
end

# =============================================================================
# 2. LOO Perturbation Computation Tests
# =============================================================================

@testset "LOO Perturbation Methods" begin
    
    @testset "Direct solve correctness" begin
        Random.seed!(11111)
        p = 4
        
        # Create H_lambda (penalized Hessian)
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        
        # Create neighbourhood Hessian H_k (rank-1 for simplicity)
        g_k = randn(p)
        H_k = 0.1 * g_k * g_k'  # Small contribution
        
        # Compute perturbation directly
        result = ncv_loo_perturbation_direct(H_lambda, H_k, g_k)
        
        # Verify: delta = (H_lambda - H_k)^{-1} g_k
        H_loo = H_lambda - H_k
        delta_expected = H_loo \ g_k
        
        @test result.delta ≈ delta_expected atol=1e-10
        @test result.indefinite == false
    end
    
    @testset "Cholesky method matches direct" begin
        Random.seed!(22222)
        p = 5
        
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        H_chol = cholesky(Symmetric(H_lambda))
        
        g_k = randn(p)
        H_k = 0.05 * g_k * g_k'  # Small rank-1 contribution
        
        result_chol = ncv_loo_perturbation_cholesky(H_chol, H_k, g_k)
        result_direct = ncv_loo_perturbation_direct(H_lambda, H_k, g_k)
        
        @test result_chol.delta ≈ result_direct.delta atol=1e-8
    end
    
    @testset "Woodbury fallback works" begin
        Random.seed!(33333)
        p = 4
        
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        H_chol = cholesky(Symmetric(H_lambda))
        
        g_k = randn(p)
        H_k = 0.1 * g_k * g_k'
        
        # Force Woodbury path
        result = ncv_loo_perturbation_woodbury(H_chol, H_k, g_k)
        
        # Verify result
        H_loo = H_lambda - H_k
        delta_expected = H_loo \ g_k
        
        @test result.delta ≈ delta_expected atol=1e-8
        @test result.indefinite == true  # Woodbury always marks as indefinite
    end
    
    @testset "Zero H_k gives H_lambda^{-1} g_k" begin
        Random.seed!(44444)
        p = 4
        
        A = randn(p, p)
        H_lambda = A' * A + 3.0 * I
        H_chol = cholesky(Symmetric(H_lambda))
        
        g_k = randn(p)
        H_k = zeros(p, p)
        
        result = ncv_loo_perturbation_cholesky(H_chol, H_k, g_k)
        expected = H_chol \ g_k
        
        @test result.delta ≈ expected atol=1e-10
        @test result.indefinite == false
    end
    
    @testset "Full-rank H_k handling" begin
        Random.seed!(55555)
        p = 3
        
        A = randn(p, p)
        H_lambda = A' * A + 10.0 * I  # Strong penalty
        H_chol = cholesky(Symmetric(H_lambda))
        
        # Full-rank H_k (but small)
        B = randn(p, p)
        H_k = 0.1 * B' * B
        
        g_k = randn(p)
        
        result = ncv_loo_perturbation_cholesky(H_chol, H_k, g_k)
        
        # Verify against direct solve
        H_loo = H_lambda - H_k
        delta_expected = H_loo \ g_k
        
        @test result.delta ≈ delta_expected atol=1e-6
    end
end

# =============================================================================
# 3. NCVState and Perturbation Computation Tests
# =============================================================================

@testset "NCVState Construction and Perturbations" begin
    
    @testset "NCVState initialization" begin
        Random.seed!(66666)
        p = 4
        n = 10
        
        # Create test data
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        subject_grads = randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        
        @test size(state.H_lambda) == (p, p)
        @test size(state.subject_grads) == (p, n)
        @test size(state.deltas) == (p, n)
        @test length(state.indefinite_flags) == n
        @test !isnothing(state.H_chol)  # Should have Cholesky
        @test all(state.deltas .== 0)  # Initially zero
    end
    
    @testset "NCVState with subject Hessians" begin
        Random.seed!(77777)
        p = 3
        n = 5
        
        A = randn(p, p)
        H_lambda = A' * A + 3.0 * I
        subject_grads = randn(p, n)
        subject_hessians = zeros(p, p, n)
        for k in 1:n
            B = randn(p, p)
            subject_hessians[:, :, k] = 0.1 * B' * B
        end
        
        state = NCVState(H_lambda, subject_grads; subject_hessians=subject_hessians)
        
        @test !isnothing(state.subject_hessians)
        @test size(state.subject_hessians) == (p, p, n)
    end
    
    @testset "compute_ncv_perturbations! correctness" begin
        Random.seed!(88888)
        p = 3
        n = 4
        
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        subject_grads = 0.5 * randn(p, n)
        
        # With outer product approximation for H_k
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        # Verify each perturbation
        for k in 1:n
            g_k = subject_grads[:, k]
            H_k = g_k * g_k'  # Outer product approximation
            H_loo = H_lambda - H_k
            delta_expected = H_loo \ g_k
            
            @test state.deltas[:, k] ≈ delta_expected atol=1e-6
        end
    end
    
    @testset "compute_ncv_perturbations! with provided Hessians" begin
        Random.seed!(99999)
        p = 3
        n = 3
        
        A = randn(p, p)
        H_lambda = A' * A + 10.0 * I
        subject_grads = randn(p, n)
        subject_hessians = zeros(p, p, n)
        for k in 1:n
            B = randn(p, p)
            subject_hessians[:, :, k] = 0.05 * B' * B
        end
        
        state = NCVState(H_lambda, subject_grads; subject_hessians=subject_hessians)
        compute_ncv_perturbations!(state)
        
        # Verify
        for k in 1:n
            g_k = subject_grads[:, k]
            H_k = subject_hessians[:, :, k]
            H_loo = H_lambda - H_k
            delta_expected = H_loo \ g_k
            
            @test state.deltas[:, k] ≈ delta_expected atol=1e-6
        end
    end
end

# =============================================================================
# 4. NCV Criterion Computation Tests
# =============================================================================

@testset "NCV Criterion" begin
    
    @testset "ncv_criterion basic computation" begin
        Random.seed!(10101)
        p = 3
        n = 5
        
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        subject_grads = 0.3 * randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        params = randn(p)
        
        # Simple loss function: squared error from zero
        loss_fn(pars, data, k) = sum(pars.^2)
        
        V = ncv_criterion(state, params, loss_fn, nothing)
        
        # Manual computation
        V_expected = 0.0
        for k in 1:n
            params_loo = params .+ state.deltas[:, k]
            V_expected += loss_fn(params_loo, nothing, k)
        end
        V_expected /= n
        
        @test V ≈ V_expected
    end
    
    @testset "ncv_criterion_quadratic approximation" begin
        Random.seed!(20202)
        p = 3
        n = 4
        
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        subject_grads = 0.2 * randn(p, n)
        
        # Create subject Hessians for quadratic approximation
        subject_hessians = zeros(p, p, n)
        for k in 1:n
            B = randn(p, p)
            subject_hessians[:, :, k] = 0.1 * B' * B
        end
        
        state = NCVState(H_lambda, subject_grads; subject_hessians=subject_hessians)
        compute_ncv_perturbations!(state)
        
        params = randn(p)
        
        # Quadratic loss
        loss_fn(pars, data, k) = 0.5 * sum(pars.^2)
        
        V_q = ncv_criterion_quadratic(state, params, loss_fn, nothing)
        
        # For small perturbations, quadratic should be close to exact
        V_exact = ncv_criterion(state, params, loss_fn, nothing)
        
        # They won't be exactly equal, but should be in same ballpark
        @test abs(V_q - V_exact) / max(abs(V_exact), 1.0) < 0.5
    end
    
    @testset "ncv_get_loo_estimates" begin
        Random.seed!(30303)
        p = 3
        n = 4
        
        A = randn(p, p)
        H_lambda = A' * A + 3.0 * I
        subject_grads = randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        params = randn(p)
        loo_estimates = ncv_get_loo_estimates(state, params)
        
        @test size(loo_estimates) == (p, n)
        for k in 1:n
            @test loo_estimates[:, k] ≈ params .+ state.deltas[:, k]
        end
    end
    
    @testset "ncv_get_perturbations" begin
        Random.seed!(40404)
        p = 3
        n = 5
        
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        subject_grads = randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        perturbations = ncv_get_perturbations(state)
        
        @test perturbations === state.deltas
    end
end

# =============================================================================
# 5. NCV Variance Estimation Tests
# =============================================================================

@testset "NCV Variance Estimation" begin
    
    @testset "ncv_vcov computation" begin
        Random.seed!(50505)
        p = 3
        n = 10
        
        A = randn(p, p)
        H_lambda = A' * A + 3.0 * I
        subject_grads = 0.5 * randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        vcov_result = ncv_vcov(state)
        
        # Check structure
        @test haskey(vcov_result, :ij_vcov)
        @test haskey(vcov_result, :jk_vcov)
        @test size(vcov_result.ij_vcov) == (p, p)
        @test size(vcov_result.jk_vcov) == (p, p)
        
        # Verify formulas
        delta_outer = state.deltas * state.deltas'
        @test vcov_result.ij_vcov ≈ Symmetric(delta_outer / n)
        @test vcov_result.jk_vcov ≈ Symmetric(((n - 1) / n) * delta_outer)
    end
    
    @testset "ncv_vcov positive semi-definite" begin
        Random.seed!(60606)
        p = 4
        n = 20
        
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        subject_grads = randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        vcov_result = ncv_vcov(state)
        
        # Both should be PSD
        eigvals_ij = eigvals(vcov_result.ij_vcov)
        eigvals_jk = eigvals(vcov_result.jk_vcov)
        
        @test all(eigvals_ij .>= -1e-10)
        @test all(eigvals_jk .>= -1e-10)
    end
end

# =============================================================================
# 6. Degeneracy Detection Tests
# =============================================================================

@testset "Degeneracy Detection" begin
    
    @testset "ncv_degeneracy_test non-degenerate case" begin
        Random.seed!(70707)
        p = 4
        n = 10
        
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        subject_grads = randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        
        params = randn(p)
        
        # Non-trivial penalty derivative
        B = randn(p, p)
        penalty_deriv = B' * B
        
        is_degenerate = ncv_degeneracy_test(state, params, penalty_deriv)
        
        # With reasonable penalty, should not be degenerate
        @test is_degenerate == false
    end
    
    @testset "ncv_degeneracy_test degenerate case" begin
        Random.seed!(80808)
        p = 4
        n = 10
        
        A = randn(p, p)
        H_lambda = A' * A + 100.0 * I  # Very strong penalty
        subject_grads = randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        
        params = 0.001 * randn(p)  # Very small params
        
        # Penalty derivative that effectively zeros out
        penalty_deriv = zeros(p, p)
        penalty_deriv[1, 1] = 1e-10  # Trivially small
        
        is_degenerate = ncv_degeneracy_test(state, params, penalty_deriv; tol=1e-4)
        
        # Should detect degeneracy
        @test is_degenerate == true
    end
end

# =============================================================================
# 7. Consistency with IJ/JK Variance Tests
# =============================================================================

@testset "NCV Consistency with IJ/JK" begin
    
    @testset "NCV perturbations match loo_perturbations_direct" begin
        Random.seed!(90909)
        p = 4
        n = 8
        
        # Create test case where H_k = g_k * g_k' (outer product)
        A = randn(p, p)
        fishinf = A' * A + 3.0 * I  # This is H_lambda
        subject_grads = 0.3 * randn(p, n)
        
        # Compute via standard IJ/JK method
        vcov_ij = inv(fishinf)
        loo_deltas_ij = loo_perturbations_direct(vcov_ij, subject_grads)
        
        # Compute via NCV (without subject Hessians, uses outer product)
        state = NCVState(fishinf, subject_grads)
        compute_ncv_perturbations!(state)
        
        # For the case where H_k = g_k * g_k', both methods should give
        # similar results, though not identical due to different formulations
        # IJ uses: Δᵢ = H⁻¹gᵢ
        # NCV uses: Δᵢ = (H - gᵢgᵢ')⁻¹gᵢ
        
        # Check that they're in the same direction at least
        for k in 1:n
            corr = dot(loo_deltas_ij[:, k], state.deltas[:, k]) / 
                   (norm(loo_deltas_ij[:, k]) * norm(state.deltas[:, k]))
            @test corr > 0.9  # High correlation
        end
    end
    
    @testset "NCV with zero H_k matches IJ exactly" begin
        Random.seed!(10110)
        p = 3
        n = 5
        
        A = randn(p, p)
        fishinf = A' * A + 2.0 * I
        subject_grads = randn(p, n)
        
        # Create NCV state with zero subject Hessians
        subject_hessians = zeros(p, p, n)
        
        state = NCVState(fishinf, subject_grads; subject_hessians=subject_hessians)
        compute_ncv_perturbations!(state)
        
        # Standard IJ perturbations
        vcov_ij = inv(fishinf)
        loo_deltas_ij = loo_perturbations_direct(vcov_ij, subject_grads)
        
        # With H_k = 0, NCV should give Δᵢ = H⁻¹gᵢ, same as IJ
        @test state.deltas ≈ loo_deltas_ij atol=1e-8
    end
end

# =============================================================================
# 8. Edge Cases and Robustness Tests
# =============================================================================

@testset "NCV Edge Cases" begin
    
    @testset "Single subject" begin
        Random.seed!(11011)
        p = 3
        n = 1
        
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        subject_grads = randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        @test size(state.deltas) == (p, 1)
        @test !any(isnan.(state.deltas))
    end
    
    @testset "High dimensional case" begin
        Random.seed!(12012)
        p = 20
        n = 50
        
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        subject_grads = 0.1 * randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        @test !any(isnan.(state.deltas))
        @test !any(isinf.(state.deltas))
    end
    
    @testset "Sparse gradients" begin
        Random.seed!(13013)
        p = 5
        n = 10
        
        A = randn(p, p)
        H_lambda = A' * A + 3.0 * I
        
        # Sparse gradients (mostly zeros)
        subject_grads = zeros(p, n)
        for k in 1:n
            idx = rand(1:p)
            subject_grads[idx, k] = randn()
        end
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        @test !any(isnan.(state.deltas))
    end
    
    @testset "Near-singular H_lambda with regularization" begin
        Random.seed!(14014)
        p = 4
        n = 5
        
        # Create near-singular H_lambda
        A = randn(p, p-1)  # Rank deficient
        H_lambda = A * A' + 1e-8 * I  # Barely regularized
        
        subject_grads = randn(p, n)
        
        # May not get Cholesky
        state = NCVState(H_lambda, subject_grads)
        
        # Should still compute perturbations (via direct method)
        compute_ncv_perturbations!(state)
        
        # May have some NaN if truly singular, but should handle gracefully
        # At minimum, shouldn't error
        @test true
    end
end
