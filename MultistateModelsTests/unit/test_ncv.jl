# =============================================================================
# NCV (Neighbourhood Cross-Validation) Tests
# =============================================================================
#
# Tests for the NCV framework implementing Wood (2024) "Neighbourhood Cross 
# Validation" (arXiv:2404.16490).
#
# All tests verify mathematical correctness through analytical formulas.

using LinearAlgebra
using Random

# Import internal functions for testing
import MultistateModels: cholesky_downdate!, cholesky_downdate_copy,
                          ncv_loo_perturbation_direct, ncv_loo_perturbation_cholesky,
                          ncv_loo_perturbation_woodbury,
                          NCVState, compute_ncv_perturbations!, ncv_criterion,
                          loo_perturbations_direct, ncv_get_loo_estimates, ncv_vcov

# =============================================================================
# 1. Cholesky Downdate Algorithm
# =============================================================================

@testset "Cholesky Downdate" begin
    
    @testset "Rank-1 downdate correctness" begin
        # Verify L*L' ≈ H - v*v' after downdate
        Random.seed!(12345)
        n = 5
        A = randn(n, n)
        H = A' * A + 2.0 * I
        v = 0.3 * randn(n)
        
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        success = cholesky_downdate!(L_copy, v)
        
        @test success == true
        @test L_copy * L_copy' ≈ H - v * v' atol=1e-10
    end
    
    @testset "Downdate preserves positive definiteness" begin
        Random.seed!(23456)
        n = 4
        A = randn(n, n)
        H = A' * A + 5.0 * I
        v_small = 0.1 * randn(n)
        
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        success = cholesky_downdate!(L_copy, v_small)
        
        @test success == true
        H_new = L_copy * L_copy'
        @test all(eigvals(Symmetric(H_new)) .> 0)
    end
    
    @testset "Downdate detects indefiniteness" begin
        Random.seed!(34567)
        n = 3
        A = randn(n, n)
        H = A' * A + 0.5 * I  # Weakly positive definite
        v_large = 2.0 * randn(n)  # Large enough to make indefinite
        
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        success = cholesky_downdate!(L_copy, v_large)
        
        @test success == false
    end
    
    @testset "Non-mutating copy version" begin
        Random.seed!(45678)
        n = 4
        A = randn(n, n)
        H = A' * A + 3.0 * I
        v = 0.2 * randn(n)
        
        L = cholesky(Symmetric(H)).L
        L_original = copy(L)
        
        L_new, success = cholesky_downdate_copy(L, v)
        
        @test L ≈ L_original  # Original unchanged
        @test success == true
        @test L_new * L_new' ≈ H - v * v' atol=1e-10
    end
    
    @testset "Sequential downdates accumulate correctly" begin
        Random.seed!(56789)
        n = 5
        A = randn(n, n)
        H = A' * A + 10.0 * I
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
# 2. LOO Perturbation Methods
# =============================================================================

@testset "LOO Perturbation Methods" begin
    
    @testset "Direct solve: delta = (H_lambda - H_k)^{-1} g_k" begin
        Random.seed!(11111)
        p = 4
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        g_k = randn(p)
        H_k = 0.1 * g_k * g_k'
        
        result = ncv_loo_perturbation_direct(H_lambda, H_k, g_k)
        delta_expected = (H_lambda - H_k) \ g_k
        
        @test result.delta ≈ delta_expected atol=1e-10
        @test result.indefinite == false
    end
    
    @testset "Cholesky method matches direct solve" begin
        Random.seed!(22222)
        p = 5
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        H_chol = cholesky(Symmetric(H_lambda))
        g_k = randn(p)
        H_k = 0.05 * g_k * g_k'
        
        result_chol = ncv_loo_perturbation_cholesky(H_chol, H_k, g_k)
        result_direct = ncv_loo_perturbation_direct(H_lambda, H_k, g_k)
        
        @test result_chol.delta ≈ result_direct.delta atol=1e-8
    end
    
    @testset "Woodbury fallback matches direct solve" begin
        Random.seed!(33333)
        p = 4
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        H_chol = cholesky(Symmetric(H_lambda))
        g_k = randn(p)
        H_k = 0.1 * g_k * g_k'
        
        result = ncv_loo_perturbation_woodbury(H_chol, H_k, g_k)
        delta_expected = (H_lambda - H_k) \ g_k
        
        @test result.delta ≈ delta_expected atol=1e-8
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
    end
    
    @testset "Full-rank H_k handling" begin
        Random.seed!(55555)
        p = 3
        A = randn(p, p)
        H_lambda = A' * A + 10.0 * I
        H_chol = cholesky(Symmetric(H_lambda))
        B = randn(p, p)
        H_k = 0.1 * B' * B  # Full-rank
        g_k = randn(p)
        
        result = ncv_loo_perturbation_cholesky(H_chol, H_k, g_k)
        delta_expected = (H_lambda - H_k) \ g_k
        
        @test result.delta ≈ delta_expected atol=1e-6
    end
end

# =============================================================================
# 3. NCV Perturbation Computation
# =============================================================================

@testset "NCV Perturbation Computation" begin
    
    @testset "Outer product approximation: delta_k = (H - g_k g_k')^{-1} g_k" begin
        Random.seed!(88888)
        p = 3
        n = 4
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        subject_grads = 0.5 * randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        for k in 1:n
            g_k = subject_grads[:, k]
            H_k = g_k * g_k'
            delta_expected = (H_lambda - H_k) \ g_k
            @test state.deltas[:, k] ≈ delta_expected atol=1e-6
        end
    end
    
    @testset "Provided Hessians: delta_k = (H - H_k)^{-1} g_k" begin
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
        
        for k in 1:n
            g_k = subject_grads[:, k]
            H_k = subject_hessians[:, :, k]
            delta_expected = (H_lambda - H_k) \ g_k
            @test state.deltas[:, k] ≈ delta_expected atol=1e-6
        end
    end
end

# =============================================================================
# 4. NCV Criterion
# =============================================================================

@testset "NCV Criterion" begin
    
    @testset "Criterion = mean of LOO losses" begin
        Random.seed!(10101)
        p = 3
        n = 5
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        subject_grads = 0.3 * randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        params = randn(p)
        loss_fn(pars, data, k) = sum(pars.^2)
        
        V = ncv_criterion(state, params, loss_fn, nothing)
        
        V_expected = 0.0
        for k in 1:n
            params_loo = params .+ state.deltas[:, k]
            V_expected += loss_fn(params_loo, nothing, k)
        end
        V_expected /= n
        
        @test V ≈ V_expected
    end
    
    @testset "LOO estimates = params + deltas" begin
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
        
        for k in 1:n
            @test loo_estimates[:, k] ≈ params .+ state.deltas[:, k]
        end
    end
end

# =============================================================================
# 5. Variance Estimation
# =============================================================================

@testset "NCV Variance Estimation" begin
    
    @testset "Covariance matrices are positive semi-definite" begin
        Random.seed!(60606)
        p = 4
        n = 20
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        subject_grads = randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        vcov_result = ncv_vcov(state)
        
        @test all(eigvals(vcov_result.ij_vcov) .>= -1e-10)
        @test all(eigvals(vcov_result.jk_vcov) .>= -1e-10)
    end
    
    @testset "Variance formulas" begin
        Random.seed!(50505)
        p = 3
        n = 10
        A = randn(p, p)
        H_lambda = A' * A + 3.0 * I
        subject_grads = 0.5 * randn(p, n)
        
        state = NCVState(H_lambda, subject_grads)
        compute_ncv_perturbations!(state)
        
        vcov_result = ncv_vcov(state)
        
        delta_outer = state.deltas * state.deltas'
        @test vcov_result.ij_vcov ≈ Symmetric(delta_outer / n)
        @test vcov_result.jk_vcov ≈ Symmetric(((n - 1) / n) * delta_outer)
    end
end

# =============================================================================
# 6. Consistency with IJ/JK Methods
# =============================================================================

@testset "NCV Consistency with IJ/JK" begin
    
    @testset "Zero H_k matches IJ exactly: delta_k = H^{-1} g_k" begin
        Random.seed!(10110)
        p = 3
        n = 5
        A = randn(p, p)
        fishinf = A' * A + 2.0 * I
        subject_grads = randn(p, n)
        subject_hessians = zeros(p, p, n)
        
        state = NCVState(fishinf, subject_grads; subject_hessians=subject_hessians)
        compute_ncv_perturbations!(state)
        
        vcov_ij = inv(fishinf)
        loo_deltas_ij = loo_perturbations_direct(vcov_ij, subject_grads)
        
        @test state.deltas ≈ loo_deltas_ij atol=1e-8
    end
    
    @testset "NCV vs IJ perturbations highly correlated" begin
        Random.seed!(90909)
        p = 4
        n = 8
        A = randn(p, p)
        fishinf = A' * A + 3.0 * I
        subject_grads = 0.3 * randn(p, n)
        
        vcov_ij = inv(fishinf)
        loo_deltas_ij = loo_perturbations_direct(vcov_ij, subject_grads)
        
        state = NCVState(fishinf, subject_grads)
        compute_ncv_perturbations!(state)
        
        for k in 1:n
            corr = dot(loo_deltas_ij[:, k], state.deltas[:, k]) / 
                   (norm(loo_deltas_ij[:, k]) * norm(state.deltas[:, k]))
            @test corr > 0.9
        end
    end
end
