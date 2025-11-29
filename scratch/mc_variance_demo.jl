#=
Minimal Example: Monte Carlo Correlation in Variance Estimation

This script demonstrates that when using the SAME Monte Carlo samples to estimate
both the gradient and Hessian (as in MCEM), there's a correlation that can bias
variance estimates when the number of MC samples is small.

Key insight: In MCEM, we estimate:
  - ĝ = Σⱼ wⱼ gⱼ  (gradient)
  - Ĥ = Σⱼ wⱼ Hⱼ  (Hessian, simplified for illustration)

The same weights wⱼ appear in both. This creates Cov(ĝ, Ĥ) ≠ 0.

For the sandwich estimator V̂ = Ĥ⁻¹ ĝĝᵀ Ĥ⁻¹, this correlation can cause bias.
=#

using Random
using Statistics
using LinearAlgebra
using Printf

"""
Simple 1D example: estimating the mean of a normal distribution.

True model: X ~ N(μ, σ²)
- True gradient (score) at MLE: g = 0
- True Fisher information: I = n/σ²
- True variance of μ̂: V = σ²/n

In an "MCEM-like" setting, we observe Y and sample latent Z|Y.
For simplicity, let's pretend each observation has "latent paths" 
whose gradients vary, and we importance-weight them.
"""

function simulate_mcem_variance_estimation(;
    n_subjects = 100,      # Number of subjects
    n_paths = 50,          # Number of MC paths per subject (like numpaths_mcem)
    true_μ = 0.0,          # True parameter
    σ = 1.0,               # Known variance
    n_reps = 1000          # Number of simulation replications
)
    
    # Storage for results
    V_same_samples = zeros(n_reps)      # Variance estimate using same samples for g and H
    V_independent = zeros(n_reps)        # Variance estimate using independent samples
    V_true = σ^2 / n_subjects            # True variance of μ̂
    
    for rep in 1:n_reps
        # Simulate "observed data" for each subject (just a single value here)
        Y = randn(n_subjects) .* σ .+ true_μ
        
        # The MLE is just the sample mean
        μ_hat = mean(Y)
        
        #= 
        MCEM-like setup:
        For each subject i, we have n_paths "latent paths" with gradients gᵢⱼ.
        In reality, these would be sampled from p(Z|Y,θ) with importance weights.
        
        For this demo, we simulate:
        - gᵢⱼ = (Yᵢ - μ̂) + εᵢⱼ  where εᵢⱼ ~ N(0, τ²) represents MC noise
        - Hᵢⱼ = -1/σ² + δᵢⱼ     where δᵢⱼ represents MC noise in Hessian
        
        The importance weights are uniform for simplicity: wⱼ = 1/n_paths
        =#
        
        τ = 0.5  # MC noise in gradients
        
        # === METHOD 1: Same samples for gradient and Hessian (what MCEM does) ===
        
        subject_grads_same = zeros(n_subjects)
        subject_hess_same = zeros(n_subjects)
        
        for i in 1:n_subjects
            # Sample path-level quantities (SAME random seed for g and H)
            ε_paths = randn(n_paths) .* τ
            
            # Path-level gradients: g_ij = (Y_i - μ̂)/σ² + noise
            g_paths = (Y[i] - μ_hat) / σ^2 .+ ε_paths
            
            # Path-level Hessians: H_ij = -1/σ² + correlated noise
            # The correlation comes from using the SAME ε_paths
            H_paths = fill(-1/σ^2, n_paths) .- 0.3 .* ε_paths  # Correlated with g!
            
            # Importance-weighted estimates (uniform weights)
            subject_grads_same[i] = mean(g_paths)
            subject_hess_same[i] = mean(H_paths)
        end
        
        # Fisher information and variance
        H_total_same = sum(subject_hess_same)
        K_same = sum(subject_grads_same.^2)  # Σᵢ gᵢ²
        
        # Sandwich variance (1D version): V = H⁻¹ K H⁻¹ = K / H²
        if H_total_same < 0  # Should be negative (Fisher info is -H)
            I_same = -H_total_same
            V_same_samples[rep] = K_same / I_same^2
        else
            V_same_samples[rep] = NaN
        end
        
        # === METHOD 2: Independent samples for gradient and Hessian ===
        
        subject_grads_ind = zeros(n_subjects)
        subject_hess_ind = zeros(n_subjects)
        
        for i in 1:n_subjects
            # Sample DIFFERENT path-level quantities for g and H
            ε_paths_g = randn(n_paths) .* τ  # Independent noise for gradient
            ε_paths_H = randn(n_paths) .* τ  # Independent noise for Hessian
            
            # Path-level gradients
            g_paths = (Y[i] - μ_hat) / σ^2 .+ ε_paths_g
            
            # Path-level Hessians with INDEPENDENT noise
            H_paths = fill(-1/σ^2, n_paths) .- 0.3 .* ε_paths_H
            
            # Importance-weighted estimates
            subject_grads_ind[i] = mean(g_paths)
            subject_hess_ind[i] = mean(H_paths)
        end
        
        # Fisher information and variance
        H_total_ind = sum(subject_hess_ind)
        K_ind = sum(subject_grads_ind.^2)
        
        if H_total_ind < 0
            I_ind = -H_total_ind
            V_independent[rep] = K_ind / I_ind^2
        else
            V_independent[rep] = NaN
        end
    end
    
    # Filter out NaN values
    V_same_valid = filter(!isnan, V_same_samples)
    V_ind_valid = filter(!isnan, V_independent)
    
    return (
        V_true = V_true,
        V_same_mean = mean(V_same_valid),
        V_same_std = std(V_same_valid),
        V_ind_mean = mean(V_ind_valid),
        V_ind_std = std(V_ind_valid),
        bias_same = mean(V_same_valid) - V_true,
        bias_ind = mean(V_ind_valid) - V_true,
        relative_bias_same = (mean(V_same_valid) - V_true) / V_true * 100,
        relative_bias_ind = (mean(V_ind_valid) - V_true) / V_true * 100
    )
end

# ============================================================================
# Run the demonstration
# ============================================================================

println("="^70)
println("Monte Carlo Correlation Demo: Same vs Independent Samples")
println("="^70)
println()

# Test with different numbers of MC paths
for n_paths in [10, 50, 100, 500, 1000]
    println("Number of MC paths per subject: $n_paths")
    println("-"^50)
    
    result = simulate_mcem_variance_estimation(n_paths=n_paths, n_reps=2000)
    
    @printf("  True variance:              %.6f\n", result.V_true)
    @printf("  Same samples (MCEM-like):   %.6f (bias: %+.2f%%)\n", 
            result.V_same_mean, result.relative_bias_same)
    @printf("  Independent samples:        %.6f (bias: %+.2f%%)\n", 
            result.V_ind_mean, result.relative_bias_ind)
    println()
end

println("="^70)
println("Key Observation:")
println("- With few MC paths, 'same samples' shows bias due to g-H correlation")
println("- As n_paths increases, bias decreases (O(1/n_paths))")
println("- With 1000+ paths (typical MCEM), the effect is negligible")
println("="^70)

# ============================================================================
# More detailed analysis: decompose the bias
# ============================================================================

println()
println("="^70)
println("Detailed Analysis: Sources of Bias")
println("="^70)
println()

function analyze_bias_components(; n_subjects=100, n_paths=50, n_reps=5000)
    τ = 0.5
    σ = 1.0
    
    # Track individual components
    g_estimates = zeros(n_reps)
    H_estimates = zeros(n_reps)
    g_H_products = zeros(n_reps)
    
    for rep in 1:n_reps
        Y = randn(n_subjects) .* σ
        μ_hat = mean(Y)
        
        total_g = 0.0
        total_H = 0.0
        
        for i in 1:n_subjects
            ε_paths = randn(n_paths) .* τ
            g_paths = (Y[i] - μ_hat) / σ^2 .+ ε_paths
            H_paths = fill(-1/σ^2, n_paths) .- 0.3 .* ε_paths
            
            total_g += mean(g_paths)
            total_H += mean(H_paths)
        end
        
        g_estimates[rep] = total_g
        H_estimates[rep] = total_H
        g_H_products[rep] = total_g * total_H
    end
    
    println("With n_paths = $n_paths:")
    @printf("  E[Σgᵢ]:           %+.4f (should be ≈ 0 at MLE)\n", mean(g_estimates))
    @printf("  E[ΣHᵢ]:           %+.4f (should be ≈ -n/σ² = -100)\n", mean(H_estimates))
    @printf("  Cov(Σgᵢ, ΣHᵢ):    %+.4f\n", cov(g_estimates, H_estimates))
    @printf("  Cor(Σgᵢ, ΣHᵢ):    %+.4f\n", cor(g_estimates, H_estimates))
    println()
    println("  The non-zero covariance is the source of bias!")
    println("  It arises because the same MC samples (ε) appear in both g and H.")
end

analyze_bias_components(n_paths=50)
println()
analyze_bias_components(n_paths=500)

# ============================================================================
# A clearer example: The IJ variance estimator
# ============================================================================

println()
println("="^70)
println("IJ Variance Estimator: Same vs Independent Samples")
println("="^70)
println()
println("The IJ estimator is V̂ = H⁻¹ K H⁻¹ where K = Σᵢ gᵢ²")
println("With MC estimation, both ĝᵢ and Ĥ have sampling error.")
println()

"""
This example is more realistic: we compute the actual sandwich variance.
The issue is that when ĝᵢ and Ĥᵢ are estimated from the same MC paths,
their errors are correlated.

For the sandwich: V̂ = Ĥ⁻¹ K̂ Ĥ⁻¹
where K̂ = Σᵢ ĝᵢ²

If ĝᵢ = gᵢ + εᵢ and Ĥ = H + E, with Cov(εᵢ, E) ≠ 0,
then E[V̂] ≠ V due to products of correlated terms.
"""
function ij_variance_demo(; n_subjects=50, n_paths=100, n_reps=3000)
    σ = 1.0
    V_true = σ^2 / n_subjects  # True variance of sample mean
    
    V_same = zeros(n_reps)
    V_ind = zeros(n_reps)
    
    # MC noise parameters
    τ_g = 0.3   # noise in gradient estimates
    τ_H = 0.05  # noise in Hessian estimates
    ρ = 0.8     # correlation between g and H noise (same samples!)
    
    for rep in 1:n_reps
        Y = randn(n_subjects) .* σ
        μ_hat = mean(Y)
        
        # True subject scores at the MLE
        true_g = (Y .- μ_hat) ./ σ^2
        
        # === Same samples (MCEM) ===
        # Generate correlated noise for g and H
        z1 = randn(n_subjects, n_paths)
        z2 = randn(n_subjects, n_paths)
        
        # MC estimates with correlation
        ε_g = τ_g .* z1 ./ sqrt(n_paths)  # noise in ĝ, scaled by 1/√B
        ε_H = τ_H .* (ρ .* z1 .+ sqrt(1-ρ^2) .* z2) ./ sqrt(n_paths)  # correlated noise in Ĥ
        
        g_hat_same = true_g .+ vec(mean(ε_g, dims=2))
        H_i_hat_same = fill(-1/σ^2, n_subjects) .+ vec(mean(ε_H, dims=2))
        
        H_hat_same = sum(H_i_hat_same)
        K_hat_same = sum(g_hat_same.^2)
        
        if H_hat_same < 0
            I_hat = -H_hat_same
            V_same[rep] = K_hat_same / I_hat^2
        else
            V_same[rep] = NaN
        end
        
        # === Independent samples ===
        z3 = randn(n_subjects, n_paths)  # Independent noise for H
        
        g_hat_ind = true_g .+ vec(mean(ε_g, dims=2))  # Same gradient estimate
        H_i_hat_ind = fill(-1/σ^2, n_subjects) .+ τ_H .* vec(mean(z3, dims=2)) ./ sqrt(n_paths)
        
        H_hat_ind = sum(H_i_hat_ind)
        K_hat_ind = sum(g_hat_ind.^2)
        
        if H_hat_ind < 0
            I_hat = -H_hat_ind
            V_ind[rep] = K_hat_ind / I_hat^2
        else
            V_ind[rep] = NaN
        end
    end
    
    V_same_valid = filter(!isnan, V_same)
    V_ind_valid = filter(!isnan, V_ind)
    
    println("n_subjects = $n_subjects, n_paths = $n_paths, ρ (g-H correlation) = $ρ")
    @printf("  True variance:         %.6f\n", V_true)
    @printf("  Same samples mean:     %.6f (bias: %+.2f%%)\n", 
            mean(V_same_valid), (mean(V_same_valid) - V_true) / V_true * 100)
    @printf("  Independent mean:      %.6f (bias: %+.2f%%)\n", 
            mean(V_ind_valid), (mean(V_ind_valid) - V_true) / V_true * 100)
    @printf("  Same samples std:      %.6f\n", std(V_same_valid))
    @printf("  Independent std:       %.6f\n", std(V_ind_valid))
    println()
    
    return (V_true=V_true, V_same=mean(V_same_valid), V_ind=mean(V_ind_valid))
end

# Test with different MC sample sizes
for n_paths in [20, 100, 500, 2000]
    ij_variance_demo(n_paths=n_paths)
end

println("="^70)
println("Conclusion:")
println("- With small n_paths and high ρ, the 'same samples' estimator has")
println("  different behavior than 'independent samples'")
println("- The correlation ρ determines how much the errors are coupled")
println("- In practice (n_paths ≥ 1000), both converge to the true variance")
println("="^70)

# ============================================================================
# The REAL issue: Within-subject MC variance inflates K
# ============================================================================

println()
println("="^70)
println("The Real Issue: MC Noise Inflates the 'K' Matrix")
println("="^70)
println()
println("""
The more important effect is that MC noise in ĝᵢ directly inflates K̂:

  True:  K = Σᵢ gᵢ²
  Est:   K̂ = Σᵢ ĝᵢ² = Σᵢ (gᵢ + εᵢ)² = Σᵢ gᵢ² + Σᵢ εᵢ² + 2Σᵢ gᵢεᵢ

At the MLE, gᵢ ≈ 0 on average, but εᵢ² > 0 always!

So: E[K̂] = K + Σᵢ Var(εᵢ) > K

This is the within-subject MC variance inflating the K matrix.
The correlation with H is secondary.
""")

function mc_variance_inflation_demo(; n_subjects=50, n_paths=100, n_reps=3000)
    σ = 1.0
    τ = 0.5  # Within-subject variance of path gradients
    
    # True quantities
    V_true = σ^2 / n_subjects
    
    K_true_vals = zeros(n_reps)
    K_hat_vals = zeros(n_reps)
    within_mc_var = zeros(n_reps)
    
    for rep in 1:n_reps
        Y = randn(n_subjects) .* σ
        μ_hat = mean(Y)
        
        # True subject scores at MLE
        true_g = (Y .- μ_hat) ./ σ^2
        
        K_true = sum(true_g.^2)
        K_true_vals[rep] = K_true
        
        # MC estimates
        K_hat = 0.0
        total_within_var = 0.0
        
        for i in 1:n_subjects
            # Path-level gradients with MC noise
            g_paths = true_g[i] .+ randn(n_paths) .* τ
            
            # Importance-weighted estimate (uniform weights)
            g_hat_i = mean(g_paths)
            
            # Within-subject MC variance
            within_var_i = var(g_paths) / n_paths
            total_within_var += within_var_i
            
            K_hat += g_hat_i^2
        end
        
        K_hat_vals[rep] = K_hat
        within_mc_var[rep] = total_within_var
    end
    
    println("n_subjects = $n_subjects, n_paths = $n_paths")
    @printf("  E[K_true]:                %.4f\n", mean(K_true_vals))
    @printf("  E[K̂]:                     %.4f\n", mean(K_hat_vals))
    @printf("  E[Σᵢ Var_MC(ĝᵢ)]:         %.4f\n", mean(within_mc_var))
    @printf("  Inflation = E[K̂] - E[K]:  %.4f (should ≈ E[Σᵢ Var_MC(ĝᵢ)])\n", 
            mean(K_hat_vals) - mean(K_true_vals))
    println()
    
    # This inflation directly biases the sandwich variance upward
    H = -n_subjects / σ^2
    I_obs = -H
    
    V_from_K_true = mean(K_true_vals) / I_obs^2
    V_from_K_hat = mean(K_hat_vals) / I_obs^2
    
    @printf("  V from K_true: %.6f\n", V_from_K_true)
    @printf("  V from K̂:     %.6f\n", V_from_K_hat)
    @printf("  V_true:        %.6f\n", V_true)
    @printf("  Upward bias:   %+.2f%%\n", (V_from_K_hat - V_true) / V_true * 100)
    println()
end

# Show how MC noise inflates K, and how it decreases with more paths
for n_paths in [20, 50, 100, 500, 1000]
    mc_variance_inflation_demo(n_paths=n_paths)
end

println("="^70)
println("Key Insight:")
println("- MC noise in ĝᵢ INFLATES K̂, biasing V̂ upward")
println("- Bias ∝ (within-subject variance) / n_paths")
println("- This is separate from (and larger than) the g-H correlation issue")
println("- With typical MCEM (1000+ paths), this inflation is < 1%")
println("="^70)

# ============================================================================
# Final Summary
# ============================================================================

println()
println("="^70)
println("SUMMARY: Monte Carlo Effects on IJ Variance in MCEM")
println("="^70)
println()
println("""
There are TWO separate Monte Carlo effects on the IJ variance estimator:

1. **K Matrix Inflation** (Primary Effect)
   - ĝᵢ = gᵢ + εᵢ where εᵢ is MC noise
   - K̂ = Σᵢ ĝᵢ² = Σᵢ (gᵢ + εᵢ)² = K + Σᵢ εᵢ² + 2Σᵢ gᵢεᵢ
   - At MLE, gᵢ ≈ 0, so E[K̂] ≈ K + Σᵢ Var(εᵢ)
   - This INFLATES variance estimates
   - Magnitude: O(n_subjects × τ² / n_paths)

2. **g-H Correlation** (Secondary Effect)
   - Same MC paths used for both ĝ and Ĥ
   - Creates Cov(ĝ, Ĥ) ≠ 0
   - Affects products like Ĥ⁻¹ K̂ Ĥ⁻¹
   - Usually much smaller than Effect 1

Both effects vanish as n_paths → ∞.

For typical MCEM with n_paths ≥ 1000:
- Effect 1 contributes < 0.1% relative error
- Effect 2 is negligible
- Standard IJ estimator is reliable

Potential correction for small n_paths:
  K̂_corrected = Σᵢ ĝᵢ² - Σᵢ V̂_MC(ĝᵢ)

where V̂_MC(ĝᵢ) = Σⱼ wᵢⱼ(gᵢⱼ - ĝᵢ)² is the within-subject MC variance.
""")
println("="^70)
