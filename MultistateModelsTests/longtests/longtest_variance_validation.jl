"""
Robust Long Test Suite: Variance-Covariance Estimation Validation

This test suite validates the IJ (infinitesimal jackknife/sandwich) and JK (jackknife)
variance estimators by comparing:

1. **IJ vs Model-based variance**: Under correct model specification, IJ variance ≈ model-based variance
2. **Estimated vs Empirical variance**: Variance estimates should match empirical variance from simulations
3. **JK = ((n-1)/n) × IJ relationship**: Algebraic identity between JK and IJ

Test strategy:
- Use large sample sizes (n=2000) to reduce MC noise
- Repeat simulations (n_reps=200) to estimate empirical variance
- Compare variances using F-ratio (eigenvalue ratio) and element-wise comparisons

Variance estimator formulas:
- Model-based: Var(θ̂) = H⁻¹ (inverse observed Fisher information)
- IJ (sandwich): Var_{IJ}(θ̂) = H⁻¹ K H⁻¹ where K = Σᵢ gᵢgᵢᵀ (sum of score outer products)
- JK: Var_{JK}(θ̂) = ((n-1)/n) × H⁻¹ K H⁻¹ = ((n-1)/n) × Var_{IJ}(θ̂)

References:
- Huber (1967) - Robust sandwich estimator
- White (1982) - Heteroskedasticity-consistent covariance
- Louis (1982) - Observed information for EM
"""

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, get_vcov, set_surrogate!, @formula

# =============================================================================
# CONSTANTS
# =============================================================================
const RNG_SEED = 0xABCD2030
const N_SUBJECTS = 2000           # Large sample for variance validation
const N_REPS = 1000               # Number of simulation replicates for empirical variance
const MAX_TIME = 10.0             # Follow-up time
const VAR_RATIO_TOL = 0.5         # 50% tolerance on variance ratio (ratio should be ~1)
const DIAG_VAR_TOL = 0.25         # 25% tolerance on diagonal variance elements (tighter with 1000 reps)
const JK_IJ_RATIO_TOL = 1e-10     # JK = ((n-1)/n) * IJ is algebraic identity

# =============================================================================
# Helper: Generate exact data from simple 2-state model
# =============================================================================
function generate_exact_data_2state(hazard, true_params; n_subj::Int = N_SUBJECTS)
    template = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = fill(MAX_TIME, n_subj),
        statefrom = ones(Int, n_subj),
        stateto = ones(Int, n_subj),
        obstype = ones(Int, n_subj)  # Exact observation
    )
    
    model = multistatemodel(hazard; data=template)
    set_parameters!(model, true_params)
    
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    return sim_result[1, 1]
end

# =============================================================================
# Helper: Generate panel data from illness-death model
# =============================================================================
function generate_panel_data_illnessdeath(hazards, true_params;
    n_subj::Int = N_SUBJECTS,
    obs_times::Vector{Float64} = [0.0, 2.5, 5.0, 7.5, MAX_TIME])
    
    nobs = length(obs_times) - 1
    
    template = DataFrame(
        id = repeat(1:n_subj, inner=nobs),
        tstart = repeat(obs_times[1:end-1], n_subj),
        tstop = repeat(obs_times[2:end], n_subj),
        statefrom = ones(Int, n_subj * nobs),
        stateto = ones(Int, n_subj * nobs),
        obstype = fill(2, n_subj * nobs)  # Panel observation
    )
    
    model = multistatemodel(hazards...; data=template)
    set_parameters!(model, true_params)
    
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false)
    return sim_result[1]
end

# =============================================================================
# Helper: Compute empirical variance from repeated simulations
# =============================================================================
function compute_empirical_variance(hazard_tuple, true_params, generate_data_fn; n_reps::Int = N_REPS)
    param_estimates = Vector{Vector{Float64}}()
    
    for rep in 1:n_reps
        # Generate new data
        data = generate_data_fn(hazard_tuple, true_params)
        
        # Refit model
        model = multistatemodel(hazard_tuple...; data=data)
        fitted = fit(model; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        
        push!(param_estimates, get_parameters_flat(fitted))
    end
    
    # Stack estimates into matrix (n_reps × n_params)
    param_matrix = reduce(hcat, param_estimates)'
    
    # Compute empirical covariance
    return cov(param_matrix)
end

# =============================================================================
# TEST SECTION 1: IJ vs MODEL-BASED VARIANCE (Exponential, Correct Specification)
# =============================================================================

@testset "IJ vs Model-based Variance (Exponential Exact Data)" begin
    Random.seed!(RNG_SEED)
    
    # Under correct specification, IJ variance ≈ model-based variance
    # This is because Var(gᵢ) ≈ -E[Hᵢ] holds when model is correct
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    true_rate = 0.25
    true_params = (h12 = [log(true_rate)],)
    
    exact_data = generate_exact_data_2state(h12, true_params; n_subj=N_SUBJECTS)
    
    model = multistatemodel(h12; data=exact_data)
    fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=true)
    
    vcov_model = get_vcov(fitted; type=:model)
    vcov_ij = get_vcov(fitted; type=:ij)
    
    @test !isnothing(vcov_model)
    @test !isnothing(vcov_ij)
    
    # Ratio of diagonals should be close to 1 under correct specification
    diag_ratio = diag(vcov_ij) ./ diag(vcov_model)
    
    @test all(isapprox.(diag_ratio, 1.0; rtol=VAR_RATIO_TOL))
    
    # Matrix norm ratio (Frobenius)
    frob_ratio = norm(vcov_ij) / norm(vcov_model)
    @test isapprox(frob_ratio, 1.0; rtol=VAR_RATIO_TOL)
    
    println("  ✓ IJ vs model-based variance ratio: $(round.(diag_ratio, digits=3))")
    println("  ✓ Frobenius norm ratio: $(round(frob_ratio, digits=3))")
end

@testset "IJ vs Model-based Variance (Weibull Exact Data)" begin
    Random.seed!(RNG_SEED + 1)
    
    # Weibull with 2 parameters (shape, scale)
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    true_shape, true_scale = 1.3, 0.20
    true_params = (h12 = [log(true_shape), log(true_scale)],)
    
    exact_data = generate_exact_data_2state(h12, true_params; n_subj=N_SUBJECTS)
    
    model = multistatemodel(h12; data=exact_data)
    fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=true)
    
    vcov_model = get_vcov(fitted; type=:model)
    vcov_ij = get_vcov(fitted; type=:ij)
    
    diag_ratio = diag(vcov_ij) ./ diag(vcov_model)
    
    @test all(isapprox.(diag_ratio, 1.0; rtol=VAR_RATIO_TOL))
    
    println("  ✓ Weibull IJ vs model-based variance ratio: $(round.(diag_ratio, digits=3))")
end

@testset "IJ vs Model-based Variance (Markov Panel Data)" begin
    Random.seed!(RNG_SEED + 2)
    
    # Illness-death model with panel data
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    
    true_params = (
        h12 = [log(0.25)],
        h23 = [log(0.20)],
        h13 = [log(0.10)]
    )
    
    panel_data = generate_panel_data_illnessdeath((h12, h23, h13), true_params)
    
    model = multistatemodel(h12, h23, h13; data=panel_data)
    fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=true)
    
    vcov_model = get_vcov(fitted; type=:model)
    vcov_ij = get_vcov(fitted; type=:ij)
    
    diag_ratio = diag(vcov_ij) ./ diag(vcov_model)
    
    @test all(isapprox.(diag_ratio, 1.0; rtol=VAR_RATIO_TOL))
    
    println("  ✓ Markov panel IJ vs model-based variance ratio: $(round.(diag_ratio, digits=3))")
end

# =============================================================================
# TEST SECTION 2: JK = ((n-1)/n) × IJ RELATIONSHIP (Algebraic Identity)
# =============================================================================

@testset "JK = ((n-1)/n) × IJ Relationship (Exact Data)" begin
    Random.seed!(RNG_SEED + 10)
    
    # This is an algebraic identity, NOT a statistical property
    # JK variance = ((n-1)/n) × IJ variance EXACTLY
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    true_params = (h12 = [log(1.2), log(0.15)],)
    
    exact_data = generate_exact_data_2state(h12, true_params; n_subj=N_SUBJECTS)
    
    model = multistatemodel(h12; data=exact_data)
    fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=true, compute_jk_vcov=true)
    
    vcov_ij = get_vcov(fitted; type=:ij)
    vcov_jk = get_vcov(fitted; type=:jk)
    
    @test !isnothing(vcov_ij)
    @test !isnothing(vcov_jk)
    
    # Expected relationship
    n = N_SUBJECTS
    expected_jk = ((n - 1) / n) * vcov_ij
    
    # Should match EXACTLY (algebraic identity)
    @test isapprox(vcov_jk, expected_jk; atol=JK_IJ_RATIO_TOL)
    
    println("  ✓ JK = ((n-1)/n) × IJ verified (algebraic identity)")
end

@testset "JK = ((n-1)/n) × IJ Relationship (Panel Data)" begin
    Random.seed!(RNG_SEED + 11)
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    
    true_params = (
        h12 = [log(0.30)],
        h23 = [log(0.25)],
        h13 = [log(0.15)]
    )
    
    panel_data = generate_panel_data_illnessdeath((h12, h23, h13), true_params)
    
    model = multistatemodel(h12, h23, h13; data=panel_data)
    fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=true, compute_jk_vcov=true)
    
    vcov_ij = get_vcov(fitted; type=:ij)
    vcov_jk = get_vcov(fitted; type=:jk)
    
    n = length(unique(panel_data.id))
    expected_jk = ((n - 1) / n) * vcov_ij
    
    @test isapprox(vcov_jk, expected_jk; atol=JK_IJ_RATIO_TOL)
    
    println("  ✓ JK = ((n-1)/n) × IJ verified for panel data")
end

# =============================================================================
# TEST SECTION 3: ESTIMATED VS EMPIRICAL VARIANCE (Simulation Study)
# =============================================================================

@testset "Model-based Variance vs Empirical Variance (Exponential)" begin
    Random.seed!(RNG_SEED + 20)
    
    # Compare estimated variance to empirical variance from repeated simulations
    # Empirical variance = sample covariance of θ̂ across N_REPS replicates
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    true_rate = 0.25
    true_params = (h12 = [log(true_rate)],)
    
    # Fit one model to get estimated variance
    exact_data = generate_exact_data_2state(h12, true_params; n_subj=N_SUBJECTS)
    model = multistatemodel(h12; data=exact_data)
    fitted = fit(model; verbose=false, compute_vcov=true)
    vcov_model = get_vcov(fitted; type=:model)
    
    # Compute empirical variance from repeated simulations
    println("  Computing empirical variance from $N_REPS replicates...")
    
    param_estimates = Vector{Vector{Float64}}()
    for rep in 1:N_REPS
        Random.seed!(RNG_SEED + 20 + rep)
        data = generate_exact_data_2state(h12, true_params; n_subj=N_SUBJECTS)
        m = multistatemodel(h12; data=data)
        f = fit(m; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        push!(param_estimates, get_parameters_flat(f))
    end
    
    param_matrix = reduce(hcat, param_estimates)'
    vcov_empirical = cov(param_matrix)
    
    # Compare diagonal elements (variances)
    var_ratio = diag(vcov_model) ./ diag(vcov_empirical)
    
    @test all(isapprox.(var_ratio, 1.0; rtol=DIAG_VAR_TOL))
    
    println("  ✓ Model-based vs empirical variance ratio: $(round.(var_ratio, digits=3))")
    println("    Model-based SE: $(round.(sqrt.(diag(vcov_model)), digits=4))")
    println("    Empirical SE: $(round.(sqrt.(diag(vcov_empirical)), digits=4))")
end

@testset "Model-based Variance vs Empirical Variance (Weibull)" begin
    Random.seed!(RNG_SEED + 30)
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    true_params = (h12 = [log(1.25), log(0.18)],)
    
    # Fit one model to get estimated variance
    exact_data = generate_exact_data_2state(h12, true_params; n_subj=N_SUBJECTS)
    model = multistatemodel(h12; data=exact_data)
    fitted = fit(model; verbose=false, compute_vcov=true)
    vcov_model = get_vcov(fitted; type=:model)
    
    # Compute empirical variance
    println("  Computing empirical variance from $N_REPS replicates...")
    
    param_estimates = Vector{Vector{Float64}}()
    for rep in 1:N_REPS
        Random.seed!(RNG_SEED + 30 + rep)
        data = generate_exact_data_2state(h12, true_params; n_subj=N_SUBJECTS)
        m = multistatemodel(h12; data=data)
        f = fit(m; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        push!(param_estimates, get_parameters_flat(f))
    end
    
    param_matrix = reduce(hcat, param_estimates)'
    vcov_empirical = cov(param_matrix)
    
    var_ratio = diag(vcov_model) ./ diag(vcov_empirical)
    
    @test all(isapprox.(var_ratio, 1.0; rtol=DIAG_VAR_TOL))
    
    println("  ✓ Weibull model-based vs empirical variance ratio: $(round.(var_ratio, digits=3))")
    println("    Model-based SE: $(round.(sqrt.(diag(vcov_model)), digits=4))")
    println("    Empirical SE: $(round.(sqrt.(diag(vcov_empirical)), digits=4))")
end

@testset "IJ Variance vs Empirical Variance (Exponential)" begin
    Random.seed!(RNG_SEED + 40)
    
    # IJ variance should also match empirical variance under correct specification
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    true_params = (h12 = [log(0.25)],)
    
    # Fit one model to get IJ variance
    exact_data = generate_exact_data_2state(h12, true_params; n_subj=N_SUBJECTS)
    model = multistatemodel(h12; data=exact_data)
    fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=true)
    vcov_ij = get_vcov(fitted; type=:ij)
    
    # Compute empirical variance
    println("  Computing empirical variance from $N_REPS replicates...")
    
    param_estimates = Vector{Vector{Float64}}()
    for rep in 1:N_REPS
        Random.seed!(RNG_SEED + 40 + rep)
        data = generate_exact_data_2state(h12, true_params; n_subj=N_SUBJECTS)
        m = multistatemodel(h12; data=data)
        f = fit(m; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        push!(param_estimates, get_parameters_flat(f))
    end
    
    param_matrix = reduce(hcat, param_estimates)'
    vcov_empirical = cov(param_matrix)
    
    var_ratio = diag(vcov_ij) ./ diag(vcov_empirical)
    
    @test all(isapprox.(var_ratio, 1.0; rtol=DIAG_VAR_TOL))
    
    println("  ✓ IJ vs empirical variance ratio: $(round.(var_ratio, digits=3))")
end

# =============================================================================
# TEST SECTION 4: VARIANCE ESTIMATION WITH COVARIATES
# =============================================================================

@testset "IJ vs Model-based Variance (With Covariates)" begin
    Random.seed!(RNG_SEED + 50)
    
    # Test with covariate effects
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    true_rate, true_beta = 0.20, 0.5
    
    # Generate data with covariate
    n_subj = N_SUBJECTS
    template = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = fill(MAX_TIME, n_subj),
        statefrom = ones(Int, n_subj),
        stateto = ones(Int, n_subj),
        obstype = ones(Int, n_subj),
        x = rand([0.0, 1.0], n_subj)
    )
    
    h12_cov = Hazard(@formula(0 ~ x), "exp", 1, 2)
    true_params = (h12 = [log(true_rate), true_beta],)
    
    model_sim = multistatemodel(h12_cov; data=template)
    set_parameters!(model_sim, true_params)
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1, 1]
    
    model = multistatemodel(h12_cov; data=exact_data)
    fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=true)
    
    vcov_model = get_vcov(fitted; type=:model)
    vcov_ij = get_vcov(fitted; type=:ij)
    
    diag_ratio = diag(vcov_ij) ./ diag(vcov_model)
    
    @test all(isapprox.(diag_ratio, 1.0; rtol=VAR_RATIO_TOL))
    
    println("  ✓ Covariate model IJ vs model-based ratio: $(round.(diag_ratio, digits=3))")
end

# =============================================================================
# TEST SECTION 5: STANDARD ERROR COVERAGE (Wald CI Coverage)
# =============================================================================

@testset "95% CI Coverage (Exponential)" begin
    Random.seed!(RNG_SEED + 60)
    
    # Verify that 95% Wald CIs achieve approximately 95% coverage
    # This validates that SEs are correctly calibrated
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    true_log_rate = log(0.25)
    true_params = (h12 = [true_log_rate],)
    
    n_covered = 0
    n_trials = 100  # Fewer than N_REPS to keep runtime reasonable
    
    println("  Computing coverage from $n_trials replicates...")
    
    for rep in 1:n_trials
        Random.seed!(RNG_SEED + 60 + rep)
        data = generate_exact_data_2state(h12, true_params; n_subj=N_SUBJECTS)
        m = multistatemodel(h12; data=data)
        f = fit(m; verbose=false, compute_vcov=true)
        
        est = get_parameters_flat(f)[1]
        se = sqrt(diag(get_vcov(f; type=:model))[1])
        
        # 95% Wald CI
        ci_lower = est - 1.96 * se
        ci_upper = est + 1.96 * se
        
        if ci_lower <= true_log_rate <= ci_upper
            n_covered += 1
        end
    end
    
    coverage = n_covered / n_trials
    
    # Coverage should be between 90% and 99% (allowing for MC noise)
    @test 0.88 <= coverage <= 0.99
    
    println("  ✓ 95% CI coverage: $(round(100 * coverage, digits=1))%")
end

# =============================================================================
# TEST SECTION 6: VARIANCE POSITIVE DEFINITENESS
# =============================================================================

@testset "Variance Matrix Positive Definiteness" begin
    Random.seed!(RNG_SEED + 70)
    
    # All variance matrices should be positive semi-definite
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    true_params = (
        h12 = [log(1.2), log(0.20)],
        h23 = [log(0.15)]
    )
    
    # Generate exact data for 3-state model
    n_subj = N_SUBJECTS
    template = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = fill(MAX_TIME, n_subj),
        statefrom = ones(Int, n_subj),
        stateto = ones(Int, n_subj),
        obstype = ones(Int, n_subj)
    )
    
    model_sim = multistatemodel(h12, h23; data=template)
    set_parameters!(model_sim, true_params)
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1, 1]
    
    model = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=true, compute_jk_vcov=true)
    
    vcov_model = get_vcov(fitted; type=:model)
    vcov_ij = get_vcov(fitted; type=:ij)
    vcov_jk = get_vcov(fitted; type=:jk)
    
    # Check positive semi-definiteness (all eigenvalues >= 0)
    eig_model = eigvals(Symmetric(vcov_model))
    eig_ij = eigvals(Symmetric(vcov_ij))
    eig_jk = eigvals(Symmetric(vcov_jk))
    
    @test all(eig_model .>= -sqrt(eps()))
    @test all(eig_ij .>= -sqrt(eps()))
    @test all(eig_jk .>= -sqrt(eps()))
    
    # Diagonal elements should be positive
    @test all(diag(vcov_model) .> 0)
    @test all(diag(vcov_ij) .> 0)
    @test all(diag(vcov_jk) .> 0)
    
    println("  ✓ All variance matrices are positive semi-definite")
    println("    Model eigenvalues: $(round.(eig_model, digits=6))")
    println("    IJ eigenvalues: $(round.(eig_ij, digits=6))")
    println("    JK eigenvalues: $(round.(eig_jk, digits=6))")
end

# =============================================================================
# TEST SECTION 7: SEMI-MARKOV MCEM VARIANCE VALIDATION
# =============================================================================
# 
# For semi-Markov models with panel data, we use Monte Carlo EM (MCEM) for fitting.
# The variance is computed via Louis's identity, which accounts for the missing 
# data (latent paths between observation times). The robust variance (IJ/sandwich)
# uses importance-weighted gradients from the MCEM E-step.
#
# IMPORTANT: MCEM fitting requires a Markov surrogate for importance sampling.
# By default, `surrogate=:markov` with `fit_surrogate=true` creates and fits
# the surrogate during model generation. Alternatively:
#   1. Use `surrogate=:markov, fit_surrogate=false` to defer fitting to fit() time
#   2. Call `set_surrogate!(model)` after model creation to manually fit
# If no surrogate exists, fit() will error: "MCEM requires a Markov surrogate..."
#
# This section validates:
# 1. Model-based variance (Louis's identity) is positive definite
# 2. IJ variance is computed correctly 
# 3. Under correct specification, IJ ≈ model-based variance
# 4. Empirical variance matches estimated variance (simulation study)
# 5. JK = ((n-1)/n) × IJ algebraic identity holds for MCEM
#
# References:
# - Louis (1982) - Observed information for EM (Louis's identity)
# - Morsomme et al. (2025) Biostatistics - MCEM for semi-Markov models
# - Wei & Tanner (1990) - Monte Carlo EM
# =============================================================================

# Constants for MCEM tests (smaller sample sizes due to computational cost)
const MCEM_N_SUBJECTS = 500       # Smaller sample for MCEM (still reasonable)
const MCEM_N_REPS = 50            # Fewer reps due to MCEM cost (~1-2 min each)
const MCEM_VAR_RATIO_TOL = 0.60   # Wider tolerance due to MC noise in both MCEM and variance

"""
    generate_panel_data_semimarkov(hazards, true_params; n_subj, obs_times)

Generate panel data from a semi-Markov model.

This helper simulates panel (interval-censored) data where true state transitions
are only observed at discrete time points. The resulting data is appropriate for
MCEM fitting (semi-Markov models with panel observations).

# Arguments
- `hazards`: Tuple of Hazard objects (must include at least one semi-Markov hazard, e.g., Weibull)
- `true_params`: NamedTuple of true parameter values in log scale
- `n_subj`: Number of subjects to simulate
- `obs_times`: Vector of observation times defining panel intervals

# Returns
- DataFrame with panel data suitable for MCEM fitting

# Notes
- This function creates a model for SIMULATION only (no surrogate needed)
- The returned data should be used to create a FITTING model; by default,
  `surrogate=:markov` fits the surrogate during model creation
"""
function generate_panel_data_semimarkov(hazards, true_params;
    n_subj::Int = MCEM_N_SUBJECTS,
    obs_times::Vector{Float64} = [0.0, 2.0, 4.0, 6.0, 8.0, MAX_TIME])
    
    nobs = length(obs_times) - 1
    
    template = DataFrame(
        id = repeat(1:n_subj, inner=nobs),
        tstart = repeat(obs_times[1:end-1], n_subj),
        tstop = repeat(obs_times[2:end], n_subj),
        statefrom = ones(Int, n_subj * nobs),
        stateto = ones(Int, n_subj * nobs),
        obstype = fill(2, n_subj * nobs)  # Panel observation (obstype=2)
    )
    
    # Create simulation model (no surrogate needed for simulation)
    model = multistatemodel(hazards...; data=template)
    set_parameters!(model, true_params)
    
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false)
    return sim_result[1]
end

@testset "MCEM: IJ vs Model-based Variance (Weibull Semi-Markov)" begin
    Random.seed!(RNG_SEED + 100)
    
    # ==========================================================================
    # Test: Under correct model specification, IJ variance ≈ model-based variance
    # 
    # For MCEM, the model-based variance uses Louis's identity to account for
    # missing data (latent paths). Under correct specification, the sandwich/IJ
    # variance should match the model-based variance.
    # ==========================================================================
    
    # Simple 2-state Weibull model (semi-Markov due to duration dependence)
    # State 1 → State 2 (absorbing)
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    
    # True parameters: shape=1.5 (increasing hazard), scale=0.15
    true_shape = 1.5
    true_scale = 0.15
    true_params = (h12 = [log(true_shape), log(true_scale)],)
    
    # Generate panel data
    panel_data = generate_panel_data_semimarkov((h12,), true_params; n_subj=MCEM_N_SUBJECTS)
    
    # Fit via MCEM - surrogate is created and fitted automatically with surrogate=:markov
    model = multistatemodel(h12; data=panel_data, surrogate=:markov)
    fitted = fit(model; 
        verbose=false, 
        compute_vcov=true, 
        compute_ij_vcov=true,
        ess_target_initial=100,  # Higher ESS for better variance estimate
        maxiter=50,
        tol=0.02)
    
    vcov_model = get_vcov(fitted; type=:model)
    vcov_ij = get_vcov(fitted; type=:ij)
    
    @test !isnothing(vcov_model) "Model-based variance should be computed"
    @test !isnothing(vcov_ij) "IJ variance should be computed"
    
    # Check positive definiteness
    @test all(eigvals(Symmetric(vcov_model)) .>= -sqrt(eps())) "Model vcov should be PSD"
    @test all(eigvals(Symmetric(vcov_ij)) .>= -sqrt(eps())) "IJ vcov should be PSD"
    
    # Under correct specification, IJ ≈ model-based variance
    # Note: For MCEM, we expect more variability due to importance sampling
    diag_ratio = diag(vcov_ij) ./ diag(vcov_model)
    
    @test all(isapprox.(diag_ratio, 1.0; rtol=MCEM_VAR_RATIO_TOL)) "IJ/model ratio should be ~1"
    
    println("  ✓ MCEM IJ vs model-based variance ratio: $(round.(diag_ratio, digits=3))")
    println("    Model-based SE: $(round.(sqrt.(diag(vcov_model)), digits=4))")
    println("    IJ SE: $(round.(sqrt.(diag(vcov_ij)), digits=4))")
end

@testset "MCEM: JK = ((n-1)/n) × IJ Relationship" begin
    Random.seed!(RNG_SEED + 101)
    
    # ==========================================================================
    # Test: JK variance = ((n-1)/n) × IJ variance (algebraic identity)
    #
    # This relationship is an algebraic identity that should hold EXACTLY
    # (up to floating point precision). It does NOT depend on model specification
    # or sample size - it's a mathematical property of the estimators.
    #
    # For MCEM, the gradients are computed using importance-weighted complete-data
    # scores, but the JK/IJ relationship still holds.
    # ==========================================================================
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    true_params = (h12 = [log(1.3), log(0.12)],)
    
    panel_data = generate_panel_data_semimarkov((h12,), true_params; n_subj=MCEM_N_SUBJECTS)
    
    # Create fitting model - surrogate created and fitted automatically
    model = multistatemodel(h12; data=panel_data, surrogate=:markov)
    
    fitted = fit(model; 
        verbose=false, 
        compute_vcov=true, 
        compute_ij_vcov=true,
        compute_jk_vcov=true,
        ess_target_initial=100,
        maxiter=50,
        tol=0.02)
    
    vcov_ij = get_vcov(fitted; type=:ij)
    vcov_jk = get_vcov(fitted; type=:jk)
    
    @test !isnothing(vcov_ij) "IJ variance should be computed"
    @test !isnothing(vcov_jk) "JK variance should be computed"
    
    # JK = ((n-1)/n) × IJ is algebraic identity (should be exact)
    n = length(unique(panel_data.id))
    expected_jk = ((n - 1) / n) * vcov_ij
    
    @test isapprox(vcov_jk, expected_jk; atol=JK_IJ_RATIO_TOL) "JK should equal ((n-1)/n) × IJ exactly"
    
    println("  ✓ MCEM JK = ((n-1)/n) × IJ verified (algebraic identity)")
end

@testset "MCEM: Model-based Variance vs Empirical Variance" begin
    Random.seed!(RNG_SEED + 102)
    
    # ==========================================================================
    # Test: Gold-standard variance validation via simulation
    #
    # This is the definitive test: we compare the estimated variance from a
    # single fit to the empirical variance computed from many repeated fits.
    # If variance estimation is correct, the estimated variance should match
    # the empirical variance of parameter estimates across replicates.
    #
    # This test is computationally expensive because:
    # 1. Each MCEM fit takes ~30 seconds to 2 minutes
    # 2. We need many replicates (50) for a stable empirical variance estimate
    # ==========================================================================
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    true_shape = 1.4
    true_scale = 0.18
    true_params = (h12 = [log(true_shape), log(true_scale)],)
    
    # Fit one model to get estimated variance - surrogate fitted during model creation
    panel_data = generate_panel_data_semimarkov((h12,), true_params; n_subj=MCEM_N_SUBJECTS)
    model = multistatemodel(h12; data=panel_data, surrogate=:markov)
    fitted = fit(model; 
        verbose=false, 
        compute_vcov=true,
        ess_target_initial=100,
        maxiter=50,
        tol=0.02)
    vcov_model = get_vcov(fitted; type=:model)
    
    # Compute empirical variance from repeated MCEM fits
    println("  Computing empirical variance from $MCEM_N_REPS MCEM replicates...")
    println("  (This may take several minutes)")
    
    param_estimates = Vector{Vector{Float64}}()
    n_converged = 0
    
    for rep in 1:MCEM_N_REPS
        Random.seed!(RNG_SEED + 102 + rep)
        
        # Generate new data
        data = generate_panel_data_semimarkov((h12,), true_params; n_subj=MCEM_N_SUBJECTS)
        
        # Fit via MCEM - surrogate fitted during model creation
        m = multistatemodel(h12; data=data, surrogate=:markov)
        try
            f = fit(m; 
                verbose=false, 
                compute_vcov=false, 
                compute_ij_vcov=false,
                ess_target_initial=50,  # Lower ESS for speed
                maxiter=30,
                tol=0.05)  # Looser tolerance for speed
            
            # Only include if converged reasonably
            if !isnothing(f.ConvergenceRecords) && !isempty(f.ConvergenceRecords.mll_trace)
                push!(param_estimates, get_parameters_flat(f))
                n_converged += 1
            end
        catch e
            @debug "MCEM replicate $rep failed" exception=(e, catch_backtrace())
            continue
        end
        
        if rep % 10 == 0
            println("    Completed $rep / $MCEM_N_REPS replicates ($n_converged converged)")
        end
    end
    
    @test n_converged >= 0.8 * MCEM_N_REPS "At least 80% of MCEM fits should converge"
    
    # Compute empirical covariance
    param_matrix = reduce(hcat, param_estimates)'
    vcov_empirical = cov(param_matrix)
    
    # Compare diagonal elements (variances)
    var_ratio = diag(vcov_model) ./ diag(vcov_empirical)
    
    # Wider tolerance due to MCEM variability in both fitting and variance estimation
    @test all(isapprox.(var_ratio, 1.0; rtol=MCEM_VAR_RATIO_TOL))
    
    println("  ✓ MCEM model-based vs empirical variance ratio: $(round.(var_ratio, digits=3))")
    println("    Model-based SE: $(round.(sqrt.(diag(vcov_model)), digits=4))")
    println("    Empirical SE: $(round.(sqrt.(diag(vcov_empirical)), digits=4))")
    println("    ($n_converged / $MCEM_N_REPS replicates converged)")
end

@testset "MCEM: Illness-Death Model Variance (3-state)" begin
    Random.seed!(RNG_SEED + 103)
    
    # ==========================================================================
    # Test: Variance estimation for multi-transition illness-death model
    #
    # This tests variance estimation for a more complex 3-state model with:
    # - h12 (Healthy → Ill): Weibull hazard (semi-Markov - duration dependent)
    # - h23 (Ill → Dead): Exponential hazard (Markov - memoryless)
    # - h13 (Healthy → Dead): Exponential hazard (Markov - memoryless)
    #
    # Because h12 is semi-Markov, the entire model requires MCEM fitting.
    # Total parameters: 4 (2 for Weibull + 1 each for exponentials)
    # ==========================================================================
    
    # State 1: Healthy, State 2: Ill, State 3: Dead (absorbing)
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)  # Healthy → Ill (semi-Markov)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)  # Ill → Dead (Markov)
    h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)  # Healthy → Dead (Markov)
    
    true_params = (
        h12 = [log(1.3), log(0.15)],   # Weibull: log(shape), log(scale)
        h23 = [log(0.25)],              # Exponential: log(rate)
        h13 = [log(0.05)]               # Exponential: log(rate)
    )
    
    panel_data = generate_panel_data_semimarkov((h12, h23, h13), true_params; n_subj=MCEM_N_SUBJECTS)
    
    # Create fitting model - surrogate created and fitted automatically
    model = multistatemodel(h12, h23, h13; data=panel_data, surrogate=:markov)
    
    fitted = fit(model; 
        verbose=false, 
        compute_vcov=true, 
        compute_ij_vcov=true,
        ess_target_initial=100,
        maxiter=50,
        tol=0.02)
    
    vcov_model = get_vcov(fitted; type=:model)
    vcov_ij = get_vcov(fitted; type=:ij)
    
    @test !isnothing(vcov_model) "Model-based variance should be computed"
    @test !isnothing(vcov_ij) "IJ variance should be computed"
    
    # Check dimensions match number of parameters (2 + 1 + 1 = 4)
    @test size(vcov_model) == (4, 4) "Variance matrix should be 4×4"
    @test size(vcov_ij) == (4, 4) "IJ variance matrix should be 4×4"
    
    # Check positive definiteness
    @test all(eigvals(Symmetric(vcov_model)) .>= -sqrt(eps())) "Model vcov should be PSD"
    @test all(eigvals(Symmetric(vcov_ij)) .>= -sqrt(eps())) "IJ vcov should be PSD"
    
    # IJ vs model-based ratio (under correct specification, should be ~1)
    diag_ratio = diag(vcov_ij) ./ diag(vcov_model)
    
    @test all(isapprox.(diag_ratio, 1.0; rtol=MCEM_VAR_RATIO_TOL)) "IJ/model ratio should be ~1"
    
    println("  ✓ MCEM illness-death model variance validation passed")
    println("    IJ/Model ratio: $(round.(diag_ratio, digits=3))")
    println("    Model-based SE: $(round.(sqrt.(diag(vcov_model)), digits=4))")
end

println("\n===== Variance Validation Long Tests Complete =====")
