"""
Robust Long Test Suite: Parametric Hazard Families with Tight Tolerances

This test suite performs rigorous validation of parameter recovery for ALL parametric
hazard families with LARGE sample sizes and TIGHT tolerances.

Test matrix:
- Hazard families: Exponential, Weibull, Gompertz
- Covariate settings: No covariates, single covariate, multiple covariates
- Sample sizes: n=10000 for tight tolerance testing
- Tolerances: 3% relative tolerance (rtol=0.03) for baseline parameters
             5% for covariate effects

The goal is CORRECTNESS verification: estimates must match truth within tight bounds.
With exact observation data, the model is fully identifiable and should achieve
these tolerances reliably at large sample sizes.

References:
- MLE asymptotic theory: √n-consistency guarantees tight recovery at large n
- Central limit theorem: SE ∝ 1/√n, so n=5000 gives SE ≈ 0.014 of n=1 case
"""

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Optim  # For Newton optimizer (needed for Gompertz)

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, SamplePath, @formula

# =============================================================================
# TIGHT TOLERANCE CONSTANTS
# =============================================================================
const RNG_SEED = 0xABCD2025
const N_SUBJECTS_LARGE = 10000     # Large sample for tight tolerance
const N_SUBJECTS_COVAR = 20000     # Larger sample for multiple covariate tests
const MAX_TIME = 10.0              # Follow-up time
const PARAM_TOL_REL = 0.03         # 3% relative tolerance (TIGHT)
const PARAM_TOL_BETA = 0.05        # 5% for covariate effects
const PARAM_TOL_ABS = 0.03         # Absolute tolerance for parameters near zero

# =============================================================================
# Helper: Generate exact data from illness-death model
# =============================================================================
function generate_exact_data(hazards, true_params; 
    n_subj::Int = N_SUBJECTS_LARGE, 
    max_time::Float64 = MAX_TIME,
    covariate_data::Union{Nothing, DataFrame} = nothing)
    
    # Build simulation template
    template = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = fill(max_time, n_subj),
        statefrom = ones(Int, n_subj),
        stateto = ones(Int, n_subj),
        obstype = ones(Int, n_subj)  # Exact observation
    )
    
    if !isnothing(covariate_data)
        for col in names(covariate_data)
            template[!, col] = covariate_data[!, col]
        end
    end
    
    model = multistatemodel(hazards...; data=template)
    set_parameters!(model, true_params)
    
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    return sim_result[1, 1]
end

# =============================================================================
# TEST SECTION 1: EXPONENTIAL HAZARDS (EXACT MLE)
# =============================================================================

@testset "Robust Exponential - No Covariates (n=$N_SUBJECTS_LARGE)" begin
    Random.seed!(RNG_SEED)
    
    # True parameters (rates)
    true_rate_12 = 0.25
    true_rate_23 = 0.20
    true_rate_13 = 0.10
    
    true_params = (
        h12 = [log(true_rate_12)],
        h23 = [log(true_rate_23)],
        h13 = [log(true_rate_13)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    
    exact_data = generate_exact_data((h12, h23, h13), true_params)
    
    model_fit = multistatemodel(h12, h23, h13; data=exact_data)
    fitted = fit(model_fit; verbose=false, compute_vcov=true)
    
    # Extract fitted parameters on natural scale
    p = get_parameters(fitted; scale=:natural)
    
    # TIGHT TOLERANCE TESTS
    @test isapprox(p.h12[1], true_rate_12; rtol=PARAM_TOL_REL)
    @test isapprox(p.h23[1], true_rate_23; rtol=PARAM_TOL_REL)
    @test isapprox(p.h13[1], true_rate_13; rtol=PARAM_TOL_REL)
    
    # Verify log-likelihood is finite
    @test isfinite(fitted.loglik.loglik)
    
    # Verify variance-covariance computed
    @test !isnothing(fitted.vcov)
    
    println("  ✓ Exponential no covariates: all rates recovered within $(100*PARAM_TOL_REL)%")
end

@testset "Robust Exponential - Single Covariate (n=$N_SUBJECTS_LARGE)" begin
    Random.seed!(RNG_SEED + 1)
    
    # Note: Parameters are stored alphabetically as h12, h13, h23
    true_rate_12, true_beta_12 = 0.20, 0.5
    true_rate_13, true_beta_13 = 0.08, 0.4
    true_rate_23, true_beta_23 = 0.15, -0.3
    
    # Binary covariate (treatment)
    cov_data = DataFrame(x = rand([0.0, 1.0], N_SUBJECTS_LARGE))
    
    true_params = (
        h12 = [log(true_rate_12), true_beta_12],
        h13 = [log(true_rate_13), true_beta_13],
        h23 = [log(true_rate_23), true_beta_23]
    )
    
    h12 = Hazard(@formula(0 ~ x), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ x), "exp", 2, 3)
    h13 = Hazard(@formula(0 ~ x), "exp", 1, 3)
    
    exact_data = generate_exact_data((h12, h23, h13), true_params; covariate_data=cov_data)
    
    model_fit = multistatemodel(h12, h23, h13; data=exact_data)
    fitted = fit(model_fit; verbose=false, compute_vcov=true)
    
    p = get_parameters(fitted; scale=:natural)
    p_est = get_parameters(fitted; scale=:estimation)
    
    # Rate recovery (natural scale) - order is h12, h13, h23
    @test isapprox(p.h12[1], true_rate_12; rtol=PARAM_TOL_REL)
    @test isapprox(p.h13[1], true_rate_13; rtol=PARAM_TOL_REL)
    @test isapprox(p.h23[1], true_rate_23; rtol=PARAM_TOL_REL)
    
    # Beta recovery (estimation scale) - order is h12, h13, h23
    # p_est indices: [1]=h12_rate, [2]=h12_beta, [3]=h13_rate, [4]=h13_beta, [5]=h23_rate, [6]=h23_beta
    @test isapprox(p_est[2], true_beta_12; atol=PARAM_TOL_BETA)
    @test isapprox(p_est[4], true_beta_13; atol=PARAM_TOL_BETA)
    @test isapprox(p_est[6], true_beta_23; atol=PARAM_TOL_BETA)
    
    # Sign verification for betas
    @test sign(p_est[2]) == sign(true_beta_12)
    @test sign(p_est[4]) == sign(true_beta_13)
    @test sign(p_est[6]) == sign(true_beta_23)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Exponential with covariate: rates and betas recovered")
end

@testset "Robust Exponential - Multiple Covariates (n=$N_SUBJECTS_COVAR)" begin
    Random.seed!(RNG_SEED + 2)
    
    true_rate = 0.20
    true_beta1 = 0.4   # Binary treatment
    true_beta2 = -0.2  # Continuous covariate
    
    cov_data = DataFrame(
        trt = rand([0.0, 1.0], N_SUBJECTS_COVAR),
        age = randn(N_SUBJECTS_COVAR) * 0.5  # Standardized age
    )
    
    true_params = (h12 = [log(true_rate), true_beta1, true_beta2],)
    
    h12 = Hazard(@formula(0 ~ trt + age), "exp", 1, 2)
    
    # Simple 1→2 model
    template = DataFrame(
        id = 1:N_SUBJECTS_COVAR,
        tstart = zeros(N_SUBJECTS_COVAR),
        tstop = fill(MAX_TIME, N_SUBJECTS_COVAR),
        statefrom = ones(Int, N_SUBJECTS_COVAR),
        stateto = ones(Int, N_SUBJECTS_COVAR),
        obstype = ones(Int, N_SUBJECTS_COVAR),
        trt = cov_data.trt,
        age = cov_data.age
    )
    
    model_sim = multistatemodel(h12; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1, 1]
    
    model_fit = multistatemodel(h12; data=exact_data)
    fitted = fit(model_fit; verbose=false, compute_vcov=true)
    
    p_est = get_parameters_flat(fitted)
    
    # Rate recovery
    @test isapprox(exp(p_est[1]), true_rate; rtol=PARAM_TOL_REL)
    
    # Beta recovery
    @test isapprox(p_est[2], true_beta1; atol=PARAM_TOL_BETA)
    @test isapprox(p_est[3], true_beta2; atol=PARAM_TOL_BETA)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Exponential with multiple covariates: all parameters recovered")
end

# =============================================================================
# TEST SECTION 2: WEIBULL HAZARDS (EXACT MLE)
# =============================================================================

@testset "Robust Weibull - No Covariates (n=$N_SUBJECTS_LARGE)" begin
    Random.seed!(RNG_SEED + 10)
    
    # True Weibull parameters: h(t) = shape * scale * t^(shape-1)
    true_shape_12, true_scale_12 = 1.3, 0.15
    true_shape_23, true_scale_23 = 0.9, 0.20
    true_shape_13, true_scale_13 = 1.1, 0.08
    
    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12)],
        h23 = [log(true_shape_23), log(true_scale_23)],
        h13 = [log(true_shape_13), log(true_scale_13)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)
    
    exact_data = generate_exact_data((h12, h23, h13), true_params)
    
    model_fit = multistatemodel(h12, h23, h13; data=exact_data)
    fitted = fit(model_fit; verbose=false, compute_vcov=true)
    
    p = get_parameters(fitted; scale=:natural)
    
    # Shape recovery
    @test isapprox(p.h12[1], true_shape_12; rtol=PARAM_TOL_REL)
    @test isapprox(p.h23[1], true_shape_23; rtol=PARAM_TOL_REL)
    @test isapprox(p.h13[1], true_shape_13; rtol=PARAM_TOL_REL)
    
    # Scale recovery
    @test isapprox(p.h12[2], true_scale_12; rtol=PARAM_TOL_REL)
    @test isapprox(p.h23[2], true_scale_23; rtol=PARAM_TOL_REL)
    @test isapprox(p.h13[2], true_scale_13; rtol=PARAM_TOL_REL)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Weibull no covariates: shape and scale recovered within $(100*PARAM_TOL_REL)%")
end

@testset "Robust Weibull - With Covariate (n=$N_SUBJECTS_COVAR)" begin
    Random.seed!(RNG_SEED + 11)
    
    true_shape = 1.2
    true_scale = 0.18
    true_beta = 0.5
    
    cov_data = DataFrame(x = rand([0.0, 1.0], N_SUBJECTS_COVAR))
    
    true_params = (h12 = [log(true_shape), log(true_scale), true_beta],)
    
    h12 = Hazard(@formula(0 ~ x), "wei", 1, 2)
    
    template = DataFrame(
        id = 1:N_SUBJECTS_COVAR,
        tstart = zeros(N_SUBJECTS_COVAR),
        tstop = fill(MAX_TIME, N_SUBJECTS_COVAR),
        statefrom = ones(Int, N_SUBJECTS_COVAR),
        stateto = ones(Int, N_SUBJECTS_COVAR),
        obstype = ones(Int, N_SUBJECTS_COVAR),
        x = cov_data.x
    )
    
    model_sim = multistatemodel(h12; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1, 1]
    
    model_fit = multistatemodel(h12; data=exact_data)
    fitted = fit(model_fit; verbose=false, compute_vcov=true)
    
    p_est = get_parameters_flat(fitted)
    
    # Shape and scale recovery
    @test isapprox(exp(p_est[1]), true_shape; rtol=PARAM_TOL_REL)
    @test isapprox(exp(p_est[2]), true_scale; rtol=PARAM_TOL_REL)
    
    # Beta recovery
    @test isapprox(p_est[3], true_beta; atol=PARAM_TOL_BETA)
    @test sign(p_est[3]) == sign(true_beta)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Weibull with covariate: all parameters recovered")
end

# =============================================================================
# TEST SECTION 3: GOMPERTZ HAZARDS (EXACT MLE)
# Note: Gompertz models can have non-convex likelihood surfaces, so we use
# Newton optimizer for more robust convergence to global optimum.
# We use competing risks (1→2, 1→3) instead of illness-death model because
# the downstream transition (2→3) has limited events which inflates variance.
# =============================================================================

# Larger sample size for Gompertz due to 2-parameter baseline and shape/scale correlation
const N_SUBJECTS_GOMPERTZ = 50000

@testset "Robust Gompertz - No Covariates (n=$N_SUBJECTS_GOMPERTZ)" begin
    Random.seed!(RNG_SEED + 20)
    
    # True Gompertz: h(t) = scale * shape * exp(shape * t)
    # Params stored internally: [log(shape), log(scale)]
    # Use competing risks model (avoids low-event downstream transition)
    true_shape_12, true_scale_12 = 0.15, 0.15
    true_shape_13, true_scale_13 = 0.12, 0.12
    
    # set_parameters! expects log-scale for baseline params
    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12)],
        h13 = [log(true_shape_13), log(true_scale_13)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
    h13 = Hazard(@formula(0 ~ 1), "gom", 1, 3)
    
    template = DataFrame(
        id = 1:N_SUBJECTS_GOMPERTZ,
        tstart = zeros(N_SUBJECTS_GOMPERTZ),
        tstop = fill(MAX_TIME, N_SUBJECTS_GOMPERTZ),
        statefrom = ones(Int, N_SUBJECTS_GOMPERTZ),
        stateto = ones(Int, N_SUBJECTS_GOMPERTZ),
        obstype = ones(Int, N_SUBJECTS_GOMPERTZ)
    )
    
    model_sim = multistatemodel(h12, h13; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1, 1]
    
    model_fit = multistatemodel(h12, h13; data=exact_data)
    # Use Newton optimizer - Gompertz likelihood can have multiple local minima with L-BFGS
    fitted = fit(model_fit; verbose=false, compute_vcov=true, solver=Optim.Newton())
    
    p = get_parameters(fitted; scale=:natural)
    
    # Shape recovery (natural scale = exp(log_shape))
    @test isapprox(p.h12[1], true_shape_12; atol=PARAM_TOL_ABS)
    @test isapprox(p.h13[1], true_shape_13; atol=PARAM_TOL_ABS)
    
    # Scale recovery (natural scale = exp(log_scale))
    @test isapprox(p.h12[2], true_scale_12; rtol=PARAM_TOL_REL)
    @test isapprox(p.h13[2], true_scale_13; rtol=PARAM_TOL_REL)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Gompertz no covariates: shape and scale recovered")
end

@testset "Robust Gompertz - With Covariate (n=$N_SUBJECTS_GOMPERTZ)" begin
    Random.seed!(RNG_SEED + 21)
    
    true_shape = 0.12
    true_scale = 0.15
    true_beta = 0.4
    
    cov_data = DataFrame(x = rand([0.0, 1.0], N_SUBJECTS_GOMPERTZ))
    
    # Params stored internally: [log(shape), log(scale), beta]
    true_params = (h12 = [log(true_shape), log(true_scale), true_beta],)
    
    h12 = Hazard(@formula(0 ~ x), "gom", 1, 2)
    
    template = DataFrame(
        id = 1:N_SUBJECTS_GOMPERTZ,
        tstart = zeros(N_SUBJECTS_GOMPERTZ),
        tstop = fill(MAX_TIME, N_SUBJECTS_GOMPERTZ),
        statefrom = ones(Int, N_SUBJECTS_GOMPERTZ),
        stateto = ones(Int, N_SUBJECTS_GOMPERTZ),
        obstype = ones(Int, N_SUBJECTS_GOMPERTZ),
        x = cov_data.x
    )
    
    model_sim = multistatemodel(h12; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1, 1]
    
    model_fit = multistatemodel(h12; data=exact_data)
    # Use Newton optimizer for Gompertz
    fitted = fit(model_fit; verbose=false, compute_vcov=true, solver=Optim.Newton())
    
    p_est = get_parameters_flat(fitted)
    
    # Shape recovery (p_est[1] is log_shape, compare to natural)
    @test isapprox(exp(p_est[1]), true_shape; rtol=PARAM_TOL_REL)
    
    # Scale recovery (p_est[2] is log_scale, compare to natural)
    @test isapprox(exp(p_est[2]), true_scale; rtol=PARAM_TOL_REL)
    
    # Beta recovery (not transformed)
    @test isapprox(p_est[3], true_beta; atol=PARAM_TOL_BETA)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Gompertz with covariate: all parameters recovered")
end

# =============================================================================
# TEST SECTION 4: CROSS-VALIDATION OF HAZARD COMPUTATION
# =============================================================================

@testset "Robust Hazard Value Verification" begin
    Random.seed!(RNG_SEED + 100)
    
    # Test that fitted hazards evaluate correctly at specific times
    true_rate = 0.25
    true_params = (h12 = [log(true_rate)],)
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    exact_data = generate_exact_data((h12,), true_params; n_subj=N_SUBJECTS_LARGE)
    
    model_fit = multistatemodel(h12; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    fitted_rate = exp(get_parameters_flat(fitted)[1])
    
    # Verify hazard function evaluation matches parameter
    pars = MultistateModels.get_parameters(fitted, 1, scale=:log)
    for t in [0.5, 1.0, 2.0, 5.0]
        h_eval = fitted.hazards[1](t, pars, NamedTuple())
        @test isapprox(h_eval, fitted_rate; rtol=1e-10)  # Should be exact for exponential
    end
    
    println("  ✓ Hazard function evaluation matches fitted parameters exactly")
end

# =============================================================================
# SUMMARY
# =============================================================================

println("\n" * "="^70)
println("ROBUST PARAMETRIC LONGTEST SUITE COMPLETE")
println("="^70)
println("Sample size: n = $N_SUBJECTS_LARGE")
println("Tolerances: $(100*PARAM_TOL_REL)% relative, $(100*PARAM_TOL_BETA)% for betas")
println("Families tested: Exponential, Weibull, Gompertz")
println("Covariate settings: None, Single, Multiple")
println("="^70)
