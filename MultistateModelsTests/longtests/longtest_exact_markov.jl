"""
Long test suite for exact data fitting (MLE parameter recovery).

This test suite validates:
1. **Parameter Recovery**: At sample size n=1000, estimated parameters should be 
   close to true values (within 3 SEs or 15% relative tolerance).
2. **Distributional Fidelity**: Trajectories simulated from fitted models (n=10000) 
   should have similar distributional properties (state prevalence) to trajectories 
   simulated from models with true parameters.

Test matrix:
- Hazard families: exponential, Weibull, Gompertz, spline
- Covariates: none, time-fixed, time-varying (TVC)
- Model structure: progressive 3-state (1→2→3 where 3 is absorbing)
  - State 1: Healthy
  - State 2: Ill
  - State 3: Dead (absorbing)

References:
- Asymptotic MLE theory: estimates √n-consistent, asymptotically normal
- Andersen & Keiding (2002) Statistical Methods in Medical Research - multi-state models
"""

# Import internal types - assumes MultistateModels is already loaded by runtests.jl
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate, 
    ExactData, MPanelData, loglik_exact, loglik_markov, build_tpm_mapping,
    get_parameters, get_parameters_flat, SamplePath, cumulative_incidence
using MultistateModels: @formula

using Test
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Distributions

const RNG_SEED = 0x12345678
const N_SUBJECTS = 1000         # Sample size for fitting
const N_SIM_TRAJ = 10000        # Trajectories for distributional comparison
const MAX_TIME = 20.0           # Maximum follow-up time
const PARAM_TOL_SE = 3.0        # Parameter should be within 3 SEs of truth
const PARAM_TOL_REL = 0.15      # Relative tolerance (15% with n=1000)

# Shared helper functions (compute_state_prevalence, count_transitions, etc.) are loaded
# from longtest_helpers.jl by the test runner.

# ============================================================================
# Helper Functions
# ============================================================================

"""
    generate_exact_data_progressive(hazards, true_params; n_subj, max_time, covariate_data)

Generate exact (continuous-time) data from progressive 3-state model (1→2→3).
Returns DataFrame with columns: id, tstart, tstop, statefrom, stateto, obstype, [covariates...]
"""
function generate_exact_data_progressive(hazards, true_params; 
    n_subj::Int = N_SUBJECTS, 
    max_time::Float64 = MAX_TIME,
    covariate_data::Union{Nothing, DataFrame} = nothing)
    
    # If covariate data provided, use its size
    actual_n_subj = isnothing(covariate_data) ? n_subj : nrow(covariate_data)
    
    # Build template for simulation
    template = DataFrame(
        id = 1:actual_n_subj,
        tstart = zeros(actual_n_subj),
        tstop = fill(max_time, actual_n_subj),
        statefrom = ones(Int, actual_n_subj),
        stateto = ones(Int, actual_n_subj),
        obstype = ones(Int, actual_n_subj)  # Exact observation
    )
    
    if !isnothing(covariate_data)
        template = hcat(template, covariate_data)
    end
    
    model = multistatemodel(hazards...; data=template)
    set_parameters!(model, true_params)
    
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    return sim_result[1]
end

"""
    check_parameter_recovery(fitted, true_params_flat; tol_se, tol_rel)

Verify fitted parameters are close to true values. Returns boolean.
"""
function check_parameter_recovery(fitted, true_params_flat; tol_se=PARAM_TOL_SE, tol_rel=PARAM_TOL_REL)
    fitted_params = get_parameters_flat(fitted)
    
    if !isnothing(fitted.vcov)
        ses = sqrt.(diag(fitted.vcov))
        for i in eachindex(fitted_params)
            se = ses[i]
            if se > 0 && isfinite(se)
                err_se = abs(fitted_params[i] - true_params_flat[i]) / se
                if err_se > tol_se
                    return false
                end
            else
                rel_err = abs(fitted_params[i] - true_params_flat[i]) / abs(true_params_flat[i])
                if rel_err > tol_rel
                    return false
                end
            end
        end
    else
        for i in eachindex(fitted_params)
            rel_err = abs(fitted_params[i] - true_params_flat[i]) / abs(true_params_flat[i])
            if rel_err > tol_rel
                return false
            end
        end
    end
    return true
end

"""
    check_distributional_fidelity(hazards, true_params, fitted_params; 
        n_traj, max_time, max_prev_diff)

Compare state prevalence between true and fitted parameter distributions.
"""
function check_distributional_fidelity(hazards, true_params, fitted_params; 
    n_traj::Int = N_SIM_TRAJ, max_time::Float64 = MAX_TIME, max_prev_diff::Float64 = 0.10,
    covariate_data::Union{Nothing, DataFrame} = nothing)
    
    n_subj = isnothing(covariate_data) ? n_traj : nrow(covariate_data)
    template = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = fill(max_time, n_subj),
        statefrom = ones(Int, n_subj),
        stateto = ones(Int, n_subj),
        obstype = ones(Int, n_subj)
    )
    
    if !isnothing(covariate_data)
        template = hcat(template, covariate_data)
    end
    
    n_states = 3
    eval_times = collect(0.0:1.0:max_time)
    
    # Simulate from true parameters
    model_true = multistatemodel(hazards...; data=template)
    set_parameters!(model_true, true_params)
    
    Random.seed!(RNG_SEED + 999)
    trajectories_true = simulate(model_true; paths=true, data=false, nsim=1)
    paths_true = trajectories_true[1]
    
    # Simulate from fitted parameters - use same model, just set different params
    model_fitted = multistatemodel(hazards...; data=template)
    # Convert flat params to named tuple using the internal hazard info
    fitted_named = _flat_to_named(fitted_params, model_fitted.hazards)
    set_parameters!(model_fitted, fitted_named)
    
    Random.seed!(RNG_SEED + 1000)
    trajectories_fitted = simulate(model_fitted; paths=true, data=false, nsim=1)
    paths_fitted = trajectories_fitted[1]
    
    # Compare prevalence
    prev_true = compute_state_prevalence(paths_true, eval_times, n_states)
    prev_fitted = compute_state_prevalence(paths_fitted, eval_times, n_states)
    
    max_diff = maximum(abs.(prev_true .- prev_fitted))
    return max_diff < max_prev_diff
end

"""
Helper to convert flat parameters back to named tuple for progressive model.
Uses model.hazards (SemiMarkovHazard) which has npar_total field.
"""
function _flat_to_named(flat_params, hazards)
    idx = 1
    params = Dict{Symbol, Vector{Float64}}()
    for haz in hazards
        npar = haz.npar_total
        params[haz.hazname] = flat_params[idx:idx+npar-1]
        idx += npar
    end
    return NamedTuple(params)
end

# ============================================================================
# TEST SECTION 1: EXPONENTIAL HAZARDS
# ============================================================================

@testset "Exponential - No Covariates" begin
    Random.seed!(RNG_SEED)
    
    # True parameters (log scale) - Progressive model: 1→2→3
    true_rate_12 = 0.25  # Healthy → Ill
    true_rate_23 = 0.15  # Ill → Dead
    
    true_params = (
        h12 = [log(true_rate_12)],
        h23 = [log(true_rate_23)]
    )
    
    # Create progressive model (no direct 1→3)
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    # Generate and fit
    exact_data = generate_exact_data_progressive((h12, h23), true_params)
    model_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "Parameter recovery" begin
        fitted_natural = get_parameters(fitted; scale=:natural)
        @test isapprox(fitted_natural.h12[1], true_rate_12; rtol=PARAM_TOL_REL)
        @test isapprox(fitted_natural.h23[1], true_rate_23; rtol=PARAM_TOL_REL)
    end
    
    @testset "Distributional fidelity" begin
        @test check_distributional_fidelity((h12, h23), true_params, get_parameters_flat(fitted))
    end
end

@testset "Exponential - With Covariate" begin
    Random.seed!(RNG_SEED + 1)
    
    # True parameters: rate * exp(beta * x)
    true_rate_12 = 0.25
    true_beta_12 = 0.5
    true_rate_23 = 0.15
    true_beta_23 = -0.3
    
    # Covariate data
    cov_data = DataFrame(x = randn(N_SUBJECTS))
    
    true_params = (
        h12 = [log(true_rate_12), true_beta_12],
        h23 = [log(true_rate_23), true_beta_23]
    )
    
    h12 = Hazard(@formula(0 ~ x), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ x), "exp", 2, 3)
    
    exact_data = generate_exact_data_progressive((h12, h23), true_params; covariate_data=cov_data)
    model_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "Parameter recovery" begin
        p_nat = get_parameters(fitted; scale=:natural)
        @test isapprox(p_nat.h12[1], true_rate_12; rtol=PARAM_TOL_REL)
        @test isapprox(p_nat.h12[2], true_beta_12; atol=0.15)
        @test isapprox(p_nat.h23[1], true_rate_23; rtol=PARAM_TOL_REL)
        @test isapprox(p_nat.h23[2], true_beta_23; atol=0.15)
    end
end

# ============================================================================
# TEST SECTION 2: WEIBULL HAZARDS
# ============================================================================

@testset "Weibull - No Covariates" begin
    Random.seed!(RNG_SEED + 10)
    
    # True parameters: h(t) = shape * scale * t^(shape-1)
    # Parameter order: (shape, scale)
    true_shape_12, true_scale_12 = 1.3, 0.20
    true_shape_23, true_scale_23 = 0.9, 0.15
    
    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12)],
        h23 = [log(true_shape_23), log(true_scale_23)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    
    exact_data = generate_exact_data_progressive((h12, h23), true_params)
    model_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "Parameter recovery" begin
        p = get_parameters(fitted; scale=:natural)
        @test isapprox(p.h12[1], true_shape_12; rtol=PARAM_TOL_REL)
        @test isapprox(p.h12[2], true_scale_12; rtol=PARAM_TOL_REL)
        @test isapprox(p.h23[1], true_shape_23; rtol=PARAM_TOL_REL)
        @test isapprox(p.h23[2], true_scale_23; rtol=PARAM_TOL_REL)
    end
    
    @testset "Distributional fidelity" begin
        @test check_distributional_fidelity((h12, h23), true_params, get_parameters_flat(fitted))
    end
end

@testset "Weibull - With Covariate" begin
    Random.seed!(RNG_SEED + 11)
    
    true_shape_12, true_scale_12, true_beta_12 = 1.3, 0.20, 0.4
    true_shape_23, true_scale_23, true_beta_23 = 1.0, 0.15, -0.3
    
    cov_data = DataFrame(x = randn(N_SUBJECTS))
    
    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12), true_beta_12],
        h23 = [log(true_shape_23), log(true_scale_23), true_beta_23]
    )
    
    h12 = Hazard(@formula(0 ~ x), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ x), "wei", 2, 3)
    
    exact_data = generate_exact_data_progressive((h12, h23), true_params; covariate_data=cov_data)
    model_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "Parameter recovery" begin
        p = get_parameters(fitted; scale=:estimation)
        @test isapprox(exp(p[1]), true_shape_12; rtol=PARAM_TOL_REL)
        @test isapprox(exp(p[2]), true_scale_12; rtol=PARAM_TOL_REL)
        @test isapprox(p[3], true_beta_12; atol=0.15)
    end
end

# ============================================================================
# TEST SECTION 3: GOMPERTZ HAZARDS
# ============================================================================

@testset "Gompertz - No Covariates" begin
    Random.seed!(RNG_SEED + 20)
    
    # Gompertz: h(t) = rate * exp(shape * t)
    # Parameter order: (shape, rate)
    # NOTE: shape is unconstrained (identity transform), rate is log-transformed
    # Use higher rates for more events in the observation window
    true_shape_12, true_rate_12 = 0.08, 0.10
    true_shape_23, true_rate_23 = 0.06, 0.08
    
    # Shape is on natural scale (identity transform), rate is log-transformed
    true_params = (
        h12 = [true_shape_12, log(true_rate_12)],
        h23 = [true_shape_23, log(true_rate_23)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "gom", 2, 3)
    
    # Use n=2000 for Gompertz to get enough 2→3 events
    exact_data = generate_exact_data_progressive((h12, h23), true_params; 
        n_subj=2000, max_time=30.0)
    model_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "Parameter recovery" begin
        p = get_parameters(fitted; scale=:estimation)
        # shape is on natural scale (identity transform) - compare directly
        @test isapprox(p[1], true_shape_12; atol=0.03)
        # rate is log-transformed - use exp()
        @test isapprox(exp(p[2]), true_rate_12; rtol=PARAM_TOL_REL)
        # h23 parameters have higher variance (fewer 2→3 events), use higher tolerance
        @test isapprox(p[3], true_shape_23; atol=0.03)
        @test isapprox(exp(p[4]), true_rate_23; rtol=0.25)
    end
    
    @testset "Distributional fidelity" begin
        @test check_distributional_fidelity((h12, h23), true_params, get_parameters_flat(fitted); max_time=30.0)
    end
end

@testset "Gompertz - With Covariate" begin
    Random.seed!(RNG_SEED + 21)
    
    # Parameter order: (shape, rate, beta)
    # NOTE: shape is unconstrained (identity transform), rate is log-transformed
    # Use higher rates for more events
    true_shape_12, true_rate_12, true_beta_12 = 0.08, 0.10, 0.3
    true_shape_23, true_rate_23, true_beta_23 = 0.06, 0.08, -0.2
    
    cov_data = DataFrame(x = randn(2000))
    
    # Shape is on natural scale (identity transform), rate is log-transformed
    true_params = (
        h12 = [true_shape_12, log(true_rate_12), true_beta_12],
        h23 = [true_shape_23, log(true_rate_23), true_beta_23]
    )
    
    h12 = Hazard(@formula(0 ~ x), "gom", 1, 2)
    h23 = Hazard(@formula(0 ~ x), "gom", 2, 3)
    
    exact_data = generate_exact_data_progressive((h12, h23), true_params; 
        covariate_data=cov_data, max_time=30.0)
    model_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "Parameter recovery" begin
        p = get_parameters(fitted; scale=:estimation)
        # shape is on natural scale (identity transform) - compare directly
        @test isapprox(p[1], true_shape_12; atol=0.03)
        # rate is log-transformed - use exp()
        @test isapprox(exp(p[2]), true_rate_12; rtol=PARAM_TOL_REL)
        @test isapprox(p[3], true_beta_12; atol=0.15)
    end
end

# ============================================================================
# TEST SECTION 4: SPLINE HAZARDS
# ============================================================================

@testset "Spline - No Covariates" begin
    Random.seed!(RNG_SEED + 30)
    
    # Generate data from Weibull, fit with splines
    # This tests that splines can approximate known parametric forms
    true_shape_12, true_scale_12 = 1.2, 0.20
    true_shape_23, true_scale_23 = 1.0, 0.15
    
    h12_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23_wei = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    
    wei_params = (
        h12 = [log(true_shape_12), log(true_scale_12)],
        h23 = [log(true_shape_23), log(true_scale_23)]
    )
    
    exact_data = generate_exact_data_progressive((h12_wei, h23_wei), wei_params)
    
    # Fit with splines
    h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3, knots=[5.0, 10.0, 15.0], 
                    boundaryknots=[0.0, MAX_TIME], extrapolation="constant")
    h23_sp = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree=3, knots=[5.0, 10.0, 15.0],
                    boundaryknots=[0.0, MAX_TIME], extrapolation="constant")
    
    model_fit = multistatemodel(h12_sp, h23_sp; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "Spline fit converges" begin
        @test isfinite(fitted.loglik.loglik)
        @test fitted.loglik.loglik < 0
    end
    
    @testset "Hazard values reasonable" begin
        # Spline hazards should be positive at evaluation points
        # Get fitted parameters for each hazard (named tuple)
        p = get_parameters(fitted; scale=:natural)
        h12_pars = p.h12
        h23_pars = p.h23
        # Use empty covariates since model has no covariates
        empty_covars = NamedTuple()
        h12_vals = [fitted.hazards[1].hazard_fn(t, h12_pars, empty_covars) for t in 1.0:5.0:15.0]
        h23_vals = [fitted.hazards[2].hazard_fn(t, h23_pars, empty_covars) for t in 1.0:5.0:15.0]
        @test all(h12_vals .> 0)
        @test all(h23_vals .> 0)
    end
end

@testset "Spline - With Covariate" begin
    Random.seed!(RNG_SEED + 31)
    
    # Generate data from Weibull with covariate effect
    true_shape_12, true_scale_12, true_beta_12 = 1.2, 0.20, 0.5
    true_shape_23, true_scale_23, true_beta_23 = 1.0, 0.15, -0.3
    
    cov_data = DataFrame(x = randn(N_SUBJECTS))
    
    h12_wei = Hazard(@formula(0 ~ x), "wei", 1, 2)
    h23_wei = Hazard(@formula(0 ~ x), "wei", 2, 3)
    
    wei_params = (
        h12 = [log(true_shape_12), log(true_scale_12), true_beta_12],
        h23 = [log(true_shape_23), log(true_scale_23), true_beta_23]
    )
    
    exact_data = generate_exact_data_progressive((h12_wei, h23_wei), wei_params; covariate_data=cov_data)
    
    # Fit with splines + covariate
    h12_sp = Hazard(@formula(0 ~ x), "sp", 1, 2; degree=3, knots=[5.0, 10.0, 15.0],
                    boundaryknots=[0.0, MAX_TIME], extrapolation="constant")
    h23_sp = Hazard(@formula(0 ~ x), "sp", 2, 3; degree=3, knots=[5.0, 10.0, 15.0],
                    boundaryknots=[0.0, MAX_TIME], extrapolation="constant")
    
    model_fit = multistatemodel(h12_sp, h23_sp; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "Spline with covariate fit converges" begin
        @test isfinite(fitted.loglik.loglik)
    end
    
    @testset "Covariate effect reasonable" begin
        # Last parameter in each hazard should be covariate effect
        p = get_parameters_flat(fitted)
        # h12 has ~6 spline coeffs + 1 beta, h23 has ~6 spline coeffs + 1 beta
        n_h12 = fitted.hazards[1].npar_total
        beta_12_est = p[n_h12]  # Last param of h12
        beta_23_est = p[end]    # Last param of h23
        
        # Covariate effects should have same sign as truth
        @test sign(beta_12_est) == sign(true_beta_12)
        @test sign(beta_23_est) == sign(true_beta_23)
    end
end

# ============================================================================
# TEST SECTION 5: TIME-VARYING COVARIATES
# ============================================================================

@testset "Exponential - TVC" begin
    Random.seed!(RNG_SEED + 40)
    
    n_subj = N_SUBJECTS
    
    # Create TVC data: x changes at time 5
    tvc_data = vcat([
        DataFrame(
            id = fill(i, 2),
            tstart = [0.0, 5.0],
            tstop = [5.0, MAX_TIME],
            statefrom = [1, 1],
            stateto = [1, 1],
            obstype = [1, 1],
            x = [0.0, 1.0]
        ) for i in 1:n_subj
    ]...)
    
    true_rate_12 = 0.20
    true_beta_12 = 0.5
    true_rate_23 = 0.15
    true_beta_23 = -0.3
    
    true_params = (
        h12 = [log(true_rate_12), true_beta_12],
        h23 = [log(true_rate_23), true_beta_23]
    )
    
    h12 = Hazard(@formula(0 ~ x), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ x), "exp", 2, 3)
    
    model_sim = multistatemodel(h12, h23; data=tvc_data)
    set_parameters!(model_sim, true_params)
    
    # Use autotmax=false to preserve TVC interval structure
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    exact_data = sim_result[1]
    
    model_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "TVC parameter recovery" begin
        p_nat = get_parameters(fitted; scale=:natural)
        @test isapprox(p_nat.h12[1], true_rate_12; rtol=PARAM_TOL_REL)
        # TVC beta may have higher variance, use relaxed tolerance
        @test isapprox(p_nat.h12[2], true_beta_12; atol=0.25)
    end
end

@testset "Weibull - TVC" begin
    Random.seed!(RNG_SEED + 41)
    
    n_subj = N_SUBJECTS
    
    # TVC data
    tvc_data = vcat([
        DataFrame(
            id = fill(i, 2),
            tstart = [0.0, 5.0],
            tstop = [5.0, MAX_TIME],
            statefrom = [1, 1],
            stateto = [1, 1],
            obstype = [1, 1],
            x = [0.0, 1.0]
        ) for i in 1:n_subj
    ]...)
    
    true_shape_12, true_scale_12, true_beta_12 = 1.2, 0.20, 0.4
    true_shape_23, true_scale_23, true_beta_23 = 1.0, 0.15, -0.3
    
    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12), true_beta_12],
        h23 = [log(true_shape_23), log(true_scale_23), true_beta_23]
    )
    
    h12 = Hazard(@formula(0 ~ x), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ x), "wei", 2, 3)
    
    model_sim = multistatemodel(h12, h23; data=tvc_data)
    set_parameters!(model_sim, true_params)
    
    # Use autotmax=false to preserve TVC interval structure
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    exact_data = sim_result[1]
    
    model_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "TVC Weibull parameter recovery" begin
        p = get_parameters(fitted; scale=:natural)
        @test isapprox(p.h12[1], true_shape_12; rtol=PARAM_TOL_REL)
        @test isapprox(p.h12[2], true_scale_12; rtol=PARAM_TOL_REL)
    end
end

@testset "Gompertz - TVC" begin
    Random.seed!(RNG_SEED + 42)
    
    n_subj = N_SUBJECTS
    
    # TVC data: x changes at time 5
    tvc_data = vcat([
        DataFrame(
            id = fill(i, 2),
            tstart = [0.0, 5.0],
            tstop = [5.0, 30.0],  # Longer follow-up for Gompertz
            statefrom = [1, 1],
            stateto = [1, 1],
            obstype = [1, 1],
            x = [0.0, 1.0]
        ) for i in 1:n_subj
    ]...)
    
    # Gompertz parameters: (shape, rate, beta)
    # Shape is identity-transformed, rate is log-transformed
    true_shape_12, true_rate_12, true_beta_12 = 0.08, 0.10, 0.3
    true_shape_23, true_rate_23, true_beta_23 = 0.06, 0.08, -0.2
    
    true_params = (
        h12 = [true_shape_12, log(true_rate_12), true_beta_12],
        h23 = [true_shape_23, log(true_rate_23), true_beta_23]
    )
    
    h12 = Hazard(@formula(0 ~ x), "gom", 1, 2)
    h23 = Hazard(@formula(0 ~ x), "gom", 2, 3)
    
    model_sim = multistatemodel(h12, h23; data=tvc_data)
    set_parameters!(model_sim, true_params)
    
    # Use autotmax=false to preserve TVC interval structure
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    exact_data = sim_result[1]
    
    model_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "TVC Gompertz parameter recovery" begin
        p = get_parameters(fitted; scale=:estimation)
        # Shape is on natural scale (identity transform)
        @test isapprox(p[1], true_shape_12; atol=0.05)
        # Rate is log-transformed
        @test isapprox(exp(p[2]), true_rate_12; rtol=PARAM_TOL_REL)
        # TVC beta may have higher variance
        @test isapprox(p[3], true_beta_12; atol=0.25)
    end
end

@testset "Spline - TVC" begin
    Random.seed!(RNG_SEED + 43)
    
    n_subj = N_SUBJECTS
    
    # TVC data: x changes at time 5
    tvc_data = vcat([
        DataFrame(
            id = fill(i, 2),
            tstart = [0.0, 5.0],
            tstop = [5.0, MAX_TIME],
            statefrom = [1, 1],
            stateto = [1, 1],
            obstype = [1, 1],
            x = [0.0, 1.0]
        ) for i in 1:n_subj
    ]...)
    
    # Generate underlying data from Weibull with TVC effect
    true_shape_12, true_scale_12, true_beta_12 = 1.2, 0.20, 0.5
    true_shape_23, true_scale_23, true_beta_23 = 1.0, 0.15, -0.3
    
    h12_wei = Hazard(@formula(0 ~ x), "wei", 1, 2)
    h23_wei = Hazard(@formula(0 ~ x), "wei", 2, 3)
    
    wei_params = (
        h12 = [log(true_shape_12), log(true_scale_12), true_beta_12],
        h23 = [log(true_shape_23), log(true_scale_23), true_beta_23]
    )
    
    model_sim = multistatemodel(h12_wei, h23_wei; data=tvc_data)
    set_parameters!(model_sim, wei_params)
    
    # Use autotmax=false to preserve TVC interval structure
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    exact_data = sim_result[1]
    
    # Fit with splines + TVC covariate
    h12_sp = Hazard(@formula(0 ~ x), "sp", 1, 2; degree=3, knots=[5.0, 10.0, 15.0],
                    boundaryknots=[0.0, MAX_TIME], extrapolation="constant")
    h23_sp = Hazard(@formula(0 ~ x), "sp", 2, 3; degree=3, knots=[5.0, 10.0, 15.0],
                    boundaryknots=[0.0, MAX_TIME], extrapolation="constant")
    
    model_fit = multistatemodel(h12_sp, h23_sp; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    @testset "TVC Spline fit converges" begin
        @test isfinite(fitted.loglik.loglik)
        @test fitted.loglik.loglik < 0
    end
    
    @testset "TVC covariate effect reasonable" begin
        # Last parameter in each hazard should be the TVC covariate effect
        p = get_parameters_flat(fitted)
        n_h12 = fitted.hazards[1].npar_total
        beta_12_est = p[n_h12]  # Last param of h12
        beta_23_est = p[end]    # Last param of h23
        
        # Covariate effects should have same sign as truth
        @test sign(beta_12_est) == sign(true_beta_12)
        @test sign(beta_23_est) == sign(true_beta_23)
    end
end

# ============================================================================
# TEST SECTION 6: EDGE CASES
# ============================================================================

@testset "Subject Weights" begin
    Random.seed!(RNG_SEED + 50)
    
    # Test that subject weights work correctly
    true_rate_12 = 0.25
    true_rate_23 = 0.15
    
    true_params = (
        h12 = [log(true_rate_12)],
        h23 = [log(true_rate_23)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    exact_data = generate_exact_data_progressive((h12, h23), true_params; n_subj=500)
    
    # Fit with uniform weights
    model_fit = multistatemodel(h12, h23; data=exact_data)
    fitted_unweighted = fit(model_fit; verbose=false)
    
    # Fit with double weights (should give same estimates)
    exact_data.subjwt = fill(2.0, nrow(exact_data))
    model_fit_wt = multistatemodel(h12, h23; data=exact_data)
    fitted_weighted = fit(model_fit_wt; verbose=false)
    
    @testset "Weighted vs unweighted estimates" begin
        p_uw = get_parameters_flat(fitted_unweighted)
        p_w = get_parameters_flat(fitted_weighted)
        # Point estimates should be identical
        @test isapprox(p_uw, p_w; rtol=1e-6)
    end
end

@testset "Log-likelihood Properties" begin
    Random.seed!(RNG_SEED + 51)
    
    # Simple model to verify basic likelihood properties
    data = DataFrame(
        id = repeat(1:100, inner=1),
        tstart = zeros(100),
        tstop = fill(5.0, 100),
        statefrom = ones(Int, 100),
        stateto = vcat(fill(2, 70), fill(1, 30)),  # 70 events, 30 censored
        obstype = ones(Int, 100)
    )
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    model = multistatemodel(h12; data=data)
    fitted = fit(model; verbose=false)
    
    @testset "Likelihood is finite and negative" begin
        @test isfinite(fitted.loglik.loglik)
        @test fitted.loglik.loglik < 0
    end
    
    @testset "Variance-covariance is positive definite" begin
        if !isnothing(fitted.vcov)
            eigvals = LinearAlgebra.eigvals(fitted.vcov)
            @test all(eigvals .> 0)
        end
    end
end

# ============================================================================
# Summary
# ============================================================================

println("\n=== Exact Data Long Test Suite Complete ===\n")
println("This test suite validated:")
println("  - Exponential hazards: no covariate, with covariate, TVC")
println("  - Weibull hazards: no covariate, with covariate, TVC")
println("  - Gompertz hazards: no covariate, with covariate, TVC")
println("  - Spline hazards: no covariate, with covariate, TVC")
println("  - Edge cases: subject weights, likelihood properties")
println("Model structure: progressive 3-state (1→2→3)")
println("Sample size: n=$(N_SUBJECTS), simulation trajectories: $(N_SIM_TRAJ)")
