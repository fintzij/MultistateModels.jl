# =============================================================================
# Hazard Evaluation Tests
# =============================================================================
#
# This test file validates that all hazard functions return correct values
# by comparing against analytically or numerically computed reference values.
#
# Coverage:
#   1. Parametric families: Exponential, Weibull, Gompertz
#   2. Spline hazards: B-spline basis with numerical integration verification
#   3. Linear predictor modes: Proportional Hazards (PH) vs Accelerated Failure Time (AFT)
#   4. Time transform optimization: Verifies parity with standard evaluation
#   5. Aggregate quantities: total_cumulhaz, survprob
#
# Analytical Reference Formulas:
# -----------------------------
# Exponential:
#   h(t) = λ                        (constant hazard)
#   H(t) = λt                       (cumulative hazard)
#   S(t) = exp(-λt)                 (survival function)
#
# Weibull (shape κ, scale λ):
#   h(t) = λκt^{κ-1}                (hazard increases/decreases with t)
#   H(t) = λt^κ                     (cumulative hazard)
#   S(t) = exp(-λt^κ)               (survival function)
#
# Gompertz (shape γ, scale λ):
#   h(t) = λγ exp(γt)               (exponentially increasing hazard)
#   H(t) = λ(exp(γt) - 1)           (cumulative hazard)
#   S(t) = exp(-λ(exp(γt) - 1))     (survival function)
#
# Covariate Effects:
# -----------------
# Proportional Hazards (PH):
#   h(t|x) = h₀(t) exp(β'x)         (hazard ratio = exp(β'x))
#   H(t|x) = H₀(t) exp(β'x)
#
# Accelerated Failure Time (AFT):
#   h(t|x) = h₀(t·exp(-β'x)) exp(-β'x)   (time is scaled)
#   H(t|x) = H₀(t·exp(-β'x))
#
# Note: ParameterHandling.positive() uses ε = sqrt(eps(Float64)) ≈ 1.49e-8 as
# a safety margin, which introduces small numerical differences when parameters
# go through the exp/log round-trip. Tests use rtol=1e-6 for robustness.
# =============================================================================

using .TestFixtures: toy_expwei_model, toy_gompertz_model
using BSplineKit  # For spline hazard verification

# Standard tolerance for tests that compare computed vs expected values
# after parameter round-trip through ParameterHandling
const PARAM_RTOL = 1e-6

# =============================================================================
# 1. SURVIVAL PROBABILITY TESTS
# =============================================================================
# survprob computes S(t₁, t₂) = exp(-∫_{t₁}^{t₂} h(u) du) = exp(-H(t₁, t₂))
# For exponential: S(0, t) = exp(-λt), so CDF F(t) = 1 - S(t) = 1 - exp(-λt)

@testset "survprob" begin
    # Test: Exponential survival probability matches Distributions.jl reference
    #
    # Setup: λ = 0.1 (rate), so mean = 1/λ = 10
    # But we set TOTAL hazard from state 1 as h12 + h13 = 0.1 + 0.1 = 0.2
    # So effective rate out of state 1 is 0.2, mean sojourn = 5
    #
    # Expected: F(2) = 1 - exp(-0.2 * 2) ≈ 0.32968
    
    fixture = toy_expwei_model()
    model = fixture.model
    
    # Set exponential rates: h12 has rate 0.1, h13 has rate 0.1 (no covariate effect)
    MultistateModels.set_parameters!(model, (h12 = [log(0.1),], h13 = [log(0.1), 0.0, 0.0, 0.0]))

    subjdat_row = model.data[1, :]
    params = MultistateModels.get_log_scale_params(model.parameters)
    
    # Compute cumulative incidence = 1 - S(0, 2)
    interval_incid = 1 - MultistateModels.survprob(0.0, 2.0, params, subjdat_row, 
                                                    model.totalhazards[1], model.hazards; 
                                                    give_log = false)
        
    # Reference: Distributions.jl Exponential uses mean parameterization (1/rate)
    # Total rate = 0.2, so mean = 5
    @test isapprox(cdf(Exponential(5), 2), interval_incid; rtol=PARAM_RTOL)
end

# =============================================================================
# 2. EXPONENTIAL HAZARD TESTS
# =============================================================================
# Exponential hazard: h(t) = λ (constant in time)
# With PH covariates: h(t|x) = λ exp(β'x) = exp(log(λ) + β'x)
# On log scale: log h(t|x) = log(λ) + β₁x₁ + β₂x₂ + ... 

@testset "test_hazards_exp" begin
    # -------------------------------------------------------------------------
    # Test 1: Exponential hazard without covariates
    # -------------------------------------------------------------------------
    # Formula: h(t) = exp(log_rate) = exp(0.8)
    # This is constant in t for exponential
    
    fixture = toy_expwei_model()
    model = fixture.model
    data = fixture.data

    # Parameters: h12 = [log_rate], h13 = [log_rate, β_trt, β_age, β_trt:age]
    MultistateModels.set_parameters!(model, (h12 = [0.8,], h13 = [0.3, 0.6, -0.4, 0.15]))

    @test isa(model.hazards[1], MultistateModels.MarkovHazard)
    subjdat_row = model.data[1, :]
    hazard1 = model.hazards[1]
    covars1 = MultistateModels.extract_covariates_fast(subjdat_row, hazard1.covar_names)
    
    # Verify: log h(t) = 0.8, h(t) = exp(0.8) for intercept-only model
    @test log(MultistateModels.eval_hazard(hazard1, 0.0, get_log_scale_params(model.parameters)[1], covars1)) ≈ 0.8
    @test MultistateModels.eval_hazard(hazard1, 0.0, get_log_scale_params(model.parameters)[1], covars1) ≈ exp(0.8)
    
    # Verify DataFrameRow interface gives identical results (zero-copy approach)
    @test MultistateModels.eval_hazard(hazard1, 0.0, get_log_scale_params(model.parameters)[1], subjdat_row) ≈ exp(0.8)

    # -------------------------------------------------------------------------
    # Test 2: Exponential hazard with covariates (PH model)
    # -------------------------------------------------------------------------
    # Formula: log h(t|x) = β₀ + β₁·trt + β₂·age + β₃·trt·age
    # where β₀ = 0.3, β₁ = 0.6, β₂ = -0.4, β₃ = 0.15
    
    pars = get_log_scale_params(model.parameters)[2]
    
    # Compute analytical log-hazard for each row
    trueval = [
        pars[1] + data.trt[1] * pars[2] + data.age[1] * pars[3] + data.trt[1] * data.age[1] * pars[4],
        pars[1] + data.trt[2] * pars[2] + data.age[2] * pars[3] + data.trt[2] * data.age[2] * pars[4],
        pars[1] + data.trt[3] * pars[2] + data.age[3] * pars[3] + data.trt[3] * data.age[3] * pars[4]
    ]
    
    hazard2 = model.hazards[2]
    for h in axes(model.data, 1)
        subjdat_row = model.data[h, :]
        covars2 = MultistateModels.extract_covariates_fast(subjdat_row, hazard2.covar_names)
        
        # Verify log hazard matches analytical formula
        @test log(MultistateModels.eval_hazard(hazard2, 0.0, get_log_scale_params(model.parameters)[2], covars2)) ≈ trueval[h]
        # Verify natural scale
        @test MultistateModels.eval_hazard(hazard2, 0.0, get_log_scale_params(model.parameters)[2], covars2) ≈ exp(trueval[h])
        # Verify DataFrameRow interface
        @test MultistateModels.eval_hazard(hazard2, 0.0, get_log_scale_params(model.parameters)[2], subjdat_row) ≈ exp(trueval[h])
    end
end

# =============================================================================
# 3. WEIBULL HAZARD TESTS
# =============================================================================
# Weibull hazard: h(t) = λκt^{κ-1} where κ = shape, λ = scale
# On log scale: log h(t) = log(λ) + log(κ) + (κ-1)·log(t)
# With PH covariates: log h(t|x) = log(λ) + log(κ) + (κ-1)·log(t) + β'x

@testset "test_hazards_weibull" begin
    # -------------------------------------------------------------------------
    # Test 1: Weibull hazard without covariates
    # -------------------------------------------------------------------------
    # Parameters: log(shape) = -0.25, log(scale) = 0.2
    # Formula: log h(t) = log(scale) + log(shape) + (shape - 1) * log(t)
    #        = 0.2 + (-0.25) + (exp(-0.25) - 1) * log(t)
    
    fixture = toy_expwei_model()
    model = fixture.model

    MultistateModels.set_parameters!(model, (h21 = [-0.25, 0.2],))

    subjdat_row = model.data[1, :]
    hazard3 = model.hazards[3]
    covars3 = MultistateModels.extract_covariates_fast(subjdat_row, hazard3.covar_names)
    pars = get_log_scale_params(model.parameters)[3]
    
    t_eval = 1.7
    log_shape = pars[1]
    log_scale = pars[2]
    shape = exp(log_shape)
    
    # Analytical log-hazard: log(λκt^{κ-1}) = log(λ) + log(κ) + (κ-1)log(t)
    log_hazard = log_scale + log_shape + (shape - 1) * log(t_eval)
    
    @test log(MultistateModels.eval_hazard(hazard3, t_eval, pars, covars3)) ≈ log_hazard
    @test MultistateModels.eval_hazard(hazard3, t_eval, pars, covars3) ≈ exp(log_hazard)

    # -------------------------------------------------------------------------
    # Test 2: Weibull hazard with covariates (PH model)
    # -------------------------------------------------------------------------
    # Parameters: log(shape) = 0.2, log(scale) = 0.25, β_trt = -0.3
    # Formula: log h(t|x) = log(κ) + (κ-1)log(t) + log(λ) + β_trt·trt
    # Note: expm1(x) = exp(x) - 1, so shape - 1 = expm1(log_shape)
    
    t = 1.0
    MultistateModels.set_parameters!(model, (h23 = [0.2, 0.25, -0.3],))
    pars_h23 = get_log_scale_params(model.parameters)[4]

    hazard4 = model.hazards[4]
    for h in axes(model.data, 1)
        subjdat_row = model.data[h, :]
        covars4 = MultistateModels.extract_covariates_fast(subjdat_row, hazard4.covar_names)
        trt_val = subjdat_row.trt
        
        # Analytical: log h(t|x) = log(κ) + (κ-1)log(t) + log(λ) + β·trt
        # At t=1, log(t)=0, so: log h(1|x) = log(κ) + log(λ) + β·trt
        # Using expm1(log_shape) = shape - 1
        trueval = pars_h23[1] + expm1(pars_h23[1]) * log(t) + pars_h23[2] + pars_h23[3] * trt_val

        @test log(MultistateModels.eval_hazard(hazard4, t, pars_h23, covars4)) ≈ trueval
        @test MultistateModels.eval_hazard(hazard4, t, pars_h23, covars4) ≈ exp(trueval)
    end
end

# =============================================================================
# 4. EXPONENTIAL CUMULATIVE HAZARD TESTS
# =============================================================================
# Exponential cumulative hazard: H(s,t) = ∫_s^t λ du = λ(t - s)
# On log scale: log H(s,t) = log(λ) + log(t - s)
# With PH covariates: log H(s,t|x) = log(λ) + log(t - s) + β'x

@testset "test_cumulativehazards_exp" begin
    fixture = toy_expwei_model()
    model = fixture.model
    data = fixture.data

    MultistateModels.set_parameters!(model, (h12 = [0.8,], h13 = [0.0, 0.6, -0.4, 0.15]))
    lb = 0
    ub = 5

    # -------------------------------------------------------------------------
    # Test 1: Cumulative hazard without covariates
    # -------------------------------------------------------------------------
    # Formula: log H(0,5) = log(rate) + log(5-0) = 0.8 + log(5)
    
    subjdat_row = model.data[1, :]
    hazard1 = model.hazards[1]
    covars1 = MultistateModels.extract_covariates_fast(subjdat_row, hazard1.covar_names)
    
    @test log(MultistateModels.eval_cumhaz(hazard1, lb, ub, get_log_scale_params(model.parameters)[1], covars1)) ≈ 0.8 + log(ub-lb)

    # -------------------------------------------------------------------------
    # Test 2: Cumulative hazard with covariates (PH model)
    # -------------------------------------------------------------------------
    # Formula: log H(s,t|x) = log(λ) + β'x + log(t-s)
    #        = (log(λ) + β'x) + log(interval_length)
    #        = log_hazard + log(ub - lb)
    
    pars = get_log_scale_params(model.parameters)[2] 
    
    # Compute log-hazard for each row
    log_haz = [
        pars[1] + pars[2]*data.trt[1] + pars[3]*data.age[1] + pars[4]*data.trt[1]*data.age[1],
        pars[1] + pars[2]*data.trt[2] + pars[3]*data.age[2] + pars[4]*data.trt[2]*data.age[2],
        pars[1] + pars[2]*data.trt[3] + pars[3]*data.age[3] + pars[4]*data.trt[3]*data.age[3]
    ]
    
    # Log cumulative hazard = log hazard + log(interval)
    trueval = log_haz .+ log(ub-lb)

    hazard2 = model.hazards[2]
    for h in axes(model.data, 1)
        subjdat_row = model.data[h, :]
        covars2 = MultistateModels.extract_covariates_fast(subjdat_row, hazard2.covar_names)
        
        @test log(MultistateModels.eval_cumhaz(hazard2, lb, ub, get_log_scale_params(model.parameters)[2], covars2)) ≈ trueval[h]
        @test MultistateModels.eval_cumhaz(hazard2, lb, ub, get_log_scale_params(model.parameters)[2], covars2) ≈ exp(trueval[h])
    end
end

# =============================================================================
# 5. WEIBULL CUMULATIVE HAZARD TESTS
# =============================================================================
# Weibull cumulative hazard: H(s,t) = λ(t^κ - s^κ)
# On log scale: log H(s,t) = log(λ) + log(t^κ - s^κ)
# With PH covariates: log H(s,t|x) = log(λ) + log(t^κ - s^κ) + β'x

@testset "test_cumulativehazards_weibull" begin
    fixture = toy_expwei_model()
    model = fixture.model
    MultistateModels.set_parameters!(model, (h21 = [-0.25, 0.2], h23 = [0.2, 0.25, -0.3]))
    lb = 0
    ub = 5

    # -------------------------------------------------------------------------
    # Test 1: Cumulative hazard without covariates
    # -------------------------------------------------------------------------
    # Formula: log H(0,5) = log(λ) + log(5^κ - 0^κ) = log(λ) + κ·log(5)
    # (since 0^κ = 0 for κ > 0)
    
    subjdat_row = model.data[1, :]
    hazard3 = model.hazards[3]
    covars3 = MultistateModels.extract_covariates_fast(subjdat_row, hazard3.covar_names)
    
    # Analytical: log H(0,5) = log(scale) + log(5^shape - 0^shape)
    pars_h21 = get_log_scale_params(model.parameters)[3]
    shape = exp(pars_h21[1])
    trueval_h21 = log(ub^shape - lb^shape) + pars_h21[2]
    
    @test log(MultistateModels.eval_cumhaz(hazard3, lb, ub, pars_h21, covars3)) ≈ trueval_h21
    @test MultistateModels.eval_cumhaz(hazard3, lb, ub, pars_h21, covars3) ≈ exp(trueval_h21)

    # -------------------------------------------------------------------------
    # Test 2: Cumulative hazard with covariates (PH model)
    # -------------------------------------------------------------------------
    # Formula: log H(s,t|x) = log(λ) + log(t^κ - s^κ) + β·trt
    
    pars = get_log_scale_params(model.parameters)[4] 
    
    hazard4 = model.hazards[4]
    for h in axes(model.data, 1)
        subjdat_row = model.data[h, :]
        covars4 = MultistateModels.extract_covariates_fast(subjdat_row, hazard4.covar_names)
        trt_val = subjdat_row.trt
        
        shape = exp(pars[1])
        trueval = log(ub^shape - lb^shape) + pars[2] + pars[3] * trt_val
                    
        @test log(MultistateModels.eval_cumhaz(hazard4, lb, ub, get_log_scale_params(model.parameters)[4], covars4)) ≈ trueval
        @test MultistateModels.eval_cumhaz(hazard4, lb, ub, get_log_scale_params(model.parameters)[4], covars4) ≈ exp(trueval)
    end
end

# =============================================================================
# 6. TOTAL CUMULATIVE HAZARD TESTS (COMPETING RISKS)
# =============================================================================
# For competing risks from state s, total cumulative hazard is:
#   H_total(s,t) = Σ_j H_{s→j}(s,t)
# where j ranges over all states reachable from s.
# This is used to compute survival in state s: S_s(t) = exp(-H_total(0,t))

@testset "test_totalcumulativehazards" begin
    fixture = toy_expwei_model()
    model = fixture.model
    
    # Set all parameters for the 4-transition model
    MultistateModels.set_parameters!(model,
                                    (h12 = [0.8,],           # exp rate
                                     h13 = [0.8, 0.6, -0.4, 0.15],  # exp + covariates
                                     h21 = [0.8, 1.2],       # weibull (shape, scale)
                                     h23 = [0.8, 0.25, 1.2]))  # weibull + covariate
    lb = 0
    ub = 5
    window = ub - lb

    # Extract all parameters
    pars_h12 = get_log_scale_params(model.parameters)[1]
    pars_h13 = get_log_scale_params(model.parameters)[2]
    pars_h21 = get_log_scale_params(model.parameters)[3]
    pars_h23 = get_log_scale_params(model.parameters)[4]

    # -------------------------------------------------------------------------
    # Compute analytical cumulative hazards for each transition
    # -------------------------------------------------------------------------
    
    # H12: exponential, no covariates
    # H(0,5) = λ * 5 = exp(0.8) * 5
    cum12 = exp(pars_h12[1]) * window
    
    # H21: Weibull, no covariates
    # H(0,5) = λ * (5^κ - 0^κ)
    shape21 = exp(pars_h21[1])
    scale21 = exp(pars_h21[2])
    cum21 = scale21 * (ub^shape21 - lb^shape21)
    
    # H23: Weibull with covariates (will be computed per-subject)
    shape23 = exp(pars_h23[1])
    scale23 = exp(pars_h23[2])
    interval_term23 = scale23 * (ub^shape23 - lb^shape23)

    # -------------------------------------------------------------------------
    # Verify total cumulative hazard for each origin state
    # -------------------------------------------------------------------------
    for row_idx in axes(model.data, 1)
        subjdat_row = model.data[row_idx, :]
        
        # H13: exponential with covariates
        # H(0,5|x) = exp(log(λ) + β'x) * 5
        linpred13 = pars_h13[1] + pars_h13[2] * subjdat_row.trt + pars_h13[3] * subjdat_row.age + pars_h13[4] * subjdat_row.trt * subjdat_row.age
        cum13 = exp(linpred13) * window
        
        # H23: Weibull with covariates
        linpred23 = pars_h23[3] * subjdat_row.trt
        cum23 = interval_term23 * exp(linpred23)

        # Check total hazard for each origin state
        for s in axes(model.tmat, 1) 
            # State 1: can transition to 2 or 3
            # State 2: can transition to 1 or 3
            # State 3: absorbing (no transitions out)
            expected_total = if s == 1
                cum12 + cum13  # H_{1→2} + H_{1→3}
            elseif s == 2
                cum21 + cum23  # H_{2→1} + H_{2→3}
            else
                0.0  # absorbing state
            end

            params = get_log_scale_params(model.parameters)
            result = MultistateModels.total_cumulhaz(lb, ub, params, subjdat_row, model.totalhazards[s], model.hazards; give_log = false)
            @test result ≈ expected_total

            # Test log scale
            log_result = MultistateModels.total_cumulhaz(lb, ub, params, subjdat_row, model.totalhazards[s], model.hazards; give_log = true)
            if expected_total == 0.0
                @test log_result == -Inf
            else
                @test log_result ≈ log(expected_total)
            end
        end
    end
end

# =============================================================================
# 7. GOMPERTZ HAZARD TESTS
# =============================================================================
# Gompertz hazard: h(t) = λγ exp(γt) where γ = shape, λ = scale
# On log scale: log h(t) = log(λ) + log(γ) + γt
# This models exponentially increasing hazard (common in aging/mortality)

@testset "test_hazards_gompertz" begin
    # -------------------------------------------------------------------------
    # Test 1: Gompertz hazard without covariates
    # -------------------------------------------------------------------------
    # Parameters: log(shape) = log(1.5), log(scale) = log(0.5)
    # Formula: log h(t) = log(scale) + log(shape) + shape * t
    #        = log(0.5) + log(1.5) + 1.5 * t
    
    fixture = toy_gompertz_model()
    model = fixture.model
    MultistateModels.set_parameters!(model, (h12 = [log(1.5), log(0.5)], 
                                             h13 = [log(0.5), log(0.5), 1.5], 
                                             h23 = [log(1), log(2/3)]))    

    pars_h12 = get_log_scale_params(model.parameters)[1]
    pars_h13 = get_log_scale_params(model.parameters)[2]
    
    subjdat_row = model.data[1, :]
    hazard1 = model.hazards[1]
    covars1 = MultistateModels.extract_covariates_fast(subjdat_row, hazard1.covar_names)
    
    # Analytical: log h(1) = log(λ) + log(γ) + γ * 1
    #           = log_scale + log_shape + exp(log_shape) * 1
    expected_log_haz = pars_h12[2] + pars_h12[1] + exp(pars_h12[1]) * 1.0
    
    @test log(MultistateModels.eval_hazard(hazard1, 1.0, pars_h12, covars1)) ≈ expected_log_haz
    @test MultistateModels.eval_hazard(hazard1, 1.0, pars_h12, covars1) ≈ exp(expected_log_haz)

    # -------------------------------------------------------------------------
    # Test 2: Gompertz hazard with covariates (PH model)
    # -------------------------------------------------------------------------
    # Parameters: h13 = [log(shape), log(scale), β_trt]
    # Formula: log h(t|x) = log(λ) + log(γ) + γt + β·trt
   
   subjdat_row2 = model.data[2, :]
   hazard2 = model.hazards[2]
   covars2 = MultistateModels.extract_covariates_fast(subjdat_row2, hazard2.covar_names)
   # For h13 with trt=1: log h(t) = log_scale + beta_trt + log_shape + shape * t
   # pars_h13 = [log_shape, log_scale, beta_trt]
   expected_log_haz_tvc = pars_h13[2] + pars_h13[1] + exp(pars_h13[1]) * 1.0 + pars_h13[3]
   @test log(MultistateModels.eval_hazard(hazard2, 1.0, pars_h13, covars2)) ≈ expected_log_haz_tvc

    @test MultistateModels.eval_hazard(hazard2, 1.0, pars_h13, covars2) ≈ exp(expected_log_haz_tvc)
end

# =============================================================================
# 8. GOMPERTZ CUMULATIVE HAZARD TESTS
# =============================================================================
# Gompertz cumulative hazard: H(s,t) = λ(exp(γt) - exp(γs))
# Note: This differs from (λ/γ)(exp(γt) - exp(γs)) in some parameterizations
# On log scale: log H(s,t) = log(λ) + log(exp(γt) - exp(γs))

@testset "test_cumulativehazards_gompertz" begin
    fixture = toy_gompertz_model()
    model = fixture.model
    MultistateModels.set_parameters!(model, (h12 = [log(1.5), log(0.5)], 
                                             h13 = [log(0.5), log(0.5), 1.5], 
                                             h23 = [log(1), log(2/3)]))
    lb = 0.0
    ub = 5.0

    pars_h12 = get_log_scale_params(model.parameters)[1]
    pars_h13 = get_log_scale_params(model.parameters)[2]
    
    # -------------------------------------------------------------------------
    # Test 1: Cumulative hazard without covariates
    # -------------------------------------------------------------------------
    # Formula: log H(0,5) = log(λ) + log(exp(γ*5) - exp(γ*0))
    #        = log_scale + log(exp(shape*5) - 1)
    
    subjdat_row = model.data[1, :]
    hazard1 = model.hazards[1]
    covars1 = MultistateModels.extract_covariates_fast(subjdat_row, hazard1.covar_names)
    
    shape_h12 = exp(pars_h12[1])
    expected_log_cumhaz = pars_h12[2] + log(exp(shape_h12 * ub) - exp(shape_h12 * lb))
    
    @test log(MultistateModels.eval_cumhaz(hazard1, lb, ub, pars_h12, covars1)) ≈ expected_log_cumhaz
    @test MultistateModels.eval_cumhaz(hazard1, lb, ub, pars_h12, covars1) ≈ exp(expected_log_cumhaz)

    # -------------------------------------------------------------------------
    # Test 2: Cumulative hazard with covariates (PH model)
    # -------------------------------------------------------------------------
    # Formula: log H(s,t|x) = log(λ) + β·trt + log(exp(γt) - exp(γs))
    
    subjdat_row2 = model.data[2, :]
    hazard2 = model.hazards[2]
    covars2 = MultistateModels.extract_covariates_fast(subjdat_row2, hazard2.covar_names)
    
    shape_h13 = exp(pars_h13[1])
    expected_log_cumhaz_tvc = pars_h13[2] + pars_h13[3] + log(exp(shape_h13 * ub) - exp(shape_h13 * lb))
    
    @test log(MultistateModels.eval_cumhaz(hazard2, lb, ub, pars_h13, covars2)) ≈ expected_log_cumhaz_tvc
    @test MultistateModels.eval_cumhaz(hazard2, lb, ub, pars_h13, covars2) ≈ exp(expected_log_cumhaz_tvc)
end

# =============================================================================
# 9. LINEAR PREDICTOR EFFECT MODE TESTS (PH vs AFT)
# =============================================================================
# Proportional Hazards (PH): h(t|x) = h₀(t) exp(β'x)
#   - Hazard ratio is constant over time: HR = exp(β)
#   - Covariates multiply the baseline hazard
#
# Accelerated Failure Time (AFT): h(t|x) = h₀(t·exp(-β'x)) exp(-β'x)
#   - Covariates accelerate/decelerate time
#   - exp(β) > 1 means shorter survival (time moves faster)
#
# For Exponential:
#   PH:  h(t|x) = λ exp(β'x)
#   AFT: h(t|x) = λ exp(-β'x)  (since h₀ is constant, only the multiplier changes)
#
# For Weibull:
#   PH:  h(t|x) = λκt^{κ-1} exp(β'x)
#   AFT: h(t|x) = λκt^{κ-1} exp(-κβ'x)

@testset "linpred_effect modes" begin
    dat = DataFrame(
        id = [1, 1],
        tstart = [0.0, 1.0],
        tstop = [1.0, 2.0],
        statefrom = [1, 1],
        stateto = [2, 2],
        obstype = [1, 1],
        x = [1.5, 1.5]
    )

    subjdat_row = dat[1, :]
    interval = (0.0, 1.25)

    # -------------------------------------------------------------------------
    # Test: Exponential PH vs AFT
    # -------------------------------------------------------------------------
    # PH:  h(t|x) = λ₀ exp(β·x)
    # AFT: h(t|x) = λ₀ exp(-β·x)
    
    h_exp_ph = Hazard(@formula(0 ~ x), "exp", 1, 2; linpred_effect = :ph)
    h_exp_aft = Hazard(@formula(0 ~ x), "exp", 1, 2; linpred_effect = :aft)
    model_exp_ph = multistatemodel(h_exp_ph; data = dat)
    model_exp_aft = multistatemodel(h_exp_aft; data = dat)
    MultistateModels.set_parameters!(model_exp_ph, (h12 = [log(0.4), 0.7],))

    MultistateModels.set_parameters!(model_exp_aft, (h12 = [log(0.4), 0.7],))

    # Use retrieved params for expected value calculation
    pars_exp_ph = get_log_scale_params(model_exp_ph.parameters)[1]
    pars_exp_aft = get_log_scale_params(model_exp_aft.parameters)[1]
    λ0_ph = exp(pars_exp_ph[1])
    β_ph = pars_exp_ph[2]
    λ0_aft = exp(pars_exp_aft[1])
    β_aft = pars_exp_aft[2]
    linear_pred_ph = β_ph * dat.x[1]
    linear_pred_aft = β_aft * dat.x[1]

    rate_ph = λ0_ph * exp(linear_pred_ph)
    rate_aft = λ0_aft * exp(-linear_pred_aft)
    log_rate_ph = log(rate_ph)
    log_rate_aft = log(rate_aft)
    hazard_exp_ph = model_exp_ph.hazards[1]
    hazard_exp_aft = model_exp_aft.hazards[1]
    covars_exp_ph = MultistateModels.extract_covariates_fast(subjdat_row, hazard_exp_ph.covar_names)
    covars_exp_aft = MultistateModels.extract_covariates_fast(subjdat_row, hazard_exp_aft.covar_names)
    @test MultistateModels.eval_hazard(hazard_exp_ph, 0.5, pars_exp_ph, covars_exp_ph) ≈ rate_ph
    @test log(MultistateModels.eval_hazard(hazard_exp_ph, 0.5, pars_exp_ph, covars_exp_ph)) ≈ log_rate_ph
    @test MultistateModels.eval_hazard(hazard_exp_aft, 0.5, pars_exp_aft, covars_exp_aft) ≈ rate_aft
    @test log(MultistateModels.eval_hazard(hazard_exp_aft, 0.5, pars_exp_aft, covars_exp_aft)) ≈ log_rate_aft
    cum_ph = rate_ph * (interval[2] - interval[1])
    cum_aft = rate_aft * (interval[2] - interval[1])
    @test MultistateModels.eval_cumhaz(hazard_exp_ph, interval[1], interval[2], pars_exp_ph, covars_exp_ph) ≈ cum_ph
    @test log(MultistateModels.eval_cumhaz(hazard_exp_ph, interval[1], interval[2], pars_exp_ph, covars_exp_ph)) ≈ log(cum_ph)
    @test MultistateModels.eval_cumhaz(hazard_exp_aft, interval[1], interval[2], pars_exp_aft, covars_exp_aft) ≈ cum_aft
    @test log(MultistateModels.eval_cumhaz(hazard_exp_aft, interval[1], interval[2], pars_exp_aft, covars_exp_aft)) ≈ log(cum_aft)

    h_wei_ph = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect = :ph)
    h_wei_aft = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect = :aft)
    model_wei_ph = multistatemodel(h_wei_ph; data = dat)
    model_wei_aft = multistatemodel(h_wei_aft; data = dat)
    MultistateModels.set_parameters!(model_wei_ph, (h12 = [log(1.4), log(0.6), -0.3],))
    MultistateModels.set_parameters!(model_wei_aft, (h12 = [log(1.4), log(0.6), -0.3],))

    # Use retrieved params for expected value calculation
    pars_wei_ph = get_log_scale_params(model_wei_ph.parameters)[1]
    pars_wei_aft = get_log_scale_params(model_wei_aft.parameters)[1]
    shape_ph = exp(pars_wei_ph[1])
    scale_ph = exp(pars_wei_ph[2])
    β_w_ph = pars_wei_ph[3]
    shape_aft = exp(pars_wei_aft[1])
    scale_aft = exp(pars_wei_aft[2])
    β_w_aft = pars_wei_aft[3]
    
    lp_w_ph = β_w_ph * dat.x[1]
    lp_w_aft = β_w_aft * dat.x[1]
    t = 1.8
    lb = 0.5
    ub = 1.6
    haz_ph = shape_ph * scale_ph * t^(shape_ph - 1) * exp(lp_w_ph)
    haz_aft = shape_aft * scale_aft * t^(shape_aft - 1) * exp(-shape_aft * lp_w_aft)
    cum_ph = scale_ph * exp(lp_w_ph) * (ub^shape_ph - lb^shape_ph)
    cum_aft = scale_aft * exp(-shape_aft * lp_w_aft) * (ub^shape_aft - lb^shape_aft)

    hazard_wei_ph = model_wei_ph.hazards[1]
    hazard_wei_aft = model_wei_aft.hazards[1]
    covars_wei_ph = MultistateModels.extract_covariates_fast(subjdat_row, hazard_wei_ph.covar_names)
    covars_wei_aft = MultistateModels.extract_covariates_fast(subjdat_row, hazard_wei_aft.covar_names)
    @test MultistateModels.eval_hazard(hazard_wei_ph, t, pars_wei_ph, covars_wei_ph) ≈ haz_ph
    @test MultistateModels.eval_hazard(hazard_wei_aft, t, pars_wei_aft, covars_wei_aft) ≈ haz_aft
    @test MultistateModels.eval_cumhaz(hazard_wei_ph, lb, ub, pars_wei_ph, covars_wei_ph) ≈ cum_ph
    @test MultistateModels.eval_cumhaz(hazard_wei_aft, lb, ub, pars_wei_aft, covars_wei_aft) ≈ cum_aft

    h_gom_ph = Hazard(@formula(0 ~ x), "gom", 1, 2; linpred_effect = :ph)
    h_gom_aft = Hazard(@formula(0 ~ x), "gom", 1, 2; linpred_effect = :aft)
    model_gom_ph = multistatemodel(h_gom_ph; data = dat)
    model_gom_aft = multistatemodel(h_gom_aft; data = dat)
    MultistateModels.set_parameters!(model_gom_ph, (h12 = [log(0.2), log(0.3), 0.5],))
    MultistateModels.set_parameters!(model_gom_aft, (h12 = [log(0.2), log(0.3), 0.5],))

    # Use retrieved params for expected value calculation
    pars_gom_ph = get_log_scale_params(model_gom_ph.parameters)[1]
    pars_gom_aft = get_log_scale_params(model_gom_aft.parameters)[1]
    shape_g_ph = exp(pars_gom_ph[1])
    scale_g_ph = exp(pars_gom_ph[2])
    β_g_ph = pars_gom_ph[3]
    shape_g_aft = exp(pars_gom_aft[1])
    scale_g_aft = exp(pars_gom_aft[2])
    β_g_aft = pars_gom_aft[3]
    
    lp_g_ph = β_g_ph * dat.x[1]
    lp_g_aft = β_g_aft * dat.x[1]
    t_g = 1.1
    lb_g = 0.2
    ub_g = 1.4
    haz_g_ph = scale_g_ph * shape_g_ph * exp(shape_g_ph * t_g + lp_g_ph)
    s = exp(-lp_g_aft)
    haz_g_aft = scale_g_aft * shape_g_aft * exp(shape_g_aft * t_g * s) * s
    cum_g_ph = begin
        base = if abs(shape_g_ph) < 1e-10
            scale_g_ph * (ub_g - lb_g)
        else
            scale_g_ph * (exp(shape_g_ph * ub_g) - exp(shape_g_ph * lb_g))
        end
        base * exp(lp_g_ph)
    end
    cum_g_aft = begin
        scaled_shape = shape_g_aft * s
        if abs(scaled_shape) < 1e-10
            scale_g_aft * s * (ub_g - lb_g)
        else
            scale_g_aft * (exp(scaled_shape * ub_g) - exp(scaled_shape * lb_g))
        end
    end

    hazard_gom_ph = model_gom_ph.hazards[1]
    hazard_gom_aft = model_gom_aft.hazards[1]
    covars_gom_ph = MultistateModels.extract_covariates_fast(subjdat_row, hazard_gom_ph.covar_names)
    covars_gom_aft = MultistateModels.extract_covariates_fast(subjdat_row, hazard_gom_aft.covar_names)
    @test MultistateModels.eval_hazard(hazard_gom_ph, t_g, pars_gom_ph, covars_gom_ph) ≈ haz_g_ph
    @test MultistateModels.eval_hazard(hazard_gom_aft, t_g, pars_gom_aft, covars_gom_aft) ≈ haz_g_aft
    @test MultistateModels.eval_cumhaz(hazard_gom_ph, lb_g, ub_g, pars_gom_ph, covars_gom_ph) ≈ cum_g_ph
    @test MultistateModels.eval_cumhaz(hazard_gom_aft, lb_g, ub_g, pars_gom_aft, covars_gom_aft) ≈ cum_g_aft
end

# --- Time-transform toggles and caches ----------------------------------------
@testset "time_transform mode" begin
    dat = DataFrame(
        id = [1],
        tstart = [0.0],
        tstop = [2.0],
        statefrom = [1],
        stateto = [2],
        obstype = [1],
        x = [1.5]
    )

    subjdat_row = dat[1, :]

    # Exponential PH hazard with Tang toggle
    h_exp_tt = Hazard(@formula(0 ~ x), "exp", 1, 2; time_transform = true)
    model_exp_tt = multistatemodel(h_exp_tt; data = dat)
    MultistateModels.set_parameters!(model_exp_tt, (h12 = [log(0.4), 0.7],))
    
    # Use retrieved params for expected value calculation
    pars_exp_tt = get_log_scale_params(model_exp_tt.parameters)[1]
    λ0_tt = exp(pars_exp_tt[1])
    β_tt = pars_exp_tt[2]
    rate_exp = λ0_tt * exp(β_tt * subjdat_row.x)
    t_eval = 0.8
    interval = (0.2, 1.1)
    hazard_exp_tt = model_exp_tt.hazards[1]
    covars_exp_tt = MultistateModels.extract_covariates_fast(subjdat_row, hazard_exp_tt.covar_names)
    @test MultistateModels.eval_hazard(hazard_exp_tt, t_eval, pars_exp_tt, covars_exp_tt; apply_transform = false) ≈ rate_exp
    @test MultistateModels.eval_hazard(hazard_exp_tt, t_eval, pars_exp_tt, covars_exp_tt; apply_transform = true) ≈ rate_exp
    @test log(MultistateModels.eval_hazard(hazard_exp_tt, t_eval, pars_exp_tt, covars_exp_tt; apply_transform = true)) ≈ log(rate_exp)
    cum_exp = rate_exp * (interval[2] - interval[1])
    @test MultistateModels.eval_cumhaz(hazard_exp_tt, interval[1], interval[2], pars_exp_tt, covars_exp_tt; apply_transform = true) ≈ cum_exp
    @test log(MultistateModels.eval_cumhaz(hazard_exp_tt, interval[1], interval[2], pars_exp_tt, covars_exp_tt; apply_transform = true)) ≈ log(cum_exp)
    @test MultistateModels.survprob(interval[1], interval[2], model_exp_tt.parameters, subjdat_row, model_exp_tt.totalhazards[1], model_exp_tt.hazards; give_log = false, apply_transform = true) ≈ exp(-cum_exp)

    # zero-length interval should return zero cumulative hazard and log zero
    @test MultistateModels.eval_cumhaz(hazard_exp_tt, interval[1], interval[1], pars_exp_tt, covars_exp_tt; apply_transform = true) == 0.0
    @test log(MultistateModels.eval_cumhaz(hazard_exp_tt, interval[1], interval[1], pars_exp_tt, covars_exp_tt; apply_transform = true)) == -Inf

    # Weibull PH hazard with Tang toggle
    h_wei_tt = Hazard(@formula(0 ~ x), "wei", 1, 2; time_transform = true)
    model_wei_tt = multistatemodel(h_wei_tt; data = dat)
    MultistateModels.set_parameters!(model_wei_tt, (h12 = [log(1.3), log(0.6), -0.4],))
    
    # Use retrieved params for expected value calculation
    pars_wei_tt = get_log_scale_params(model_wei_tt.parameters)[1]
    shape = exp(pars_wei_tt[1])
    scale = exp(pars_wei_tt[2])
    β_w = pars_wei_tt[3]
    lp_w = β_w * subjdat_row.x
    t_w = 1.25
    lb_w, ub_w = 0.3, 1.6
    haz_w = scale * shape * t_w^(shape - 1) * exp(lp_w)
    cum_w = scale * exp(lp_w) * (ub_w^shape - lb_w^shape)
    hazard_wei_tt = model_wei_tt.hazards[1]
    covars_wei_tt = MultistateModels.extract_covariates_fast(subjdat_row, hazard_wei_tt.covar_names)
    @test MultistateModels.eval_hazard(hazard_wei_tt, t_w, pars_wei_tt, covars_wei_tt; apply_transform = true) ≈ haz_w
    @test MultistateModels.eval_cumhaz(hazard_wei_tt, lb_w, ub_w, pars_wei_tt, covars_wei_tt; apply_transform = true) ≈ cum_w
    @test log(MultistateModels.eval_hazard(hazard_wei_tt, t_w, pars_wei_tt, covars_wei_tt; apply_transform = true)) ≈ log(haz_w)
    @test log(MultistateModels.eval_cumhaz(hazard_wei_tt, lb_w, ub_w, pars_wei_tt, covars_wei_tt; apply_transform = true)) ≈ log(cum_w)

    # Gompertz PH hazard with Tang toggle
    h_gom_tt = Hazard(@formula(0 ~ x), "gom", 1, 2; time_transform = true)
    model_gom_tt = multistatemodel(h_gom_tt; data = dat)
    MultistateModels.set_parameters!(model_gom_tt, (h12 = [log(0.5), log(0.8), 0.25],))
    
    # Use retrieved params for expected value calculation
    pars_gom_tt = get_log_scale_params(model_gom_tt.parameters)[1]
    shape_g = exp(pars_gom_tt[1])
    scale_g = exp(pars_gom_tt[2])
    β_g = pars_gom_tt[3]
    lp_g = β_g * subjdat_row.x
    t_g = 0.9
    lb_g, ub_g = 0.1, 1.4
    haz_g = scale_g * shape_g * exp(shape_g * t_g + lp_g)
    cum_g = scale_g * exp(lp_g) * (exp(shape_g * ub_g) - exp(shape_g * lb_g))
    hazard_gom_tt = model_gom_tt.hazards[1]
    covars_gom_tt = MultistateModels.extract_covariates_fast(subjdat_row, hazard_gom_tt.covar_names)
    @test MultistateModels.eval_hazard(hazard_gom_tt, t_g, pars_gom_tt, covars_gom_tt; apply_transform = true) ≈ haz_g
    @test MultistateModels.eval_cumhaz(hazard_gom_tt, lb_g, ub_g, pars_gom_tt, covars_gom_tt; apply_transform = true) ≈ cum_g
    @test log(MultistateModels.eval_hazard(hazard_gom_tt, t_g, pars_gom_tt, covars_gom_tt; apply_transform = true)) ≈ log(haz_g)
    @test log(MultistateModels.eval_cumhaz(hazard_gom_tt, lb_g, ub_g, pars_gom_tt, covars_gom_tt; apply_transform = true)) ≈ log(cum_g)

    # Cache ownership: shared-baseline table per origin state/family
    h_cache = Hazard(@formula(0 ~ x), "exp", 1, 2; time_transform = true)
    model_cache = multistatemodel(h_cache; data = dat)
    MultistateModels.set_parameters!(model_cache, (h12 = [log(0.5), 0.3],))
    pars_cache = copy(get_log_scale_params(model_cache.parameters)[1])
    t_cache = 0.7
    lb_cache, ub_cache = 0.1, 0.9
    lin_type = eltype(pars_cache)
    time_type = Float64
    ctx = MultistateModels.TimeTransformContext(lin_type, time_type, length(model_cache.hazards))
    hazard_slot = 1
    hazard_cache = model_cache.hazards[hazard_slot]
    covars_cache = MultistateModels.extract_covariates_fast(subjdat_row, hazard_cache.covar_names)
    shared_key = hazard_cache.shared_baseline_key
    @test shared_key !== nothing
    @test ctx.caches[hazard_slot] === nothing
    @test isempty(ctx.shared_baselines.caches)

    _ = MultistateModels.eval_hazard(hazard_cache, t_cache, pars_cache, covars_cache; apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)

    @test ctx.caches[hazard_slot] === nothing
    @test haskey(ctx.shared_baselines.caches, shared_key)
    cache = ctx.shared_baselines.caches[shared_key]
    @test length(cache.hazard_values) == 1
    _ = MultistateModels.eval_hazard(hazard_cache, t_cache, pars_cache, covars_cache; apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)
    @test length(cache.hazard_values) == 1
    pars_cache[2] += 0.2
    _ = MultistateModels.eval_hazard(hazard_cache, t_cache, pars_cache, covars_cache; apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)
    @test length(cache.hazard_values) == 2

    pars_cum = copy(get_log_scale_params(model_cache.parameters)[1])
    _ = MultistateModels.eval_cumhaz(hazard_cache, lb_cache, ub_cache, pars_cum, covars_cache; apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)
    @test length(cache.cumulhaz_values) == 1
    _ = MultistateModels.eval_cumhaz(hazard_cache, lb_cache, ub_cache, pars_cum, covars_cache; apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)
    @test length(cache.cumulhaz_values) == 1
    new_ub = 1.25
    _ = MultistateModels.eval_cumhaz(hazard_cache, lb_cache, new_ub, pars_cum, covars_cache; apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)
    @test length(cache.cumulhaz_values) == 2

    # Shared baseline caches are reused across hazards from the same origin
    dat_shared = DataFrame(
        id = [1, 1, 2, 2],
        tstart = [0.0, 0.5, 0.0, 0.6],
        tstop = [0.5, 1.0, 0.6, 1.2],
        statefrom = [1, 1, 1, 1],
        stateto = [2, 2, 3, 3],
        obstype = [1, 1, 1, 1],
        x = [0.5, 0.5, 1.0, 1.0]
    )

    h12_shared = Hazard(@formula(0 ~ x), "exp", 1, 2; time_transform = true)
    h13_shared = Hazard(@formula(0 ~ x), "exp", 1, 3; time_transform = true)
    model_shared = multistatemodel(h12_shared, h13_shared; data = dat_shared)
    MultistateModels.set_parameters!(model_shared, (h12 = [log(0.4), 0.2], h13 = [log(0.4), -0.1]))

    key12 = model_shared.hazards[1].shared_baseline_key
    key13 = model_shared.hazards[2].shared_baseline_key
    @test key12 !== nothing
    @test key12 == key13

    ctx_shared = MultistateModels.TimeTransformContext(Float64, Float64, length(model_shared.hazards))
    row12 = dat_shared[1, :]
    row13 = dat_shared[2, :]
    hazard12 = model_shared.hazards[1]
    hazard13 = model_shared.hazards[2]
    covars12 = MultistateModels.extract_covariates_fast(row12, hazard12.covar_names)
    covars13 = MultistateModels.extract_covariates_fast(row13, hazard13.covar_names)
    pars12 = copy(get_log_scale_params(model_shared.parameters)[1])
    pars13 = copy(get_log_scale_params(model_shared.parameters)[2])
    t_shared = 0.6

    @test ctx_shared.caches[1] === nothing
    @test ctx_shared.caches[2] === nothing
    @test isempty(ctx_shared.shared_baselines.caches)

    _ = MultistateModels.eval_hazard(hazard12, t_shared, pars12, covars12; apply_transform = true, cache_context = ctx_shared, hazard_slot = 1)
    cache_shared = ctx_shared.shared_baselines.caches[key12]
    @test cache_shared !== nothing
    @test length(cache_shared.hazard_values) == 1

    _ = MultistateModels.eval_hazard(hazard13, t_shared, pars13, covars13; apply_transform = true, cache_context = ctx_shared, hazard_slot = 2)
    @test ctx_shared.shared_baselines.caches[key13] === cache_shared
    @test length(cache_shared.hazard_values) == 2

    _ = MultistateModels.eval_cumhaz(hazard13, 0.0, 0.5, pars13, covars13; apply_transform = true, cache_context = ctx_shared, hazard_slot = 2)
    @test length(cache_shared.cumulhaz_values) == 1

    # Helper exposes context construction to callers
    subj_df = DataFrame(sojourn = Float64[0.2])
    params_shared = get_log_scale_params(model_shared.parameters)
    ctx_auto = MultistateModels.maybe_time_transform_context(params_shared, subj_df, model_shared.hazards)
    @test ctx_auto isa MultistateModels.TimeTransformContext
    plain_hazard = Hazard(@formula(0 ~ x), "exp", 1, 2)
    plain_model = multistatemodel(plain_hazard; data = dat)
    params_plain = get_log_scale_params(plain_model.parameters)
    ctx_none = MultistateModels.maybe_time_transform_context(params_plain, subj_df, plain_model.hazards)
    @test ctx_none === nothing

    # Spline hazards now support time_transform
    spline_spec = Hazard(@formula(0 ~ 1), "sp", 1, 2; time_transform = true, boundaryknots = [0.0, 1.0], degree = 1, natural_spline = false)
    spline_model = multistatemodel(spline_spec; data = dat)
    @test spline_model.hazards[1].metadata.time_transform == true
end

# =============================================================================
# TIME-TRANSFORM PARITY TESTS
# =============================================================================
# Verify that `time_transform = true` produces identical hazard/cumhaz values
# to the standard (non-transformed) evaluation for all parametric families.
#
# The `time_transform` option uses a log-time scale internally for numerical
# stability, but should produce mathematically equivalent results.
#
# For each family, we test:
#   1. Point hazards at various times
#   2. Cumulative hazards over intervals
#   3. Survival probabilities S(lb, ub) = exp(-H(lb, ub))
#
# Tests cover: Exponential, Weibull, and Gompertz PH hazards
# =============================================================================

@testset "time_transform parity" begin
    dat = DataFrame(
        id = [1, 1, 2, 2],
        tstart = [0.0, 0.5, 0.0, 0.4],
        tstop = [0.5, 1.0, 0.4, 1.1],
        statefrom = [1, 1, 1, 1],
        stateto = [2, 2, 2, 2],
        obstype = [1, 1, 1, 1],
        x = [0.2, 0.2, -0.5, -0.5],
        z = [1.0, 1.0, 2.0, 2.0]
    )

    # Exponential PH hazard with covariates
    h_plain_exp = Hazard(@formula(0 ~ x + z), "exp", 1, 2)
    h_tt_exp = Hazard(@formula(0 ~ x + z), "exp", 1, 2; time_transform = true)
    model_plain_exp = multistatemodel(h_plain_exp; data = dat)
    model_tt_exp = multistatemodel(h_tt_exp; data = dat)
    pars_exp = [log(0.35), 0.25, -0.15]
    MultistateModels.set_parameters!(model_plain_exp, (h12 = pars_exp,))
    MultistateModels.set_parameters!(model_tt_exp, (h12 = pars_exp,))

    eval_times = (0.2, 0.95)
    interval = (0.1, 0.9)

    hazard_plain_exp = model_plain_exp.hazards[1]
    hazard_tt_exp = model_tt_exp.hazards[1]
    for row in eachrow(dat)
        covars_plain_exp = MultistateModels.extract_covariates_fast(row, hazard_plain_exp.covar_names)
        covars_tt_exp = MultistateModels.extract_covariates_fast(row, hazard_tt_exp.covar_names)
        plain_haz = MultistateModels.eval_hazard(hazard_plain_exp, eval_times[1], get_log_scale_params(model_plain_exp.parameters)[1], covars_plain_exp)
        tt_haz = MultistateModels.eval_hazard(hazard_tt_exp, eval_times[1], get_log_scale_params(model_tt_exp.parameters)[1], covars_tt_exp; apply_transform = true)
        @test tt_haz ≈ plain_haz

        plain_log_haz = log(MultistateModels.eval_hazard(hazard_plain_exp, eval_times[2], get_log_scale_params(model_plain_exp.parameters)[1], covars_plain_exp))
        tt_log_haz = log(MultistateModels.eval_hazard(hazard_tt_exp, eval_times[2], get_log_scale_params(model_tt_exp.parameters)[1], covars_tt_exp; apply_transform = true))
        @test tt_log_haz ≈ plain_log_haz

        plain_cum = MultistateModels.eval_cumhaz(hazard_plain_exp, interval[1], interval[2], get_log_scale_params(model_plain_exp.parameters)[1], covars_plain_exp)
        tt_cum = MultistateModels.eval_cumhaz(hazard_tt_exp, interval[1], interval[2], get_log_scale_params(model_tt_exp.parameters)[1], covars_tt_exp; apply_transform = true)
        @test tt_cum ≈ plain_cum

        plain_surv = MultistateModels.survprob(interval[1], interval[2], model_plain_exp.parameters, row, model_plain_exp.totalhazards[1], model_plain_exp.hazards; give_log = false)
        tt_surv = MultistateModels.survprob(interval[1], interval[2], model_tt_exp.parameters, row, model_tt_exp.totalhazards[1], model_tt_exp.hazards; give_log = false, apply_transform = true)
        @test tt_surv ≈ plain_surv
    end

    # Weibull PH hazard with covariates
    h_plain_wei = Hazard(@formula(0 ~ x), "wei", 1, 2)
    h_tt_wei = Hazard(@formula(0 ~ x), "wei", 1, 2; time_transform = true)
    model_plain_wei = multistatemodel(h_plain_wei; data = dat)
    model_tt_wei = multistatemodel(h_tt_wei; data = dat)
    pars_wei = [log(1.4), log(0.8), 0.3]
    MultistateModels.set_parameters!(model_plain_wei, (h12 = pars_wei,))
    MultistateModels.set_parameters!(model_tt_wei, (h12 = pars_wei,))

    t_eval = 1.05
    interval_wei = (0.3, 1.2)

    hazard_plain_wei = model_plain_wei.hazards[1]
    hazard_tt_wei = model_tt_wei.hazards[1]
    for row in eachrow(dat)
        covars_plain_wei = MultistateModels.extract_covariates_fast(row, hazard_plain_wei.covar_names)
        covars_tt_wei = MultistateModels.extract_covariates_fast(row, hazard_tt_wei.covar_names)
        plain_haz = MultistateModels.eval_hazard(hazard_plain_wei, t_eval, get_log_scale_params(model_plain_wei.parameters)[1], covars_plain_wei)
        tt_haz = MultistateModels.eval_hazard(hazard_tt_wei, t_eval, get_log_scale_params(model_tt_wei.parameters)[1], covars_tt_wei; apply_transform = true)
        @test tt_haz ≈ plain_haz

        plain_cum = MultistateModels.eval_cumhaz(hazard_plain_wei, interval_wei[1], interval_wei[2], get_log_scale_params(model_plain_wei.parameters)[1], covars_plain_wei)
        tt_cum = MultistateModels.eval_cumhaz(hazard_tt_wei, interval_wei[1], interval_wei[2], get_log_scale_params(model_tt_wei.parameters)[1], covars_tt_wei; apply_transform = true)
        @test tt_cum ≈ plain_cum

        plain_surv = MultistateModels.survprob(interval_wei[1], interval_wei[2], model_plain_wei.parameters, row, model_plain_wei.totalhazards[1], model_plain_wei.hazards; give_log = false)
        tt_surv = MultistateModels.survprob(interval_wei[1], interval_wei[2], model_tt_wei.parameters, row, model_tt_wei.totalhazards[1], model_tt_wei.hazards; give_log = false, apply_transform = true)
        @test tt_surv ≈ plain_surv
    end

    # Gompertz PH hazard with covariates
    h_plain_gom = Hazard(@formula(0 ~ x + z), "gom", 1, 2)
    h_tt_gom = Hazard(@formula(0 ~ x + z), "gom", 1, 2; time_transform = true)
    model_plain_gom = multistatemodel(h_plain_gom; data = dat)
    model_tt_gom = multistatemodel(h_tt_gom; data = dat)
    pars_gom = [log(0.3), log(0.75), 0.4, -0.2]
    MultistateModels.set_parameters!(model_plain_gom, (h12 = pars_gom,))
    MultistateModels.set_parameters!(model_tt_gom, (h12 = pars_gom,))

    t_eval_g = 0.85
    interval_g = (0.15, 0.95)

    hazard_plain_gom = model_plain_gom.hazards[1]
    hazard_tt_gom = model_tt_gom.hazards[1]
    for row in eachrow(dat)
        covars_plain_gom = MultistateModels.extract_covariates_fast(row, hazard_plain_gom.covar_names)
        covars_tt_gom = MultistateModels.extract_covariates_fast(row, hazard_tt_gom.covar_names)
        plain_haz = MultistateModels.eval_hazard(hazard_plain_gom, t_eval_g, get_log_scale_params(model_plain_gom.parameters)[1], covars_plain_gom)
        tt_haz = MultistateModels.eval_hazard(hazard_tt_gom, t_eval_g, get_log_scale_params(model_tt_gom.parameters)[1], covars_tt_gom; apply_transform = true)
        @test tt_haz ≈ plain_haz

        plain_log_haz = log(MultistateModels.eval_hazard(hazard_plain_gom, t_eval_g, get_log_scale_params(model_plain_gom.parameters)[1], covars_plain_gom))
        tt_log_haz = log(MultistateModels.eval_hazard(hazard_tt_gom, t_eval_g, get_log_scale_params(model_tt_gom.parameters)[1], covars_tt_gom; apply_transform = true))
        @test tt_log_haz ≈ plain_log_haz

        plain_cum = MultistateModels.eval_cumhaz(hazard_plain_gom, interval_g[1], interval_g[2], get_log_scale_params(model_plain_gom.parameters)[1], covars_plain_gom)
        tt_cum = MultistateModels.eval_cumhaz(hazard_tt_gom, interval_g[1], interval_g[2], get_log_scale_params(model_tt_gom.parameters)[1], covars_tt_gom; apply_transform = true)
        @test tt_cum ≈ plain_cum

        plain_surv = MultistateModels.survprob(interval_g[1], interval_g[2], model_plain_gom.parameters, row, model_plain_gom.totalhazards[1], model_plain_gom.hazards; give_log = false)
        tt_surv = MultistateModels.survprob(interval_g[1], interval_g[2], model_tt_gom.parameters, row, model_tt_gom.totalhazards[1], model_tt_gom.hazards; give_log = false, apply_transform = true)
        @test tt_surv ≈ plain_surv
    end
end

# =============================================================================
# Spline hazard tests are in test_splines.jl
# =============================================================================
