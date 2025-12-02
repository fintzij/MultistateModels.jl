# =============================================================================
# Hazard Evaluation Tests
# =============================================================================
#
# Validates hazards/cumulative hazards across families (exp, wei, gom), linpred
# modalities (PH vs AFT), and optional Tang-style time transforms. Each section
# mirrors how modeling code consumes the APIs: single-hazard calls first, then
# aggregate quantities (`total_cumulhaz`, `survprob`), and finally cache/control
# paths around time transforms.
#
# Note: ParameterHandling.positive() uses ε = sqrt(eps(Float64)) ≈ 1.49e-8 as
# a safety margin, which introduces small numerical differences when parameters
# go through the exp/log round-trip. Tests use rtol=1e-6 for robustness.

using .TestFixtures: toy_expwei_model, toy_gompertz_model

# Standard tolerance for tests that compare computed vs expected values
# after parameter round-trip through ParameterHandling
const PARAM_RTOL = 1e-6

# --- Survival probability versus reference -------------------------------------
@testset "survprob" begin
    # what is the cumulative incidence from time 0 to 2 of exponential with mean time to event of 5
    # should be around 0.32967995
    fixture = toy_expwei_model()
    model = fixture.model
    MultistateModels.set_parameters!(model, (h12 = [log(0.1),], h13 = [log(0.1), 0.0, 0.0, 0.0]))

    # survprob expects DataFrame row, not row index
    subjdat_row = model.data[1, :]
    params = MultistateModels.get_log_scale_params(model.parameters)
    interval_incid = 1 - MultistateModels.survprob(0.0, 2.0, params, subjdat_row, model.totalhazards[1], model.hazards; give_log = false)
        
    # note that Distributions.jl uses the mean parameterization
    # i.e., 1/rate.
    # Use rtol since parameter goes through ParameterHandling round-trip
    @test isapprox(cdf(Exponential(5), 2), interval_incid; rtol=PARAM_RTOL)
end

# --- Exponential hazard evaluations -------------------------------------------
@testset "test_hazards_exp" begin
    fixture = toy_expwei_model()
    model = fixture.model
    data = fixture.data

    # create a parameters object
    MultistateModels.set_parameters!(model, (h12 = [0.8,], h13 = [0.3, 0.6, -0.4, 0.15]))

    # exponential hazards, no covariate adjustment
    @test isa(model.hazards[1], MultistateModels.MarkovHazard)
    subjdat_row = model.data[1, :]
    @test MultistateModels.call_haz(0.0, get_log_scale_params(model.parameters)[1], subjdat_row, model.hazards[1]; give_log = true) ≈ 0.8
    @test MultistateModels.call_haz(0.0, get_log_scale_params(model.parameters)[1], subjdat_row, model.hazards[1]; give_log = false) ≈ exp(0.8)

    # correct hazard value on log scale, for each row of data
    pars = get_log_scale_params(model.parameters)[2]
    trueval = 
        [pars[1] + data.trt[1] * pars[2] + data.age[1] * pars[3] + 
        data.trt[1] * data.age[1] * pars[4],
        pars[1] + data.trt[2] * pars[2] + data.age[2] * pars[3] + 
        data.trt[2] * data.age[2] * pars[4],
        pars[1] + data.trt[3] * pars[2] + data.age[3] * pars[3] + 
        data.trt[3] * data.age[3] * pars[4]]
    
    # loop through each row of data embedded in the model
    for h in axes(model.data, 1)
        subjdat_row = model.data[h, :]
        @test MultistateModels.call_haz(0.0, get_log_scale_params(model.parameters)[2], subjdat_row, model.hazards[2]; give_log = true) ≈ trueval[h]

        @test MultistateModels.call_haz(0.0, get_log_scale_params(model.parameters)[2], subjdat_row, model.hazards[2]; give_log = false) ≈ exp(trueval[h])
    end
end

# --- Weibull hazard evaluations ------------------------------------------------
@testset "test_hazards_weibull" begin
    fixture = toy_expwei_model()
    model = fixture.model

    # set parameters, log(shape, scale), no covariate adjustment
    MultistateModels.set_parameters!(model, (h21 = [-0.25, 0.2],))

    # h(t) = scale * shape * t^{shape - 1}
    subjdat_row = model.data[1, :]
    pars = get_log_scale_params(model.parameters)[3]
    t_eval = 1.7
    log_shape = pars[1]
    log_scale = pars[2]
    shape = exp(log_shape)
    log_hazard = log_scale + log_shape + (shape - 1) * log(t_eval)
    @test MultistateModels.call_haz(t_eval, pars, subjdat_row, model.hazards[3]; give_log = true) ≈ log_hazard

    @test MultistateModels.call_haz(t_eval, pars, subjdat_row, model.hazards[3]; give_log = false) ≈ exp(log_hazard)

    # set parameters, log(shape_intercept, scale_intercept, scale_trt) weibull PH with covariate adjustment
    # also set time at which to check hazard for correctness
    t = 1.0
    MultistateModels.set_parameters!(model, (h23 = [0.2, 0.25, -0.3],))
    # Get retrieved params (accounts for ParameterHandling round-trip)
    pars_h23 = get_log_scale_params(model.parameters)[4]

    # loop through each row of data embedded in the model, comparing truth to MultistateModels.call_haz output
    for h in axes(model.data, 1)
        subjdat_row = model.data[h, :]
        # Extract covariates from the row manually for truth calculation
        trt_val = subjdat_row.trt
        trueval = pars_h23[1] + expm1(pars_h23[1]) * log(t) + pars_h23[2] + pars_h23[3] * trt_val

        @test MultistateModels.call_haz(t, pars_h23, subjdat_row, model.hazards[4]; give_log=true) ≈ trueval

        @test MultistateModels.call_haz(t, pars_h23, subjdat_row, model.hazards[4]; give_log=false) ≈ exp(trueval)
    end
end

# --- Exponential cumulative hazards -------------------------------------------
@testset "test_cumulativehazards_exp" begin
    fixture = toy_expwei_model()
    model = fixture.model
    data = fixture.data

    # set parameters, lb (start time), and ub (end time)
    MultistateModels.set_parameters!(model, (h12 = [0.8,], h13 = [0.0, 0.6, -0.4, 0.15]))
    lb = 0
    ub = 5

    # cumulative hazard for exponential cause specific hazards, no covariate adjustment
    subjdat_row = model.data[1, :]
    @test MultistateModels.call_cumulhaz(lb, ub, get_log_scale_params(model.parameters)[1], subjdat_row, model.hazards[1], give_log = true) ≈ 0.8 + log(ub-lb)

    # cumulative hazard for exponential proportional hazards over [lb, ub], with covariate adjustment
    pars =  get_log_scale_params(model.parameters)[2] 
    log_haz = 
        [pars[1] + pars[2]*data.trt[1] + pars[3]*data.age[1] + pars[4]*data.trt[1]*data.age[1],
        pars[1] + pars[2]*data.trt[2] + pars[3]*data.age[2] + pars[4]*data.trt[2]*data.age[2],
        pars[1] + pars[2]*data.trt[3] + pars[3]*data.age[3] + pars[4]*data.trt[3]*data.age[3]]
    
    trueval = log_haz .+ log(ub-lb)

    for h in axes(model.data, 1)
        subjdat_row = model.data[h, :]
        @test MultistateModels.call_cumulhaz(lb, ub, get_log_scale_params(model.parameters)[2], subjdat_row, model.hazards[2], give_log = true) ≈ trueval[h]

        @test MultistateModels.call_cumulhaz(lb, ub, get_log_scale_params(model.parameters)[2], subjdat_row, model.hazards[2], give_log = false) ≈ exp(trueval[h])
    end
end

# --- Weibull cumulative hazards ------------------------------------------------
@testset "test_cumulativehazards_weibull" begin

    # set up log parameters, lower bound, and upper bound
    fixture = toy_expwei_model()
    model = fixture.model
    MultistateModels.set_parameters!(model, (h21 = [-0.25, 0.2], h23 = [0.2, 0.25, -0.3]))
    lb = 0
    ub = 5

    # cumulative hazard for weibull cause specific hazards, no covariate adjustment
    subjdat_row = model.data[1, :]
    # Use retrieved params for truth calculation (accounts for ParameterHandling round-trip)
    pars_h21 = get_log_scale_params(model.parameters)[3]
    trueval_h21 = log(ub^exp(pars_h21[1]) - lb^exp(pars_h21[1])) + pars_h21[2]
    @test MultistateModels.call_cumulhaz(lb, ub, pars_h21, subjdat_row, model.hazards[3], give_log = true) ≈ trueval_h21

    @test MultistateModels.call_cumulhaz(lb, ub, pars_h21, subjdat_row, model.hazards[3], give_log = false) ≈ exp(trueval_h21)

    # cumulative hazard for weibull proportional hazards over [lb, ub], with covariate adjustment
    pars =  get_log_scale_params(model.parameters)[4] 
    
    # loop through each row of data embedded in the model, comparing truth to MultistateModels.call_cumulhaz output
    for h in axes(model.data, 1)
        subjdat_row = model.data[h, :]
        # Extract covariates manually for truth calculation
        trt_val = subjdat_row.trt
        trueval = log(ub^exp(pars[1]) - lb^exp(pars[1])) + pars[2] + pars[3] * trt_val
                    
        @test MultistateModels.call_cumulhaz(lb, ub, get_log_scale_params(model.parameters)[4], subjdat_row, model.hazards[4]; give_log=true) ≈ trueval
    
        @test MultistateModels.call_cumulhaz(lb, ub, get_log_scale_params(model.parameters)[4], subjdat_row, model.hazards[4]; give_log=false) ≈ exp(trueval)
    end
end

# --- Total cumulative hazards --------------------------------------------------
@testset "test_totalcumulativehazards" begin

    # set parameters, lower bound, and upper bound
    fixture = toy_expwei_model()
    model = fixture.model
    MultistateModels.set_parameters!(model,
                                    (h12 = [0.8,],
                                     h13 = [0.8, 0.6, -0.4, 0.15],
                                     h21 = [0.8, 1.2],
                                     h23 = [0.8, 0.25, 1.2]))
    lb=0
    ub=5
    window = ub - lb

    pars_h12 = get_log_scale_params(model.parameters)[1]
    pars_h13 = get_log_scale_params(model.parameters)[2]
    pars_h21 = get_log_scale_params(model.parameters)[3]
    pars_h23 = get_log_scale_params(model.parameters)[4]

    cum12 = exp(pars_h12[1]) * window
    shape21 = exp(pars_h21[1])
    scale21 = exp(pars_h21[2])
    cum21 = scale21 * (ub^shape21 - lb^shape21)
    shape23 = exp(pars_h23[1])
    scale23 = exp(pars_h23[2])
    interval_term23 = scale23 * (ub^shape23 - lb^shape23)

    # test total cumulative hazard for each origin state    
    for row_idx in axes(model.data, 1)
        subjdat_row = model.data[row_idx, :]
        linpred13 = pars_h13[1] + pars_h13[2] * subjdat_row.trt + pars_h13[3] * subjdat_row.age + pars_h13[4] * subjdat_row.trt * subjdat_row.age
        cum13 = exp(linpred13) * window
        linpred23 = pars_h23[3] * subjdat_row.trt
        cum23 = interval_term23 * exp(linpred23)

        for s in axes(model.tmat, 1) 
            expected_total = if s == 1
                cum12 + cum13
            elseif s == 2
                cum21 + cum23
            else
                0.0
            end

            params = get_log_scale_params(model.parameters)
            result = MultistateModels.total_cumulhaz(lb, ub, params, subjdat_row, model.totalhazards[s], model.hazards; give_log = false)
            @test result ≈ expected_total

            log_result = MultistateModels.total_cumulhaz(lb, ub, params, subjdat_row, model.totalhazards[s], model.hazards; give_log = true)
            if expected_total == 0.0
                @test log_result == -Inf
            else
                @test log_result ≈ log(expected_total)
            end
        end
    end
end

# --- Gompertz hazard evaluations ----------------------------------------------
@testset "test_hazards_gompertz" begin

    # set parameters, log(shape, scale), no covariate adjustment
    fixture = toy_gompertz_model()
    model = fixture.model
    MultistateModels.set_parameters!(model, (h12 = [log(1.5), log(0.5)], h13 = [log(0.5), log(0.5), 1.5], h23 = [log(1), log(2/3)]))    

    # Get retrieved params (accounts for ParameterHandling round-trip)
    pars_h12 = get_log_scale_params(model.parameters)[1]
    pars_h13 = get_log_scale_params(model.parameters)[2]
    
    # h(t) = scale * exp(shape * t)
    subjdat_row = model.data[1, :]
    # log_shape = pars_h12[1], log_scale = pars_h12[2]
    # log h(t) = log_scale + log_shape + shape * t = pars[2] + pars[1] + exp(pars[1]) * t
    expected_log_haz = pars_h12[2] + pars_h12[1] + exp(pars_h12[1]) * 1.0
    @test MultistateModels.call_haz(1.0, pars_h12, subjdat_row, model.hazards[1]; give_log = true) ≈ expected_log_haz

    @test MultistateModels.call_haz(1.0, pars_h12, subjdat_row, model.hazards[1]; give_log = false) ≈ exp(expected_log_haz)

   # now with covariate adjustment
   subjdat_row2 = model.data[2, :]
   # For h13 with trt=1: log h(t) = log_scale + beta_trt + log_shape + shape * t
   # pars_h13 = [log_shape, log_scale, beta_trt]
   expected_log_haz_tvc = pars_h13[2] + pars_h13[1] + exp(pars_h13[1]) * 1.0 + pars_h13[3]
   @test MultistateModels.call_haz(1.0, pars_h13, subjdat_row2, model.hazards[2]; give_log = true) ≈ expected_log_haz_tvc

    @test MultistateModels.call_haz(1.0, pars_h13, subjdat_row2, model.hazards[2]; give_log = false) ≈ exp(expected_log_haz_tvc)
end

# --- Gompertz cumulative hazards ----------------------------------------------
@testset "test_cumulativehazards_gompertz" begin

    # set up log parameters, lower bound, and upper bound
    fixture = toy_gompertz_model()
    model = fixture.model
    MultistateModels.set_parameters!(model, (h12 = [log(1.5), log(0.5)], h13 = [log(0.5), log(0.5), 1.5], h23 = [log(1), log(2/3)]))
    lb = 0.0
    ub = 5.0

    # Get retrieved params
    pars_h12 = get_log_scale_params(model.parameters)[1]
    pars_h13 = get_log_scale_params(model.parameters)[2]
    
    # Cumulative hazard for Gompertz: H(t) = (scale/shape) * (exp(shape*t) - 1)
    # log H(lb, ub) = log_scale - log_shape + log(exp(shape*ub) - exp(shape*lb))
    # Note: for Gompertz, shape = exp(log_shape)
    subjdat_row = model.data[1, :]
    shape_h12 = exp(pars_h12[1])
    expected_log_cumhaz = pars_h12[2] + log(exp(shape_h12 * ub) - exp(shape_h12 * lb))
    @test MultistateModels.call_cumulhaz(lb, ub, pars_h12, subjdat_row, model.hazards[1]; give_log = true) ≈ expected_log_cumhaz
    
    @test MultistateModels.call_cumulhaz(lb, ub, pars_h12, subjdat_row, model.hazards[1]; give_log = false) ≈ exp(expected_log_cumhaz)

   # now with covariate adjustment
   subjdat_row2 = model.data[2, :]
   shape_h13 = exp(pars_h13[1])
   expected_log_cumhaz_tvc = pars_h13[2] + pars_h13[3] + log(exp(shape_h13 * ub) - exp(shape_h13 * lb))
   @test MultistateModels.call_cumulhaz(lb, ub, pars_h13, subjdat_row2, model.hazards[2]; give_log = true) ≈ expected_log_cumhaz_tvc

    @test MultistateModels.call_cumulhaz(lb, ub, pars_h13, subjdat_row2, model.hazards[2]; give_log = false) ≈ exp(expected_log_cumhaz_tvc)
end

# --- Linear predictor effect modes -------------------------------------------
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
    @test MultistateModels.call_haz(0.5, pars_exp_ph, subjdat_row, model_exp_ph.hazards[1]; give_log = false) ≈ rate_ph
    @test MultistateModels.call_haz(0.5, pars_exp_ph, subjdat_row, model_exp_ph.hazards[1]; give_log = true) ≈ log_rate_ph
    @test MultistateModels.call_haz(0.5, pars_exp_aft, subjdat_row, model_exp_aft.hazards[1]; give_log = false) ≈ rate_aft
    @test MultistateModels.call_haz(0.5, pars_exp_aft, subjdat_row, model_exp_aft.hazards[1]; give_log = true) ≈ log_rate_aft
    cum_ph = rate_ph * (interval[2] - interval[1])
    cum_aft = rate_aft * (interval[2] - interval[1])
    @test MultistateModels.call_cumulhaz(interval[1], interval[2], pars_exp_ph, subjdat_row, model_exp_ph.hazards[1]; give_log = false) ≈ cum_ph
    @test MultistateModels.call_cumulhaz(interval[1], interval[2], pars_exp_ph, subjdat_row, model_exp_ph.hazards[1]; give_log = true) ≈ log(cum_ph)
    @test MultistateModels.call_cumulhaz(interval[1], interval[2], pars_exp_aft, subjdat_row, model_exp_aft.hazards[1]; give_log = false) ≈ cum_aft
    @test MultistateModels.call_cumulhaz(interval[1], interval[2], pars_exp_aft, subjdat_row, model_exp_aft.hazards[1]; give_log = true) ≈ log(cum_aft)

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

    @test MultistateModels.call_haz(t, pars_wei_ph, subjdat_row, model_wei_ph.hazards[1]; give_log = false) ≈ haz_ph
    @test MultistateModels.call_haz(t, pars_wei_aft, subjdat_row, model_wei_aft.hazards[1]; give_log = false) ≈ haz_aft
    @test MultistateModels.call_cumulhaz(lb, ub, pars_wei_ph, subjdat_row, model_wei_ph.hazards[1]; give_log = false) ≈ cum_ph
    @test MultistateModels.call_cumulhaz(lb, ub, pars_wei_aft, subjdat_row, model_wei_aft.hazards[1]; give_log = false) ≈ cum_aft

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

    @test MultistateModels.call_haz(t_g, pars_gom_ph, subjdat_row, model_gom_ph.hazards[1]; give_log = false) ≈ haz_g_ph
    @test MultistateModels.call_haz(t_g, pars_gom_aft, subjdat_row, model_gom_aft.hazards[1]; give_log = false) ≈ haz_g_aft
    @test MultistateModels.call_cumulhaz(lb_g, ub_g, pars_gom_ph, subjdat_row, model_gom_ph.hazards[1]; give_log = false) ≈ cum_g_ph
    @test MultistateModels.call_cumulhaz(lb_g, ub_g, pars_gom_aft, subjdat_row, model_gom_aft.hazards[1]; give_log = false) ≈ cum_g_aft
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
    @test MultistateModels.call_haz(t_eval, pars_exp_tt, subjdat_row, model_exp_tt.hazards[1]; give_log = false, apply_transform = false) ≈ rate_exp
    @test MultistateModels.call_haz(t_eval, pars_exp_tt, subjdat_row, model_exp_tt.hazards[1]; give_log = false, apply_transform = true) ≈ rate_exp
    @test MultistateModels.call_haz(t_eval, pars_exp_tt, subjdat_row, model_exp_tt.hazards[1]; give_log = true, apply_transform = true) ≈ log(rate_exp)
    cum_exp = rate_exp * (interval[2] - interval[1])
    @test MultistateModels.call_cumulhaz(interval[1], interval[2], pars_exp_tt, subjdat_row, model_exp_tt.hazards[1]; give_log = false, apply_transform = true) ≈ cum_exp
    @test MultistateModels.call_cumulhaz(interval[1], interval[2], pars_exp_tt, subjdat_row, model_exp_tt.hazards[1]; give_log = true, apply_transform = true) ≈ log(cum_exp)
    @test MultistateModels.survprob(interval[1], interval[2], model_exp_tt.parameters, subjdat_row, model_exp_tt.totalhazards[1], model_exp_tt.hazards; give_log = false, apply_transform = true) ≈ exp(-cum_exp)

    # zero-length interval should return zero cumulative hazard and log zero
    @test MultistateModels.call_cumulhaz(interval[1], interval[1], pars_exp_tt, subjdat_row, model_exp_tt.hazards[1]; give_log = false, apply_transform = true) == 0.0
    @test MultistateModels.call_cumulhaz(interval[1], interval[1], pars_exp_tt, subjdat_row, model_exp_tt.hazards[1]; give_log = true, apply_transform = true) == -Inf

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
    @test MultistateModels.call_haz(t_w, pars_wei_tt, subjdat_row, model_wei_tt.hazards[1]; give_log = false, apply_transform = true) ≈ haz_w
    @test MultistateModels.call_cumulhaz(lb_w, ub_w, pars_wei_tt, subjdat_row, model_wei_tt.hazards[1]; give_log = false, apply_transform = true) ≈ cum_w
    @test MultistateModels.call_haz(t_w, pars_wei_tt, subjdat_row, model_wei_tt.hazards[1]; give_log = true, apply_transform = true) ≈ log(haz_w)
    @test MultistateModels.call_cumulhaz(lb_w, ub_w, pars_wei_tt, subjdat_row, model_wei_tt.hazards[1]; give_log = true, apply_transform = true) ≈ log(cum_w)

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
    @test MultistateModels.call_haz(t_g, pars_gom_tt, subjdat_row, model_gom_tt.hazards[1]; give_log = false, apply_transform = true) ≈ haz_g
    @test MultistateModels.call_cumulhaz(lb_g, ub_g, pars_gom_tt, subjdat_row, model_gom_tt.hazards[1]; give_log = false, apply_transform = true) ≈ cum_g
    @test MultistateModels.call_haz(t_g, pars_gom_tt, subjdat_row, model_gom_tt.hazards[1]; give_log = true, apply_transform = true) ≈ log(haz_g)
    @test MultistateModels.call_cumulhaz(lb_g, ub_g, pars_gom_tt, subjdat_row, model_gom_tt.hazards[1]; give_log = true, apply_transform = true) ≈ log(cum_g)

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
    shared_key = model_cache.hazards[hazard_slot].shared_baseline_key
    @test shared_key !== nothing
    @test ctx.caches[hazard_slot] === nothing
    @test isempty(ctx.shared_baselines.caches)

    _ = MultistateModels.call_haz(t_cache, pars_cache, subjdat_row, model_cache.hazards[hazard_slot]; give_log = false, apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)

    @test ctx.caches[hazard_slot] === nothing
    @test haskey(ctx.shared_baselines.caches, shared_key)
    cache = ctx.shared_baselines.caches[shared_key]
    @test length(cache.hazard_values) == 1
    _ = MultistateModels.call_haz(t_cache, pars_cache, subjdat_row, model_cache.hazards[hazard_slot]; give_log = false, apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)
    @test length(cache.hazard_values) == 1
    pars_cache[2] += 0.2
    _ = MultistateModels.call_haz(t_cache, pars_cache, subjdat_row, model_cache.hazards[hazard_slot]; give_log = false, apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)
    @test length(cache.hazard_values) == 2

    pars_cum = copy(get_log_scale_params(model_cache.parameters)[1])
    _ = MultistateModels.call_cumulhaz(lb_cache, ub_cache, pars_cum, subjdat_row, model_cache.hazards[hazard_slot]; give_log = false, apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)
    @test length(cache.cumulhaz_values) == 1
    _ = MultistateModels.call_cumulhaz(lb_cache, ub_cache, pars_cum, subjdat_row, model_cache.hazards[hazard_slot]; give_log = false, apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)
    @test length(cache.cumulhaz_values) == 1
    new_ub = 1.25
    _ = MultistateModels.call_cumulhaz(lb_cache, new_ub, pars_cum, subjdat_row, model_cache.hazards[hazard_slot]; give_log = false, apply_transform = true, cache_context = ctx, hazard_slot = hazard_slot)
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
    pars12 = copy(get_log_scale_params(model_shared.parameters)[1])
    pars13 = copy(get_log_scale_params(model_shared.parameters)[2])
    t_shared = 0.6

    @test ctx_shared.caches[1] === nothing
    @test ctx_shared.caches[2] === nothing
    @test isempty(ctx_shared.shared_baselines.caches)

    _ = MultistateModels.call_haz(t_shared, pars12, row12, model_shared.hazards[1]; give_log = false, apply_transform = true, cache_context = ctx_shared, hazard_slot = 1)
    cache_shared = ctx_shared.shared_baselines.caches[key12]
    @test cache_shared !== nothing
    @test length(cache_shared.hazard_values) == 1

    _ = MultistateModels.call_haz(t_shared, pars13, row13, model_shared.hazards[2]; give_log = false, apply_transform = true, cache_context = ctx_shared, hazard_slot = 2)
    @test ctx_shared.shared_baselines.caches[key13] === cache_shared
    @test length(cache_shared.hazard_values) == 2

    _ = MultistateModels.call_cumulhaz(0.0, 0.5, pars13, row13, model_shared.hazards[2]; give_log = false, apply_transform = true, cache_context = ctx_shared, hazard_slot = 2)
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

    for row in eachrow(dat)
        plain_haz = MultistateModels.call_haz(eval_times[1], get_log_scale_params(model_plain_exp.parameters)[1], row, model_plain_exp.hazards[1]; give_log = false)
        tt_haz = MultistateModels.call_haz(eval_times[1], get_log_scale_params(model_tt_exp.parameters)[1], row, model_tt_exp.hazards[1]; give_log = false, apply_transform = true)
        @test tt_haz ≈ plain_haz

        plain_log_haz = MultistateModels.call_haz(eval_times[2], get_log_scale_params(model_plain_exp.parameters)[1], row, model_plain_exp.hazards[1]; give_log = true)
        tt_log_haz = MultistateModels.call_haz(eval_times[2], get_log_scale_params(model_tt_exp.parameters)[1], row, model_tt_exp.hazards[1]; give_log = true, apply_transform = true)
        @test tt_log_haz ≈ plain_log_haz

        plain_cum = MultistateModels.call_cumulhaz(interval[1], interval[2], get_log_scale_params(model_plain_exp.parameters)[1], row, model_plain_exp.hazards[1]; give_log = false)
        tt_cum = MultistateModels.call_cumulhaz(interval[1], interval[2], get_log_scale_params(model_tt_exp.parameters)[1], row, model_tt_exp.hazards[1]; give_log = false, apply_transform = true)
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

    for row in eachrow(dat)
        plain_haz = MultistateModels.call_haz(t_eval, get_log_scale_params(model_plain_wei.parameters)[1], row, model_plain_wei.hazards[1]; give_log = false)
        tt_haz = MultistateModels.call_haz(t_eval, get_log_scale_params(model_tt_wei.parameters)[1], row, model_tt_wei.hazards[1]; give_log = false, apply_transform = true)
        @test tt_haz ≈ plain_haz

        plain_cum = MultistateModels.call_cumulhaz(interval_wei[1], interval_wei[2], get_log_scale_params(model_plain_wei.parameters)[1], row, model_plain_wei.hazards[1]; give_log = false)
        tt_cum = MultistateModels.call_cumulhaz(interval_wei[1], interval_wei[2], get_log_scale_params(model_tt_wei.parameters)[1], row, model_tt_wei.hazards[1]; give_log = false, apply_transform = true)
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

    for row in eachrow(dat)
        plain_haz = MultistateModels.call_haz(t_eval_g, get_log_scale_params(model_plain_gom.parameters)[1], row, model_plain_gom.hazards[1]; give_log = false)
        tt_haz = MultistateModels.call_haz(t_eval_g, get_log_scale_params(model_tt_gom.parameters)[1], row, model_tt_gom.hazards[1]; give_log = false, apply_transform = true)
        @test tt_haz ≈ plain_haz

        plain_log_haz = MultistateModels.call_haz(t_eval_g, get_log_scale_params(model_plain_gom.parameters)[1], row, model_plain_gom.hazards[1]; give_log = true)
        tt_log_haz = MultistateModels.call_haz(t_eval_g, get_log_scale_params(model_tt_gom.parameters)[1], row, model_tt_gom.hazards[1]; give_log = true, apply_transform = true)
        @test tt_log_haz ≈ plain_log_haz

        plain_cum = MultistateModels.call_cumulhaz(interval_g[1], interval_g[2], get_log_scale_params(model_plain_gom.parameters)[1], row, model_plain_gom.hazards[1]; give_log = false)
        tt_cum = MultistateModels.call_cumulhaz(interval_g[1], interval_g[2], get_log_scale_params(model_tt_gom.parameters)[1], row, model_tt_gom.hazards[1]; give_log = false, apply_transform = true)
        @test tt_cum ≈ plain_cum

        plain_surv = MultistateModels.survprob(interval_g[1], interval_g[2], model_plain_gom.parameters, row, model_plain_gom.totalhazards[1], model_plain_gom.hazards; give_log = false)
        tt_surv = MultistateModels.survprob(interval_g[1], interval_g[2], model_tt_gom.parameters, row, model_tt_gom.totalhazards[1], model_tt_gom.hazards; give_log = false, apply_transform = true)
        @test tt_surv ≈ plain_surv
    end
end

# TODO: Splines not yet implemented - splinemod not defined
# @testset "test_splines" begin
#     # test that crudely integrated hazard and cumulative hazard are rougly the same
#     cumul_haz_crude = 0.0
#     hazind = 1
#     ntimes = 1000000
#     delta = 1/ntimes
# 
#     for h in eachindex(splinemod.hazards)
#         # initialize
#         chaz_crude_interp = 0.0
#         chaz_crude_extrap = 0.0
# 
#         # Get DataFrameRow for first subject
#         subjdat_row = splinemod.data[1, :]
# 
#         # integrate over the boundaries
#         boundaries = [BSplineKit.boundaries(splinemod.hazards[h].hazsp.spline.basis)...]
#         times = (boundaries[1] + delta):delta:boundaries[2]
#         for t in times
#             chaz_crude_interp += MultistateModels.call_haz(t, splinemod.parameters[h], subjdat_row, splinemod.hazards[h]; give_log = false) * delta
#         end
# 
#         # compute the cumulative hazard
#         chaz_interp = MultistateModels.call_cumulhaz(boundaries[1], boundaries[2], splinemod.parameters[h], subjdat_row, splinemod.hazards[h]; give_log = false)
# 
#         # integrate over the timespan
#         @test isapprox(chaz_crude_interp, chaz_interp; atol = delta * 10)    
# 
#         # integrate over the timespan
#         # boundaries = splinemod.hazards[1].timespan
#         boundaries = [0.0, 1.0]
#         times = (boundaries[1] + delta):delta:boundaries[2]
#         for t in times
#             chaz_crude_extrap += MultistateModels.call_haz(t, splinemod.parameters[h], subjdat_row, splinemod.hazards[h]; give_log = false) * delta
#         end
# 
#         # compute the cumulative hazard
#         chaz_extrap = MultistateModels.call_cumulhaz(boundaries[1], boundaries[2], splinemod.parameters[h], subjdat_row, splinemod.hazards[h]; give_log = false)
# 
#         # integrate over the timespan
#         @test isapprox(chaz_crude_extrap, chaz_extrap; atol = delta * 10)
#     end
# end
