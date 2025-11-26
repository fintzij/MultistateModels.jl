# Tests for Hazard and _hazard structs and call_haz methods
# 1. Check accuracy of hazards, cumulative hazards, total hazard
#       - Check for non Float64 stuff
#       - Edge cases? Zero hazard, infinite hazard, negative hazard (should throw error)
#       - Test for numerical problems (see Distributions.jl for ideas)
# 2. function to validate data
# 3. validate MultistateModel object    

# test that we are generating correct cumulative incidence using survprob
@testset "survprob" begin
    # what is the cumulative incidence from time 0 to 2 of exponential with mean time to event of 5
    # should be around 0.32967995
    MultistateModels.set_parameters!(msm_expwei, (h12 = [log(0.1),], h13 = [log(0.1), 0.0, 0.0, 0.0]))

    # survprob expects DataFrame row, not row index
    subjdat_row = msm_expwei.data[1, :]
    interval_incid = 1 - MultistateModels.survprob(0.0, 2.0, msm_expwei.parameters, subjdat_row, msm_expwei.totalhazards[1], msm_expwei.hazards; give_log = false)
        
    # note that Distributions.jl uses the mean parameterization
    # i.e., 1/rate. 
    @test cdf(Exponential(5), 2) ≈ interval_incid
end

# tests for individual hazards
@testset "test_hazards_exp" begin
    
    # create a parameters object
    MultistateModels.set_parameters!(msm_expwei, (h12 = [0.8,], h13 = [0.0, 0.6, -0.4, 0.15]))

    # exponential hazards, no covariate adjustment
    @test isa(msm_expwei.hazards[1], MultistateModels.MarkovHazard)
    subjdat_row = msm_expwei.data[1, :]
    @test MultistateModels.call_haz(0.0, msm_expwei.parameters[1], subjdat_row, msm_expwei.hazards[1]; give_log = true) ≈ 0.8
    @test MultistateModels.call_haz(0.0, msm_expwei.parameters[1], subjdat_row, msm_expwei.hazards[1]; give_log = false) ≈ exp(0.8)

    # correct hazard value on log scale, for each row of data
    pars = msm_expwei.parameters[2]
    trueval = 
        [dat_exact2.trt[1] * pars[2] + dat_exact2.age[1] * pars[3] + 
        dat_exact2.trt[1] * dat_exact2.age[1] * pars[4],
        dat_exact2.trt[2] * pars[2] + dat_exact2.age[2] * pars[3] + 
        dat_exact2.trt[2] * dat_exact2.age[2] * pars[4],
        dat_exact2.trt[3] * pars[2] + dat_exact2.age[3] * pars[3] + 
        dat_exact2.trt[3] * dat_exact2.age[3] * pars[4]]
    
    # loop through each row of data embedded in the msm_expwei object
    for h in axes(msm_expwei.data, 1)
        subjdat_row = msm_expwei.data[h, :]
        @test MultistateModels.call_haz(0.0, msm_expwei.parameters[2], subjdat_row, msm_expwei.hazards[2]; give_log = true) ≈ trueval[h]

        @test MultistateModels.call_haz(0.0, msm_expwei.parameters[2], subjdat_row, msm_expwei.hazards[2]; give_log = false) ≈ exp(trueval[h])
    end
end

@testset "test_hazards_weibull" begin

    # set parameters, log(shape, scale), no covariate adjustment
    MultistateModels.set_parameters!(msm_expwei, (h21 = [-0.25, 0.2],))
    

        # h(t) = scale * shape * t^{shape - 1}
    subjdat_row = msm_expwei.data[1, :]
    @test MultistateModels.call_haz(1.0, msm_expwei.parameters[3], subjdat_row, msm_expwei.hazards[3]; give_log = true) ≈ 0.2 - 0.25

    @test MultistateModels.call_haz(1.0, msm_expwei.parameters[3], subjdat_row, msm_expwei.hazards[3]; give_log = false) ≈ exp(0.2) * exp(-0.25)

    # set parameters, log(shape_intercept, scale_intercept, scale_trt) weibull PH with covariate adjustment
    # also set time at which to check hazard for correctness
    pars = [0.2, 0.25, -0.3]
    t = 1.0
    MultistateModels.set_parameters!(msm_expwei, (h23 = pars,))

    # loop through each row of data embedded in the msm_expwei object, comparing truth to MultistateModels.call_haz output
    for h in axes(msm_expwei.data, 1)
        subjdat_row = msm_expwei.data[h, :]
        # Extract covariates from the row manually for truth calculation
        trt_val = subjdat_row.trt
        trueval = pars[1] + expm1(pars[1]) * log(t) + pars[2] + pars[3] * trt_val
                
        @test MultistateModels.call_haz(t, msm_expwei.parameters[4], subjdat_row, msm_expwei.hazards[4]; give_log=true) ≈ trueval

        @test MultistateModels.call_haz(t, msm_expwei.parameters[4], subjdat_row, msm_expwei.hazards[4]; give_log=false) ≈ exp(trueval)
    end
end

@testset "test_cumulativehazards_exp" begin
    
    # set parameters, lb (start time), and ub (end time)
    MultistateModels.set_parameters!(msm_expwei, (h12 = [0.8,], h13 = [0.0, 0.6, -0.4, 0.15]))
    lb = 0
    ub = 5

    # cumulative hazard for exponential cause specific hazards, no covariate adjustment
    subjdat_row = msm_expwei.data[1, :]
    @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[1], subjdat_row, msm_expwei.hazards[1], give_log = true) == 0.8 + log(ub-lb)

    # cumulative hazard for exponential proportional hazards over [lb, ub], with covariate adjustment
    pars =  msm_expwei.parameters[2] 
    log_haz = 
        [pars[2]*dat_exact2.trt[1] + pars[3]*dat_exact2.age[1] + pars[4]*dat_exact2.trt[1]*dat_exact2.age[1],
        pars[2]*dat_exact2.trt[2] + pars[3]*dat_exact2.age[2] + pars[4]*dat_exact2.trt[2]*dat_exact2.age[2],
        pars[2]*dat_exact2.trt[1] + pars[3]*dat_exact2.age[3] + pars[4]*dat_exact2.trt[3]*dat_exact2.age[3]]
    
    trueval = log_haz .+ log(ub-lb)

    for h in axes(msm_expwei.data, 1)
        subjdat_row = msm_expwei.data[h, :]
        @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[2], subjdat_row, msm_expwei.hazards[2], give_log = true) ≈ trueval[h]

        @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[2], subjdat_row, msm_expwei.hazards[2], give_log = false) ≈ exp(trueval[h])
    end
end

@testset "test_cumulativehazards_weibull" begin

    # set up log parameters, lower bound, and upper bound
    MultistateModels.set_parameters!(msm_expwei, (h21 = [-0.25, 0.2], h23 = [0.2, 0.25, -0.3]))
    lb = 0
    ub = 5

    # cumulative hazard for weibull cause specific hazards, no covariate adjustment
    subjdat_row = msm_expwei.data[1, :]
    @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[3], subjdat_row, msm_expwei.hazards[3], give_log = true) ≈ log(ub^exp(-0.25)-lb^exp(-0.25)) + 0.2

    @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[3], subjdat_row, msm_expwei.hazards[3], give_log = false) ≈ exp(log(ub^exp(-0.25)-lb^exp(-0.25)) + 0.2)

    # cumulative hazard for weibull proportional hazards over [lb, ub], with covariate adjustment
    pars =  msm_expwei.parameters[4] 
    
    # loop through each row of data embedded in the msm_expwei object, comparing truth to MultistateModels.call_cumulhaz output
    for h in axes(msm_expwei.data, 1)
        subjdat_row = msm_expwei.data[h, :]
        # Extract covariates manually for truth calculation
        trt_val = subjdat_row.trt
        trueval = log(ub^exp(pars[1]) - lb^exp(pars[1])) + pars[2] + pars[3] * trt_val
                    
        @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[4], subjdat_row, msm_expwei.hazards[4]; give_log=true) ≈ trueval
    
        @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[4], subjdat_row, msm_expwei.hazards[4]; give_log=false) ≈ exp(trueval)
    end
end

@testset "test_totalcumulativehazards" begin

    # set parameters, lower bound, and upper bound
    MultistateModels.set_parameters!(msm_expwei,
                                    (h12 = [0.8,],
                                     h13 = [0.8, 0.6, -0.4, 0.15],
                                     h21 = [0.8, 1.2],
                                     h23 = [0.8, 0.25, 1.2]))
    lb=0
    ub=5

    # test total cumulative hazard for each origin state    
    for h in axes(msm_expwei.data, 1)
        subjdat_row = msm_expwei.data[h, :]
        for s in axes(msm_expwei.tmat, 1) 
            if s == 1
                total_cumulhaz = 
                    MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[1], subjdat_row, msm_expwei.hazards[1]; give_log = false) +
                     MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[2], subjdat_row, msm_expwei.hazards[2]; give_log = false)
                
            elseif s == 2
                total_cumulhaz = 
                    MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[3], subjdat_row, msm_expwei.hazards[3]; give_log = false) +
                     MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[4], subjdat_row, msm_expwei.hazards[4]; give_log = false)
            else
                total_cumulhaz = 0.0
            end            

            @test MultistateModels.total_cumulhaz(lb, ub, msm_expwei.parameters, subjdat_row, msm_expwei.totalhazards[s], msm_expwei.hazards; give_log = false) ≈ total_cumulhaz
            
            @test MultistateModels.total_cumulhaz(lb, ub, msm_expwei.parameters, subjdat_row, msm_expwei.totalhazards[s], msm_expwei.hazards; give_log = true) ≈ log(total_cumulhaz)
        end
    end
end

@testset "test_hazards_gompertz" begin

    # set parameters, log(shape, scale), no covariate adjustment
    MultistateModels.set_parameters!(msm_gom, (h12 = [log(1.5), log(0.5)], h13 = [log(0.5), log(0.5), 1.5], h23 = [log(1), log(2/3)]))    

    # h(t) = scale * exp(shape * t)
    subjdat_row = msm_gom.data[1, :]
    @test MultistateModels.call_haz(1.0, msm_gom.parameters[1], subjdat_row, msm_gom.hazards[1]; give_log = true) == log(0.5) + log(1.5) + 1.5

    @test MultistateModels.call_haz(1.0, msm_gom.parameters[1], subjdat_row, msm_gom.hazards[1]; give_log = false) ≈ 0.5 * 1.5 * exp(1.5)

   # now with covariate adjustment
   subjdat_row2 = msm_gom.data[2, :]
   @test MultistateModels.call_haz(1.0, msm_gom.parameters[2], subjdat_row2, msm_gom.hazards[2]; give_log = true) == log(0.5) + log(0.5) + exp(log(0.5)) + 1.5

    @test MultistateModels.call_haz(1.0, msm_gom.parameters[2], subjdat_row2, msm_gom.hazards[2]; give_log = false) == exp(log(0.5) + log(0.5) + exp(log(0.5)) + 1.5)
end

@testset "test_cumulativehazards_gompertz" begin

    # set up log parameters, lower bound, and upper bound
    MultistateModels.set_parameters!(msm_gom, (h12 = [log(1.5), log(0.5)], h13 = [log(0.5), log(0.5), 1.5], h23 = [log(1), log(2/3)]))
    lb = 0.0
    ub = 5.0

    # test
    subjdat_row = msm_gom.data[1, :]
    @test MultistateModels.call_cumulhaz(lb, ub, msm_gom.parameters[1], subjdat_row, msm_gom.hazards[1]; give_log = true) ≈ log(0.5) + log(exp(1.5 * ub) - exp(1.5 * lb))
    
    @test MultistateModels.call_cumulhaz(lb, ub, msm_gom.parameters[1], subjdat_row, msm_gom.hazards[1]; give_log = false) ≈ exp(log(0.5) + log(exp(1.5 * ub) - exp(1.5 * lb)))

   # now with covariate adjustment
   subjdat_row2 = msm_gom.data[2, :]
   @test MultistateModels.call_cumulhaz(lb, ub, msm_gom.parameters[2], subjdat_row2, msm_gom.hazards[2]; give_log = true) ≈ log(0.5) + 1.5 + log(exp(0.5 * ub) - exp(0.5 * lb))

    @test MultistateModels.call_cumulhaz(lb, ub, msm_gom.parameters[2], subjdat_row2, msm_gom.hazards[2]; give_log = false) ≈ exp(log(0.5) + 1.5 + log(exp(0.5 * ub) - exp(0.5 * lb)))
end

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
    λ0 = 0.4
    β = 0.7
    interval = (0.0, 1.25)
    linear_pred = β * dat.x[1]

    h_exp_ph = Hazard(@formula(0 ~ x), "exp", 1, 2; linpred_effect = :ph)
    h_exp_aft = Hazard(@formula(0 ~ x), "exp", 1, 2; linpred_effect = :aft)
    model_exp_ph = multistatemodel(h_exp_ph; data = dat)
    model_exp_aft = multistatemodel(h_exp_aft; data = dat)
    MultistateModels.set_parameters!(model_exp_ph, (h12 = [log(λ0), β],))
    MultistateModels.set_parameters!(model_exp_aft, (h12 = [log(λ0), β],))

    rate_ph = λ0 * exp(linear_pred)
    rate_aft = λ0 * exp(-linear_pred)
    @test MultistateModels.call_haz(0.5, model_exp_ph.parameters[1], subjdat_row, model_exp_ph.hazards[1]; give_log = false) ≈ rate_ph
    @test MultistateModels.call_haz(0.5, model_exp_aft.parameters[1], subjdat_row, model_exp_aft.hazards[1]; give_log = false) ≈ rate_aft
    @test MultistateModels.call_cumulhaz(interval[1], interval[2], model_exp_ph.parameters[1], subjdat_row, model_exp_ph.hazards[1]; give_log = false) ≈ rate_ph * (interval[2] - interval[1])
    @test MultistateModels.call_cumulhaz(interval[1], interval[2], model_exp_aft.parameters[1], subjdat_row, model_exp_aft.hazards[1]; give_log = false) ≈ rate_aft * (interval[2] - interval[1])

    log_shape = log(1.4)
    log_scale = log(0.6)
    β_w = -0.3
    h_wei_ph = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect = :ph)
    h_wei_aft = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect = :aft)
    model_wei_ph = multistatemodel(h_wei_ph; data = dat)
    model_wei_aft = multistatemodel(h_wei_aft; data = dat)
    MultistateModels.set_parameters!(model_wei_ph, (h12 = [log_shape, log_scale, β_w],))
    MultistateModels.set_parameters!(model_wei_aft, (h12 = [log_shape, log_scale, β_w],))

    shape = exp(log_shape)
    scale = exp(log_scale)
    lp_w = β_w * dat.x[1]
    t = 1.8
    lb = 0.5
    ub = 1.6
    haz_ph = shape * scale * t^(shape - 1) * exp(lp_w)
    haz_aft = shape * scale * t^(shape - 1) * exp(-shape * lp_w)
    cum_ph = scale * exp(lp_w) * (ub^shape - lb^shape)
    cum_aft = scale * exp(-shape * lp_w) * (ub^shape - lb^shape)

    @test MultistateModels.call_haz(t, model_wei_ph.parameters[1], subjdat_row, model_wei_ph.hazards[1]; give_log = false) ≈ haz_ph
    @test MultistateModels.call_haz(t, model_wei_aft.parameters[1], subjdat_row, model_wei_aft.hazards[1]; give_log = false) ≈ haz_aft
    @test MultistateModels.call_cumulhaz(lb, ub, model_wei_ph.parameters[1], subjdat_row, model_wei_ph.hazards[1]; give_log = false) ≈ cum_ph
    @test MultistateModels.call_cumulhaz(lb, ub, model_wei_aft.parameters[1], subjdat_row, model_wei_aft.hazards[1]; give_log = false) ≈ cum_aft

    log_shape_g = log(0.2)
    log_scale_g = log(0.3)
    β_g = 0.5
    h_gom_ph = Hazard(@formula(0 ~ x), "gom", 1, 2; linpred_effect = :ph)
    h_gom_aft = Hazard(@formula(0 ~ x), "gom", 1, 2; linpred_effect = :aft)
    model_gom_ph = multistatemodel(h_gom_ph; data = dat)
    model_gom_aft = multistatemodel(h_gom_aft; data = dat)
    MultistateModels.set_parameters!(model_gom_ph, (h12 = [log_shape_g, log_scale_g, β_g],))
    MultistateModels.set_parameters!(model_gom_aft, (h12 = [log_shape_g, log_scale_g, β_g],))

    shape_g = exp(log_shape_g)
    scale_g = exp(log_scale_g)
    lp_g = β_g * dat.x[1]
    t_g = 1.1
    lb_g = 0.2
    ub_g = 1.4
    haz_g_ph = scale_g * shape_g * exp(shape_g * t_g + lp_g)
    s = exp(-lp_g)
    haz_g_aft = scale_g * shape_g * exp(shape_g * t_g * s) * s
    cum_g_ph = begin
        base = if abs(shape_g) < 1e-10
            scale_g * (ub_g - lb_g)
        else
            scale_g * (exp(shape_g * ub_g) - exp(shape_g * lb_g))
        end
        base * exp(lp_g)
    end
    cum_g_aft = begin
        scaled_shape = shape_g * s
        if abs(scaled_shape) < 1e-10
            scale_g * s * (ub_g - lb_g)
        else
            scale_g * (exp(scaled_shape * ub_g) - exp(scaled_shape * lb_g))
        end
    end

    @test MultistateModels.call_haz(t_g, model_gom_ph.parameters[1], subjdat_row, model_gom_ph.hazards[1]; give_log = false) ≈ haz_g_ph
    @test MultistateModels.call_haz(t_g, model_gom_aft.parameters[1], subjdat_row, model_gom_aft.hazards[1]; give_log = false) ≈ haz_g_aft
    @test MultistateModels.call_cumulhaz(lb_g, ub_g, model_gom_ph.parameters[1], subjdat_row, model_gom_ph.hazards[1]; give_log = false) ≈ cum_g_ph
    @test MultistateModels.call_cumulhaz(lb_g, ub_g, model_gom_aft.parameters[1], subjdat_row, model_gom_aft.hazards[1]; give_log = false) ≈ cum_g_aft
end

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
    λ0 = 0.4
    β = 0.7
    MultistateModels.set_parameters!(model_exp_tt, (h12 = [log(λ0), β],))
    rate_exp = λ0 * exp(β * subjdat_row.x)
    t_eval = 0.8
    interval = (0.2, 1.1)
    @test MultistateModels.call_haz(t_eval, model_exp_tt.parameters[1], subjdat_row, model_exp_tt.hazards[1]; give_log = false, apply_transform = false) ≈ rate_exp
    @test MultistateModels.call_haz(t_eval, model_exp_tt.parameters[1], subjdat_row, model_exp_tt.hazards[1]; give_log = false, apply_transform = true) ≈ rate_exp
    cum_exp = rate_exp * (interval[2] - interval[1])
    @test MultistateModels.call_cumulhaz(interval[1], interval[2], model_exp_tt.parameters[1], subjdat_row, model_exp_tt.hazards[1]; give_log = false, apply_transform = true) ≈ cum_exp
    @test MultistateModels.survprob(interval[1], interval[2], model_exp_tt.parameters, subjdat_row, model_exp_tt.totalhazards[1], model_exp_tt.hazards; give_log = false, apply_transform = true) ≈ exp(-cum_exp)

    # Weibull PH hazard with Tang toggle
    h_wei_tt = Hazard(@formula(0 ~ x), "wei", 1, 2; time_transform = true)
    model_wei_tt = multistatemodel(h_wei_tt; data = dat)
    log_shape = log(1.3)
    log_scale = log(0.6)
    β_w = -0.4
    MultistateModels.set_parameters!(model_wei_tt, (h12 = [log_shape, log_scale, β_w],))
    shape = exp(log_shape)
    scale = exp(log_scale)
    lp_w = β_w * subjdat_row.x
    t_w = 1.25
    lb_w, ub_w = 0.3, 1.6
    haz_w = scale * shape * t_w^(shape - 1) * exp(lp_w)
    cum_w = scale * exp(lp_w) * (ub_w^shape - lb_w^shape)
    @test MultistateModels.call_haz(t_w, model_wei_tt.parameters[1], subjdat_row, model_wei_tt.hazards[1]; give_log = false, apply_transform = true) ≈ haz_w
    @test MultistateModels.call_cumulhaz(lb_w, ub_w, model_wei_tt.parameters[1], subjdat_row, model_wei_tt.hazards[1]; give_log = false, apply_transform = true) ≈ cum_w

    # Gompertz PH hazard with Tang toggle
    h_gom_tt = Hazard(@formula(0 ~ x), "gom", 1, 2; time_transform = true)
    model_gom_tt = multistatemodel(h_gom_tt; data = dat)
    log_shape_g = log(0.5)
    log_scale_g = log(0.8)
    β_g = 0.25
    MultistateModels.set_parameters!(model_gom_tt, (h12 = [log_shape_g, log_scale_g, β_g],))
    shape_g = exp(log_shape_g)
    scale_g = exp(log_scale_g)
    lp_g = β_g * subjdat_row.x
    t_g = 0.9
    lb_g, ub_g = 0.1, 1.4
    haz_g = scale_g * shape_g * exp(shape_g * t_g + lp_g)
    cum_g = scale_g * exp(lp_g) * (exp(shape_g * ub_g) - exp(shape_g * lb_g))
    @test MultistateModels.call_haz(t_g, model_gom_tt.parameters[1], subjdat_row, model_gom_tt.hazards[1]; give_log = false, apply_transform = true) ≈ haz_g
    @test MultistateModels.call_cumulhaz(lb_g, ub_g, model_gom_tt.parameters[1], subjdat_row, model_gom_tt.hazards[1]; give_log = false, apply_transform = true) ≈ cum_g

    # Cache ownership: shared-baseline table per origin state/family
    h_cache = Hazard(@formula(0 ~ x), "exp", 1, 2; time_transform = true)
    model_cache = multistatemodel(h_cache; data = dat)
    MultistateModels.set_parameters!(model_cache, (h12 = [log(0.5), 0.3],))
    pars_cache = copy(model_cache.parameters[1])
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

    pars_cum = copy(model_cache.parameters[1])
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
    pars12 = copy(model_shared.parameters[1])
    pars13 = copy(model_shared.parameters[2])
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
    ctx_auto = MultistateModels.maybe_time_transform_context(model_shared.parameters, subj_df, model_shared.hazards)
    @test ctx_auto isa MultistateModels.TimeTransformContext
    plain_hazard = Hazard(@formula(0 ~ x), "exp", 1, 2)
    plain_model = multistatemodel(plain_hazard; data = dat)
    ctx_none = MultistateModels.maybe_time_transform_context(plain_model.parameters, subj_df, plain_model.hazards)
    @test ctx_none === nothing
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
