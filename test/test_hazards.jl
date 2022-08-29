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
    interval_incid = 
        1 - MultistateModels.survprob(0.0, 2.0, msm_expwei.parameters, 1, msm_expwei.totalhazards[1], msm_expwei.hazards)
        
    # note that Distributions.jl uses the mean parameterization
    # i.e., 1/rate. 
    @test cdf(Exponential(0.5), 2) ≈ interval_incid
end

# tests for individual hazards
@testset "test_hazards_exp" begin
    
    # create a parameters object
    msm_expwei.parameters[1] = [0.8,]
    msm_expwei.parameters[2] = [0.0, 0.6, -0.4, 0.15]

    # exponential hazards, no covariate adjustment
    @test isa(msm_expwei.hazards[1], MultistateModels._Exponential)
    @test MultistateModels.call_haz(0.0, msm_expwei.parameters[1], 1, msm_expwei.hazards[1]; give_log = true) == 0.8
    @test MultistateModels.call_haz(0.0, msm_expwei.parameters[1], 1, msm_expwei.hazards[1]; give_log = false) == exp(0.8)


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
        @test MultistateModels.call_haz(0.0,  msm_expwei.parameters[2], h, msm_expwei.hazards[2]; give_log = true) == trueval[h]

        @test MultistateModels.call_haz(0.0, msm_expwei.parameters[2], h, msm_expwei.hazards[2]; give_log = false) == exp(trueval[h])
    end
end

@testset "test_hazards_weibull" begin

    # set parameters, log(shape, scale), no covariate adjustment
    msm_expwei.parameters[3] = [-0.25, 0.2]

    # h(t) = shape * scale^shape * t^(shape-1)
    @test MultistateModels.call_haz(1.0, msm_expwei.parameters[3], 1, msm_expwei.hazards[3]; give_log = true) == exp(-0.25) * 0.2 - 0.25

    @test MultistateModels.call_haz(1.0, msm_expwei.parameters[3], 1, msm_expwei.hazards[3]; give_log = false) == exp(-0.25 + exp(-0.25) * 0.2)

    # set parameters, log(shape_intercept, scale_intercept, scale_trt) weibull PH with covariate adjustment
    # also set time at which to check hazard for correctness
    pars = [0.2, 0.25, -0.3]
    t = 1.0
    msm_expwei.parameters[4] = pars

    # loop through each row of data embedded in the msm_expwei object, comparing truth to MultistateModels.call_haz output
    for h in axes(msm_expwei.data, 1)
                    
        trueval = pars[1] + expm1(pars[1]) * log(t) + dot(msm_expwei.hazards[4].data[h,:], pars[2:3])
                
        @test MultistateModels.call_haz(t, msm_expwei.parameters[4], h, msm_expwei.hazards[4]; give_log=true) == trueval

        @test MultistateModels.call_haz(t, msm_expwei.parameters[4], h, msm_expwei.hazards[4]; give_log=false) == exp(trueval)
    end
end

@testset "test_cumulativehazards_exp" begin
    
    # set parameters, lb (start time), and ub (end time)
    msm_expwei.parameters[1] = [0.8,]
    msm_expwei.parameters[2] = [0.0, 0.6, -0.4, 0.15]
    lb = 0
    ub = 5

    # cumulative hazard for exponential cause specific hazards, no covariate adjustment
    @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[1], 1, msm_expwei.hazards[1], give_log = true) == 0.8 + log(ub-lb)

    # cumulative hazard for exponential proportional hazards over [lb, ub], with covariate adjustment
    pars =  msm_expwei.parameters[2] 
    log_haz = 
        [pars[2]*dat_exact2.trt[1] + pars[3]*dat_exact2.age[1] + pars[4]*dat_exact2.trt[1]*dat_exact2.age[1],
        pars[2]*dat_exact2.trt[2] + pars[3]*dat_exact2.age[2] + pars[4]*dat_exact2.trt[2]*dat_exact2.age[2],
        pars[2]*dat_exact2.trt[1] + pars[3]*dat_exact2.age[3] + pars[4]*dat_exact2.trt[3]*dat_exact2.age[3]]
    
    trueval = log_haz .+ log(ub-lb)

    for h in axes(msm_expwei.data, 1)
        @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[2], h, msm_expwei.hazards[2], give_log = true) == trueval[h]

        @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[2], h, msm_expwei.hazards[2], give_log = false) == exp(trueval[h])
    end
end

@testset "test_cumulativehazards_weibull" begin

    # set up log parameters, lower bound, and upper bound
    msm_expwei.parameters[3] = [-0.25, 0.2]
    msm_expwei.parameters[4] = [0.2, 0.25, -0.3]
    lb = 0
    ub = 5

    # cumulative hazard for weibullcause specific hazards, no covariate adjustment
    @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[3], 1, msm_expwei.hazards[3], give_log = true) == log(ub^exp(-0.25)-lb^exp(-0.25)) + exp(-0.25)*0.2

    @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[3], 1, msm_expwei.hazards[3], give_log = false) == exp(log(ub^exp(-0.25)-lb^exp(-0.25)) + exp(-0.25)*0.2)

    # cumulative hazard for weibull proportional hazards over [lb, ub], with covariate adjustment
    pars =  msm_expwei.parameters[4] 
    
    # loop through each row of data embedded in the msm_expwei object, comparing truth to MultistateModels.call_cumulhaz output
    for h in axes(msm_expwei.data, 1)
                    
        trueval = log(ub^exp(pars[1]) - lb^exp(pars[1])) + dot(msm_expwei.hazards[4].data[h,:], pars[2:3])
                    
        @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[4], h, msm_expwei.hazards[4]; give_log=true) == trueval
    
        @test MultistateModels.call_cumulhaz(lb, ub, msm_expwei.parameters[4], h, msm_expwei.hazards[4]; give_log=false) == exp(trueval)
    end

end

@testset "test_totalcumulativehazards" begin


    # set parameters
    msm_expwei.hazards[1].parameters[1] = 0.8
    msm_expwei.hazards[2].parameters[1:4] = [0.8, 0.6, -0.4, 0.15]
    msm_expwei.hazards[3].parameters[1:2] = [0.8, 1.2]
    msm_expwei.hazards[4].parameters[1:4] = [0.8, 0.25, 1.2, 0.5]

    # test total hazard for each origin state    
    for h in axes(msm_expwei.data, 1) 
        for s in axes(msm_expwei.tmat, 1) 

            if s == 1
                tot_haz = 
                    MultistateModels.call_haz(1.0, h, msm_expwei.hazards[1]; give_log = false) + MultistateModels.call_haz(1.0, h, msm_expwei.hazards[2]; give_log = false)
                
            elseif s == 2
                tot_haz = 
                    MultistateModels.call_haz(1.0, h, msm_expwei.hazards[3]; give_log = false) + MultistateModels.call_haz(1.0, h, msm_expwei.hazards[4]; give_log = false)
            else
                tot_haz = 0.0
            end            

            @test MultistateModels.tothaz(1.0, h, msm_expwei.totalhazards[s], msm_expwei.hazards; give_log = false) ≈ tot_haz
            
            @test MultistateModels.tothaz(1.0, h, msm_expwei.totalhazards[s], msm_expwei.hazards; give_log = true) ≈ log(tot_haz)
        end
    end
end

@testset "test_cumulativehazards" begin

    msm_expwei.hazards[1].parameters[1] = 0.8
    msm_expwei.hazards[2].parameters[1:4] = [0.8, 0.6, -0.4, 0.15]
    msm_expwei.hazards[3].parameters[1:2] = [0.8, 1.2]
    msm_expwei.hazards[4].parameters[1:4] = [0.8, 0.25, 1.2, 0.5]

    MultistateModels.cumulhaz(
        msm_expwei.totalhazards[1], 
        msm_expwei.hazards, 
        0.0,
        1.1,
        1)
    
end