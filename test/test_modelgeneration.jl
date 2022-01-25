# Tests for Hazard and _hazard structs and call_haz methods
# 1. Check accuracy of hazards, cumulative hazards, total hazard
#       - Check for non Float64 stuff
#       - Edge cases? Zero hazard, infinite hazard, negative hazard (should throw error)
#       - Test for numerical problems (see Distributions.jl for ideas)
# 2. function to validate data
# 3. validate MultistateModel object    

msm = multistatemodel(h12, h23, h13, h21; data = dat_exact2)

# validate the transition matrix
@testset "test_tmat" begin
    
    # check that primary order is by origin state
    # and that secondary order is by destination
    @test sort(msm.tmat[[2,4,7,8]]) == collect(1:4)
    @test all(msm.tmat[Not([2,4,7,8])] .== 0)

end

# tests for individual hazards
@testset "test_hazards_exp" begin
    
    # set parameters, no covariate adjustment
    msm.hazards[1].parameters[1] = 0.8

    @test MultistateModels.call_haz(0.0, 0, msm.hazards[1]; give_log = true) == 0.8

    @test MultistateModels.call_haz(0.0, 0, msm.hazards[1]; give_log = false) == exp(0.8)

    # set parameters, exponential with covariate adjustment
    pars = [0.0, 0.6, -0.4, 0.15]
    
    msm.hazards[2].parameters[1:4] = pars

    # correct hazard value on log scale, for each row of data
    truevals = 
        [dat_exact2.trt[1] * pars[2] + dat_exact2.age[1] * pars[3] + 
        dat_exact2.trt[1] * dat_exact2.age[1] * pars[4],
        dat_exact2.trt[2] * pars[2] + dat_exact2.age[2] * pars[3] + 
        dat_exact2.trt[2] * dat_exact2.age[2] * pars[4],
        dat_exact2.trt[3] * pars[2] + dat_exact2.age[3] * pars[3] + 
        dat_exact2.trt[3] * dat_exact2.age[3] * pars[4]]
    
    # loop through each row of data embedded in the msm object
    for h in axes(msm.data, 1)
        @test MultistateModels.call_haz(0.0, h, msm.hazards[2]; give_log = true) == 
            truevals[h]

        @test MultistateModels.call_haz(0.0, h, msm.hazards[2]; give_log = false) == exp(truevals[h])
    end
end


@testset "test_hazards_weibull" begin

    # set parameters, log(scale, shape), no covariate adjustment
    msm.hazards[3].parameters[1:2] = [0.2, -0.25]

    # h(t) = shape * scale^shape * t^(shape-1)
    @test MultistateModels.call_haz(1.0, 0, msm.hazards[3]; give_log = true) == -0.25 + exp(-0.25) * 0.2

    @test MultistateModels.call_haz(1.0, 0, msm.hazards[3]; give_log = false) == exp(-0.25 + exp(-0.25) * 0.2)

    # set parameters, log(scale_intercept, scale_trt, shape_intercept, shape_trt) weibull with covariate adjustment
    # also set time at which to check hazard for correctness
    pars = [0.2, 0.25, -0.3, 0.25]
    t = 1.0
    msm.hazards[4].parameters[1:4] = pars

    # loop through each row of data embedded in the msm object, comparing truth to Multistatemodels.call_haz output
    for h in axes(msm.data, 1)
        log_scale = 
            pars[1] + pars[2]*dat_exact2.trt[h]
        log_shape = 
            pars[3] + pars[4]*dat_exact2.trt[h]
            
        true_val = log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t)
        
        @test MultistateModels.call_haz(1.0, h, msm.hazards[4]; give_log=true) == true_val

        @test MultistateModels.call_haz(1.0, h, msm.hazards[4]; give_log=false) == exp(true_val)
    end
    
end

@testset "test_totalhazards" begin

    # set parameters
    msm.hazards[1].parameters[1] = 0.8
    msm.hazards[2].parameters[1:4] = [0.8, 0.6, -0.4, 0.15]
    msm.hazards[3].parameters[1:2] = [0.8, 1.2]
    msm.hazards[4].parameters[1:4] = [0.8, 0.25, 1.2, 0.5]

    # test total hazard for each origin state    
    for h in axes(msm.data, 1) 
        for s in axes(msm.tmat, 1) 

            if s == 1
                tot_haz = 
                    MultistateModels.call_haz(1.0, h, msm.hazards[1]; give_log = false) + MultistateModels.call_haz(1.0, h, msm.hazards[2]; give_log = false)
                
            elseif s == 2
                tot_haz = 
                    MultistateModels.call_haz(1.0, h, msm.hazards[3]; give_log = false) + MultistateModels.call_haz(1.0, h, msm.hazards[4]; give_log = false)
            else
                tot_haz = 0.0
            end            

            @test MultistateModels.call_tothaz(1.0, h, msm.totalhazards[s], msm.hazards; give_log = false) ≈ tot_haz
            
            @test MultistateModels.call_tothaz(1.0, h, msm.totalhazards[s], msm.hazards; give_log = true) ≈ log(tot_haz)
        end
    end
end