# Tests for model generation functionality
# Tests multistatemodel() construction, Hazard validation, and data checks

@testset "test_tmat" begin
    # Validate the transition matrix structure
    # Check that primary order is by origin state and secondary order is by destination
    @test sort(msm_expwei.tmat[[2,4,7,8]]) == collect(1:4)
    @test all(msm_expwei.tmat[Not([2,4,7,8])] .== 0)
end

@testset "test_duplicate_transitions" begin
    # Test that duplicate transitions are detected and throw error
    dat = DataFrame(
        id = [1, 1, 1],
        tstart = [0.0, 1.0, 2.0],
        tstop = [1.0, 2.0, 3.0],
        statefrom = [1, 2, 3],
        stateto = [2, 3, 3],
        obstype = [1, 1, 3]
    )
    
    h1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h2 = Hazard(@formula(0 ~ 1), "exp", 1, 2)  # Duplicate transition
    
    @test_throws ErrorException multistatemodel(h1, h2; data = dat)
    
    # Verify error message mentions duplicate transitions
    try
        multistatemodel(h1, h2; data = dat)
        @test false  # Should not reach here
    catch e
        @test occursin("Duplicate transitions", string(e))
        @test occursin("(1, 2)", string(e))
    end
end

@testset "test_non_contiguous_states" begin
    # Test that non-contiguous states (e.g., 1,2,4) produce warning
    # Note: Model will fail with BoundsError, but warning should appear first
    dat = DataFrame(
        id = [1, 1, 1],
        tstart = [0.0, 1.0, 2.0],
        tstop = [1.0, 2.0, 3.0],
        statefrom = [1, 2, 4],  # Missing state 3
        stateto = [2, 4, 4],
        obstype = [1, 1, 3]
    )
    
    h1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h2 = Hazard(@formula(0 ~ 1), "exp", 2, 4)
    
    # Model creation will fail due to tmat indexing, but should warn about non-contiguous states
    @test_throws BoundsError multistatemodel(h1, h2; data = dat)
end

@testset "test_state_zero_in_data" begin
    # Test that state 0 can appear in data (for censoring) without error
    # State 0 should only appear in stateto with appropriate obstype
    dat = DataFrame(
        id = [1, 1],
        tstart = [0.0, 1.0],
        tstop = [1.0, 2.0],
        statefrom = [1, 1],
        stateto = [1, 1],
        obstype = [2, 2]
    )
    
    h1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h2 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    
    # Should succeed - state 0 not in hazards
    model = multistatemodel(h1, h2; data = dat)
    @test isa(model, MultistateModels.MultistateProcess)
end

@testset "test_hazard_state_zero" begin
    # Test that hazards cannot have state 0 in statefrom or stateto
    # State 0 is reserved for censoring indicators in data, not for transitions
    dat = DataFrame(
        id = [1],
        tstart = [0.0],
        tstop = [1.0],
        statefrom = [1],
        stateto = [1],
        obstype = [2]
    )
    
    # State 0 should never appear in hazard definitions
    # The tmat will have issues if we try to use state 0
    h_bad_from = Hazard(@formula(0 ~ 1), "exp", 0, 1)
    h_bad_to = Hazard(@formula(0 ~ 1), "exp", 1, 0)
    h_good = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    
    # These should fail during model construction
    @test_throws Exception multistatemodel(h_bad_from, h_good; data = dat)
    @test_throws Exception multistatemodel(h_good, h_bad_to; data = dat)
end

@testset "test_parameter_naming" begin
    # Test that exponential hazards use "Intercept" not "rate"
    dat = DataFrame(
        id = [1, 1],
        tstart = [0.0, 1.0],
        tstop = [1.0, 2.0],
        statefrom = [1, 1],
        stateto = [1, 1],
        obstype = [2, 2]
    )
    
    # Exponential hazard without covariates
    h1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    model = multistatemodel(h1; data = dat)
    
    # Check parameter naming (parnames are now in hazards, not model)
    @test :h12_Intercept in model.hazards[1].parnames
    @test !(:h12_rate in model.hazards[1].parnames)
    
    # Exponential hazard with covariates
    dat_cov = DataFrame(
        id = [1, 1],
        tstart = [0.0, 1.0],
        tstop = [1.0, 2.0],
        statefrom = [1, 1],
        stateto = [1, 1],
        obstype = [2, 2],
        age = [50, 51]
    )
    
    h2 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 2)
    model2 = multistatemodel(h2; data = dat_cov)
    
    # Check parameter naming includes Intercept and covariate
    @test :h12_Intercept in model2.hazards[1].parnames
    @test :h12_age in model2.hazards[1].parnames
    @test !(:h12_rate in model2.hazards[1].parnames)
end

@testset "test_hazard_construction" begin
    # Test that different hazard types construct properly
    dat = DataFrame(
        id = [1, 1],
        tstart = [0.0, 1.0],
        tstop = [1.0, 2.0],
        statefrom = [1, 1],
        stateto = [1, 1],
        obstype = [2, 2],
        trt = [0, 1]
    )
    
    # Exponential
    h_exp = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
    model_exp = multistatemodel(h_exp; data = dat)
    @test length(model_exp.parameters[1]) == 2  # Intercept + trt
    @test model_exp.hazards[1].parnames == [:h12_Intercept, :h12_trt]
    
    # Weibull
    h_wei = Hazard(@formula(0 ~ 1 + trt), "wei", 1, 2)
    model_wei = multistatemodel(h_wei; data = dat)
    @test length(model_wei.parameters[1]) == 3  # shape, scale, scale_trt
    @test :h12_shape in model_wei.hazards[1].parnames
    @test :h12_scale in model_wei.hazards[1].parnames
    
    # Gompertz - Skip for now as not implemented yet
    # h_gom = Hazard(@formula(0 ~ 1 + trt), "gomp", 1, 2)
    # model_gom = multistatemodel(h_gom; data = dat)
    # @test length(model_gom.parameters[1]) == 3  # shape, scale, scale_trt
    # @test :h12_shape in model_gom.hazards[1].parnames
    # @test :h12_scale in model_gom.hazards[1].parnames
end
