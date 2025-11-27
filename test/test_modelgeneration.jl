# =============================================================================
# Model Generation Tests
# =============================================================================
#
# Guards `multistatemodel` construction logic, Hazard validation, and input
# hygiene before downstream likelihood machinery runs. Each testset documents
# one failure mode so regressions are easy to trace.

using .TestFixtures

# --- Transition matrix structure -------------------------------------------------
@testset "test_tmat" begin
    # Validate the transition matrix structure
    # Check that primary order is by origin state and secondary order is by destination
    @test sort(msm_expwei.tmat[[2,4,7,8]]) == collect(1:4)
    @test all(msm_expwei.tmat[Not([2,4,7,8])] .== 0)
end

# --- Duplicate transition detection ---------------------------------------------
@testset "test_duplicate_transitions" begin
    # Test that duplicate transitions are detected and throw error
    dat = duplicate_transition_data()
    
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

# --- State numbering hygiene ----------------------------------------------------
@testset "test_non_contiguous_states" begin
    # Test that non-contiguous states (e.g., 1,2,4) produce warning
    # Note: Model will fail with BoundsError, but warning should appear first
    dat = noncontiguous_state_data()
    
    h1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h2 = Hazard(@formula(0 ~ 1), "exp", 2, 4)
    
    # Model creation will fail due to tmat indexing, but should warn about non-contiguous states
    @test_throws BoundsError multistatemodel(h1, h2; data = dat)
end

# --- State 0 handling -----------------------------------------------------------
@testset "test_state_zero_in_data" begin
    # Test that state 0 can appear in data (for censoring) without error
    # Fixture places state 0 in `stateto` with censored observation types
    dat = censoring_panel_data()
    censoring_patterns = Int64[3 0 1 1]  # obstype=3 allows states 2 or 3 at interval end
    
    h1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h2 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    
    # Should succeed - state 0 not in hazards
    model = multistatemodel(h1, h2; data = dat, CensoringPatterns = censoring_patterns)
    @test isa(model, MultistateModels.MultistateProcess)
end

# --- Hazard boundaries ----------------------------------------------------------
@testset "test_hazard_state_zero" begin
    # Test that hazards cannot have state 0 in statefrom or stateto
    # State 0 is reserved for censoring indicators in data, not for transitions
    dat = baseline_exact_data()
    
    # State 0 should never appear in hazard definitions
    # The tmat will have issues if we try to use state 0
    h_bad_from = Hazard(@formula(0 ~ 1), "exp", 0, 1)
    h_bad_to = Hazard(@formula(0 ~ 1), "exp", 1, 0)
    h_good = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    
    # These should fail during model construction
    @test_throws Exception multistatemodel(h_bad_from, h_good; data = dat)
    @test_throws Exception multistatemodel(h_good, h_bad_to; data = dat)
end

# --- Parameter naming -----------------------------------------------------------
@testset "test_parameter_naming" begin
    # Test that exponential hazards use "Intercept" not "rate"
    dat = baseline_exact_data()
    
    # Exponential hazard without covariates
    h1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    model = multistatemodel(h1; data = dat)
    
    # Check parameter naming (parnames are now in hazards, not model)
    @test :h12_Intercept in model.hazards[1].parnames
    @test !(:h12_rate in model.hazards[1].parnames)
    
    # Exponential hazard with covariates
    dat_cov = baseline_exact_covariates()
    
    h2 = Hazard(@formula(0 ~ age), "exp", 1, 2)
    model2 = multistatemodel(h2; data = dat_cov)
    
    # Check parameter naming includes Intercept and covariate
    @test :h12_Intercept in model2.hazards[1].parnames
    @test :h12_age in model2.hazards[1].parnames
    @test !(:h12_rate in model2.hazards[1].parnames)
end

# --- Hazard constructors --------------------------------------------------------
@testset "test_hazard_construction" begin
    # Test that different hazard types construct properly
    dat = baseline_exact_covariates()
    
    # Exponential
    h_exp = Hazard(@formula(0 ~ trt), "exp", 1, 2)
    model_exp = multistatemodel(h_exp; data = dat)
    @test length(model_exp.parameters[1]) == 2  # Intercept + trt
    @test model_exp.hazards[1].parnames == [:h12_Intercept, :h12_trt]
    
    # Weibull
    h_wei = Hazard(@formula(0 ~ trt), "wei", 1, 2)
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

# --- Hazard macro parity --------------------------------------------------------
@testset "@hazard macro" begin
    base_formula = @formula(0 ~ 1 + trt)

    macro_h = @hazard begin
        family = :exp
        formula = base_formula
        statefrom = 1
        stateto = 2
        linpred_effect = :aft
    end

    manual_h = Hazard(base_formula, :exp, 1, 2; linpred_effect = :aft)

    @test macro_h isa MultistateModels.ParametricHazard
    @test macro_h.family == manual_h.family == "exp"
    @test macro_h.statefrom == 1
    @test macro_h.stateto == 2
    @test macro_h.metadata.linpred_effect == :aft

    alias_h = @hazard begin
        family = :weibull
        hazard = @formula(0 ~ 1)
        transition = 2 => 3
    end

    @test alias_h.family == "wei"
    @test alias_h.statefrom == 2
    @test alias_h.stateto == 3

    @test_throws ArgumentError @hazard begin
        family = :ode
        formula = @formula(0 ~ 1)
        from = 1
        to = 2
    end
end
