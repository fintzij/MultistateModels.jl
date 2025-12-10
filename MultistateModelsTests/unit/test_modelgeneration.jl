# =============================================================================
# Model Generation Tests
# =============================================================================
#
# Tests user-facing error handling that prevents silent incorrect behavior.

using .TestFixtures

# --- Duplicate transition detection ---------------------------------------------
@testset "test_duplicate_transitions" begin
    # Duplicate transitions would cause silent likelihood errors
    dat = duplicate_transition_data()
    h1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h2 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    @test_throws ErrorException multistatemodel(h1, h2; data = dat)
end

# --- State 0 handling -----------------------------------------------------------
@testset "test_state_zero_in_data" begin
    # Validates censoring pattern handling works correctly
    dat = censoring_panel_data()
    censoring_patterns = Int64[3 0 1 1]
    h1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h2 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    model = multistatemodel(h1, h2; data = dat, CensoringPatterns = censoring_patterns)
    @test isa(model, MultistateModels.MultistateProcess)
end

# --- Hazard boundaries ----------------------------------------------------------
@testset "test_hazard_state_zero" begin
    # Guards against invalid hazard construction
    @test_throws AssertionError Hazard(@formula(0 ~ 1), "exp", 0, 1)
    @test_throws AssertionError Hazard(@formula(0 ~ 1), "exp", 1, 0)
    @test_throws AssertionError Hazard(@formula(0 ~ 1), "exp", 1, 1)
end
