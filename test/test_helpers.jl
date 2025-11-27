# =============================================================================
# Helper Utility Tests
# =============================================================================
#
# Exercises low-level helper functions that mutation-heavy code paths depend on.
# Shared fixtures provide reusable datasets for helper checks so each testset
# isolates a single helper.
using .TestFixtures

# --- Parameter setters ----------------------------------------------------------
@testset "test_set_parameters!" begin

    # vector
    vec_vals = [randn(length(msm_expwei.parameters[1])),
                randn(length(msm_expwei.parameters[2])),
                randn(length(msm_expwei.parameters[3])),
                randn(length(msm_expwei.parameters[4]))]
    set_parameters!(msm_expwei, vec_vals)

    @test msm_expwei.parameters[1] == vec_vals[1]
    @test all(msm_expwei.parameters[2] .== vec_vals[2])
    @test all(msm_expwei.parameters[3] .== vec_vals[3])
    @test all(msm_expwei.parameters[4] .== vec_vals[4])
    
    # unnamed tuple
    unnamed_tuple = (randn(1), randn(4), randn(2), randn(3))
    set_parameters!(msm_expwei, unnamed_tuple)

    @test msm_expwei.parameters[1] == unnamed_tuple[1]
    @test all(msm_expwei.parameters[2] .== unnamed_tuple[2])
    @test all(msm_expwei.parameters[3] .== unnamed_tuple[3])
    @test all(msm_expwei.parameters[4] .== unnamed_tuple[4])

    # named tuple
    named_tuple = (h12 = randn(1), h13 = randn(4), h21 = randn(2), h23 = randn(3))
    set_parameters!(msm_expwei, named_tuple)

    @test msm_expwei.parameters[1] == named_tuple[1]
    @test all(msm_expwei.parameters[2] .== named_tuple[2])
    @test all(msm_expwei.parameters[3] .== named_tuple[3])
    @test all(msm_expwei.parameters[4] .== named_tuple[4])
end

# --- Subject index construction -------------------------------------------------
@testset "test_get_subjinds" begin
    sid_df = subject_id_df()
    expected_groups = subject_id_groups()

    subjinds, nsubj = MultistateModels.get_subjinds(sid_df)

    @test subjinds == expected_groups
    @test nsubj == length(expected_groups)
end