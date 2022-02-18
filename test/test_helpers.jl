# Test that parameter views propagate to cause-specific hazards

@testset "test_hazpar_views" begin
    
    # check that values set in the collated parameters vector propagate to cause-specific hazards
    vals1 = randn(length(msm_expwei.parameters))

    copyto!(msm_expwei.parameters, vals1)
A
    @test msm_expwei.hazards[1].parameters .== msm_expwei.parameters[1] 
    @test all(msm_expwei.hazards[2].parameters .== [1.1, 1.2, 1.3, 1.4])
end