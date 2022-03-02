# Test that parameter views propagate to cause-specific hazards

@testset "test_hazpar_views" begin
    
    # check that values set in the collated parameters vector propagate to cause-specific hazards
    vals1 = randn(length(msm_expwei.parameters))

    copyto!(msm_expwei.parameters, vals1)
A
    @test msm_expwei.hazards[1].parameters == msm_expwei.parameters[1] 
    @test all(msm_expwei.hazards[2].parameters .== msm_expwei.parameters[2:5])
end

# Test parameter setting function

@testset "test_set_parameters!" begin

    vals = randn(length(msm_expwei.parameters))

    set_parameters!(msm_expwei, vals)

    @test msm_expwei.hazards[1].parameters .== vals[1]
    @test all(msm_expwei.hazards[2].parameters .== vals[2:5])
    @test all(msm_expwei.parameters .== vals)

    val_tuple = (randn(1), randn(4), randn(2), randn(4))
    set_parameters!(msm_expwei, val_tuple)

    @test msm_expwei.hazards[1].parameters == val_tuple[1]
    @test all(msm_expwei.hazards[2].parameters .== val_tuple[2])
    @test all(msm_expwei.hazards[3].parameters .== val_tuple[3])
    @test all(msm_expwei.hazards[4].parameters .== val_tuple[4])
end

# Test function for converting vector of subject IDs to vector of vector of indices 

@testset "test_get_subjinds" begin
    
    sidv = [1, 2, 2, 3, 3, 3, 42, 42]
    sidvv = [[1], [2, 3], [4, 5, 6], [7, 8]]

    @test get_subjinds(DataFrame(id = sidv)) .== sidvv

end