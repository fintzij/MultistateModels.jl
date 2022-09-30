# Test parameter setting function
@testset "test_set_parameters!" begin

    vals = randn(length(msm_expwei.parameters))

    # vector
    set_parameters!(msm_expwei, vals)

    @test msm_expwei.hazards[1].parameters[1] .== vals[1]
    @test all(msm_expwei.hazards[2].parameters .== vals[2:5])
    @test all(msm_expwei.parameters .== vals)

    # unnamed tuple
    val_tuple = (randn(1), randn(4), randn(2), randn(4))
    set_parameters!(msm_expwei, val_tuple)

    @test msm_expwei.hazards[1].parameters[1] == val_tuple[1][1]
    @test all(msm_expwei.hazards[2].parameters .== val_tuple[2])
    @test all(msm_expwei.hazards[3].parameters .== val_tuple[3])
    @test all(msm_expwei.hazards[4].parameters .== val_tuple[4])

    # named tuple
    val_tuple = (h12 = randn(1), h13 = randn(4), h21 = randn(2), h23 = randn(4))
    set_parameters!(msm_expwei, val_tuple)

    @test msm_expwei.hazards[1].parameters[1] == val_tuple[1][1]
    @test all(msm_expwei.hazards[2].parameters .== val_tuple[2])
    @test all(msm_expwei.hazards[3].parameters .== val_tuple[3])
    @test all(msm_expwei.hazards[4].parameters .== val_tuple[4])
end

# Test function for converting vector of subject IDs to vector of vector of indices 

@testset "test_get_subjinds" begin
    
    sidv = [1, 2, 2, 3, 3, 3, 42, 42]
    sidvv = [[1], [2, 3], [4, 5, 6], [7, 8]]

    @test MultistateModels.get_subjinds(DataFrame(id = sidv)) == sidvv

end