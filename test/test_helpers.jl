# Test parameter setting function
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

# Test function for converting vector of subject IDs to vector of vector of indices 

@testset "test_get_subjinds" begin
    
    sidv = [1, 2, 2, 3, 3, 3, 42, 42]
    sidvv = [[1], [2, 3], [4, 5, 6], [7, 8]]

    @test MultistateModels.get_subjinds(DataFrame(id = sidv)) == sidvv

end