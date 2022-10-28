# Tests for Hazard and _hazard structs and call_haz methods
# 1. Check accuracy of hazards, cumulative hazards, total hazard
#       - Check for non Float64 stuff
#       - Edge cases? Zero hazard, infinite hazard, negative hazard (should throw error)
#       - Test for numerical problems (see Distributions.jl for ideas)
# 2. function to validate data
# 3. validate MultistateModel object    

# validate the transition matrix
@testset "test_tmat" begin
    
    # check that primary order is by origin state
    # and that secondary order is by destination
    @test sort(msm_expwei.tmat[[2,4,7,8]]) == collect(1:4)
    @test all(msm_expwei.tmat[Not([2,4,7,8])] .== 0)

    # need a test for subject indices

end
