# Tests for Hazard and _hazard structs and call_haz methods
# 1. Check accuracy of hazards, cumulative hazards, total hazard
#       - Check for non Float64 stuff
#       - Edge cases? Zero hazard, infinite hazard, negative hazard (should throw error)
#       - Test for numerical problems (see Distributions.jl for ideas)
# 2. function to validate data
# 3. validate MultistateModel object    

msm = multistatemodel(h12, h23, h13; data = dat_exact2)

# function that enumerates the hazards
@testset "test_enumerate_hazards" begin
    @test nrow(hazinfo) == length(hazards)

    # check that primary order is by state from
    @test all(hazinfo.statefrom .== sort(hazinfo.statefrom))
    
    # and then that secondary order is state to within state from
    sorted_stateto = 
        @chain hazinfo begin
            groupby(:statefrom)
            combine(:stateto => (x -> (Bool(all(x .== sort(x))))) => :sorted)
        end

    @test all(sorted_stateto.sorted .== 1)

    # double check that order of hazards corresponds to hazinfo
    for h in eachindex(hazards)
        @test hazards[h].statefrom == hazinfo.statefrom[h]
        @test hazards[h].stateto == hazinfo.stateto[h]
    end
end

# tests for individual hazards
@testset "test_hazards_for_numerical_issues" begin
    @test 1 == 1
end

# tests for total hazards

# basic structure of a testset
@testset "arithmetic" begin
    @test 1+1 == 2
end