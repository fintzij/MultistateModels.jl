# 1. State changes occur (from the path)
# 2. Covariate values change (when multiple rows and covariates present)
# The function handles three cases:
# - Multiple rows with covariates: includes both state change and covariate change times
# - Single row with covariates OR no covariates: uses only path times
# - Returns appropriate columns based on whether covariates exist
using MultistateModels: SamplePath, make_subjdat

@testset "test_make_subjdat" begin
    
    @testset "Basic functionality with covariates" begin
        # Setup hazards with covariates for this test
        h12 = Hazard(@formula(0 ~ 1 + trt + age), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt + age), "wei", 2, 1)
        # Test case from the scratch file - data with covariate changes
        dat = DataFrame(
            id = [1,1,1,1,1],
            tstart = [0.0, 3.0, 7.0, 12.0, 18.0],
            tstop = [3.0, 7.0, 12.0, 18.0, 25.0],
            statefrom = [1, 1, 1, 1, 1],
            stateto = [1, 1, 1, 1, 1],
            obstype = [2, 2, 2, 2, 2],
            trt = [0, 1, 1, 0, 1],
            age = [50, 50, 55, 55, 60]
        )
        
        model = multistatemodel(h12, h21; data = dat)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 5.04, 15.01, 25.0], [1, 2, 1, 1])
        
        result = make_subjdat(path, subjectdata)
        
        # Test that we get a DataFrame
        @test isa(result, DataFrame)
        
        # Test that required columns are present
        expected_cols = ["tstart", "tstop", "increment", "sojourn", "statefrom", "stateto", "trt", "age"]
        @test all(col in names(result) for col in expected_cols)
        
        # Test that times include both state changes AND covariate changes
        # State changes at: 5.04, 15.01
        # Covariate changes at: 3.0 (trt: 0->1), 7.0 (age: 50->55), 12.0 (trt: 1->0), 18.0 (age: 55->60)
        # Expected unique times: [0.0, 3.0, 5.04, 7.0, 12.0, 15.01, 18.0, 25.0]
        expected_times = [0.0, 3.0, 5.04, 7.0, 12.0, 15.01, 18.0, 25.0]
        result_times = vcat(result.tstart, result.tstop[end])
        @test result_times ≈ expected_times
        
        # Test that covariates are correctly propagated
        @test result.trt == [0, 1, 1, 1, 0, 0, 1]  # trt values at each interval start
        @test result.age == [50, 50, 50, 55, 55, 55, 60] # age values at each interval start
        
        # Test that states are correctly from the path
        @test result.statefrom == [1, 1, 2, 2, 2, 1, 1]  # states at interval starts
        @test result.stateto == [1, 2, 2, 2, 1, 1, 1]    # states at interval ends
        
        # Test that increments sum to total time
        @test sum(result.increment) ≈ 25.0
        
        # Test sojourn times are computed correctly
        # Each sojourn should restart when state changes
        expected_sojourns = [0.0,  3.0,  0.0,  1.96,  6.96,  0.0,  2.99] # cumulative within each state visit
        @test result.sojourn ≈ expected_sojourns
    end
    
    @testset "Data without covariates" begin
        # Setup hazards without covariates for this test
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
        
        # Test with only the basic 6 columns - should use else branch and return only subjdat_lik
        dat_no_cov = DataFrame(
            id = [1,1,1],
            tstart = [0.0, 5.0, 15.0],
            tstop = [5.0, 15.0, 25.0],
            statefrom = [1, 1, 1],
            stateto = [1, 1, 1],
            obstype = [2, 2, 2]
        )
        
        model = multistatemodel(h12, h21; data = dat_no_cov)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 8.01, 20.02, 25.0], [1, 2, 1, 1])
        
        result = make_subjdat(path, subjectdata)
        
        # Test that result only includes path times (no covariate changes)
        expected_times = [0.0, 8.01, 20.02, 25.0]
        result_times = vcat(result.tstart, result.tstop[end])
        @test result_times ≈ expected_times
        
        # Test that no covariate columns are added
        @test !(:trt in names(result))
        
        # Test that only the basic columns are present (no covariates should be added)
        expected_cols = ["tstart", "tstop", "increment", "sojourn", "statefrom", "stateto"]
        @test Set(names(result)) == Set(expected_cols)
        @test length(names(result)) == length(expected_cols)
        
        # Test basic structure
        @test nrow(result) == 3
        @test result.statefrom == [1, 2, 1]
        @test result.stateto == [2, 1, 1]
    end
    
    @testset "Edge cases" begin
        # Setup hazards with covariates for this test
        h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 1)
        
        # Test single observation with covariates
        dat_single = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [10.0],
            statefrom = [1],
            stateto = [1],
            obstype = [2],
            trt = [1]
        )
        
        model = multistatemodel(h12, h21; data = dat_single)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 5.05, 10.0], [1, 2, 1])
        
        result = make_subjdat(path, subjectdata)
        
        # With single row and covariates, should use only path times (else branch)
        @test nrow(result) == 2
        @test result.trt == [1, 1]  # covariate should be constant
        expected_times = [0.0, 5.05, 10.0]
        result_times = vcat(result.tstart, result.tstop[end])
        @test result_times ≈ expected_times
        
        # Test when path times exactly match covariate change times
        dat_exact = DataFrame(
            id = [1,1],
            tstart = [0.0, 5.0],
            tstop = [5.0, 10.0],
            statefrom = [1, 1],
            stateto = [1, 1],
            obstype = [2, 2],
            trt = [0, 1]
        )
        
        model = multistatemodel(h12, h21; data = dat_exact)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 5.0, 10.0], [1, 2, 1])  # state change at same time as covariate change
        
        result = make_subjdat(path, subjectdata)

        # Should not duplicate the time at 5.0
        expected_times = [0.0, 5.0, 10.0]
        result_times = vcat(result.tstart, result.tstop[end])
        @test result_times ≈ expected_times
        @test nrow(result) == 2
        @test result.trt == [0, 1]  # should reflect the covariate change
    end
    
    @testset "Constant covariates (no changes)" begin
        # Setup hazards with covariates for this test
        h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 1)
        
        # Test case where covariates are present but don't change
        dat_constant = DataFrame(
            id = [1,1,1,1],
            tstart = [0.0, 5.0, 10.0, 15.0],
            tstop = [5.0, 10.0, 15.0, 20.0],
            statefrom = [1, 1, 1, 1],
            stateto = [1, 1, 1, 1],
            obstype = [2, 2, 2, 2],
            trt = [1, 1, 1, 1]  # constant covariate
        )
        
        model = multistatemodel(h12, h21; data = dat_constant)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 7.04, 18.03, 20.0], [1, 2, 1, 1])
        
        result = make_subjdat(path, subjectdata)
        
        # Should use only path times since no covariate changes detected
        expected_times = [0.0, 7.04, 18.03, 20.0]
        result_times = vcat(result.tstart, result.tstop[end])
        @test result_times ≈ expected_times
        
        # Should have 3 intervals
        @test nrow(result) == 3
        
        # Covariates should be constant
        @test all(result.trt .== 1)
        
        # Test structure
        @test result.statefrom == [1, 2, 1]
        @test result.stateto == [2, 1, 1]
    end
    
    @testset "Single row with covariates" begin
        # Setup hazards with covariates for this test
        h12 = Hazard(@formula(0 ~ 1 + trt + age), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt + age), "wei", 2, 1)
        
        # Test the specific case that was causing the bug: covariates present but only one row
        # This should now use the else branch (path.times only) rather than trying to find covariate changes
        dat_single_cov = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [15.0],
            statefrom = [1],
            stateto = [1],
            obstype = [2],
            trt = [0],
            age = [45]
        )
        
        model = multistatemodel(h12, h21; data = dat_single_cov)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 7.05, 12.02, 15.0], [1, 2, 1, 2])
        
        result = make_subjdat(path, subjectdata)
        
        # Should use only path times (no covariate change detection with single row)
        expected_times = [0.0, 7.05, 12.02, 15.0]
        result_times = vcat(result.tstart, result.tstop[end])
        @test result_times ≈ expected_times
        
        # Should have 3 intervals
        @test nrow(result) == 3
        
        # Covariates should be constant across all intervals
        @test all(result.trt .== 0)
        @test all(result.age .== 45)
        
        # Test structure
        @test result.statefrom == [1, 2, 1]
        @test result.stateto == [2, 1, 2]
    end
    
    @testset "Sojourn time calculations" begin
        # Setup hazards with covariates for this test
        h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 1)
        
        # Test specific sojourn time calculations
        dat = DataFrame(
            id = [1,1,1],
            tstart = [0.0, 10.0, 20.0],
            tstop = [10.0, 20.0, 30.0],
            statefrom = [1, 1, 1],
            stateto = [1, 1, 1],
            obstype = [2, 2, 2],
            trt = [0, 0, 1]  # covariate change at t=20
        )
        
        model = multistatemodel(h12, h21; data = dat)
        subjectdata = view(model.data, model.data.id .== 1, :)
        # Path: state 1 from 0-5, state 2 from 5-25, state 1 from 25-30
        path = SamplePath(1, [0.0, 5.02, 25.05, 30.0], [1, 2, 1, 1])
        
        result = make_subjdat(path, subjectdata)

        # Expected times: [0.0, 5.02, 20.0, 25.05, 30.0]
        # Intervals and their sojourns:
        # [0,5]: state 1, sojourn 0.0 (start of first visit to state 1)
        # [5,20]: state 2, sojourn 0.0 (start of visit to state 2)  
        # [20,25]: state 2, sojourn 15.0 (continuing in state 2)
        # [25,30]: state 1, sojourn 0.0 (start of second visit to state 1)
        
        expected_sojourns = [0.0, 0.0, 14.98, 0.0]
        @test result.sojourn ≈ expected_sojourns
    end
end