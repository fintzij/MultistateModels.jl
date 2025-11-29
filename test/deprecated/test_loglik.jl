# Test that log-likelihoods are correctly calculated. Note that both hazards are technically exponential even though we specify weibull-PH for the 2->1 transition.
# 1. State changes occur (from the path)
# 2. Covariate values change (when multiple rows and covariates present)
using MultistateModels: SamplePath, make_subjdat
using .TestFixtures:
    toy_two_state_transition_model,
    make_subjdat_covariate_panel,
    make_subjdat_baseline_panel,
    make_subjdat_single_observation_panel,
    make_subjdat_exact_match_panel,
    make_subjdat_constant_covariates_panel,
    make_subjdat_single_row_full_panel,
    make_subjdat_sojourn_panel

@testset "test_loglik_2state_trans" begin
    fixture = toy_two_state_transition_model()
    msm = fixture.model
    path1 = fixture.paths.path1
    path2 = fixture.paths.path2
    
    # initialize log-likelihoods
    ll1 = 
        logpdf(Exponential(10), path1.times[2]) + 
        logccdf(Exponential(10), 10.0 - path1.times[2]) +
        logpdf(Exponential(5), path1.times[3] - 10.0) +  
        logccdf(Exponential(5), 20.0 - path1.times[3]) + 
        logccdf(Exponential(10), 10.0)
        
    ll2 = 
        logpdf(Exponential(5), path2.times[2]) + 
        logccdf(Exponential(5), 10.0 - path2.times[2]) + 
        logccdf(Exponential(10), 10.0) + 
        logpdf(Exponential(5), path2.times[3] - 20.0) + 
        logpdf(Exponential(5), path2.times[4] - path2.times[3]) + 
        logpdf(Exponential(5), path2.times[5] - path2.times[4])
        
    # test for equality vs. loglik function
    @test MultistateModels.loglik(msm.parameters, path1, msm.hazards, msm) ≈ ll1
    @test MultistateModels.loglik(msm.parameters, path2, msm.hazards, msm) ≈ ll2

end

# compare new path loglik function with old one 
@testset "test_make_subjdat" begin
    
    @testset "Basic functionality with covariates" begin
        # Setup hazards with covariates for this test
        h12 = Hazard(@formula(0 ~ 1 + trt + age), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt + age), "wei", 2, 1)
        # Test case from the scratch file - data with covariate changes
        dat = make_subjdat_covariate_panel()
        
        model = multistatemodel(h12, h21; data = dat)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = MultistateModels.SamplePath(1, [0.0, 5.04, 15.01, 25.0], [1, 2, 1, 1])
        
        subjdat_path = MultistateModels.make_subjdat(path, subjectdata)

        ll = MultistateModels.loglik_path(model.parameters, subjdat_path, model.hazards, model.totalhazards, model.tmat)
        ll_OLD = loglik_path_OLD(model.parameters, path, model.hazards, model) 
        @test ll ≈ ll_OLD

        # join new Teams invite when your meeting is over.

        # I think there is an error in how we're indexing into hazards and parameters

        #ooooh we forgot to pass tmat

    end
    
    @testset "Data without covariates" begin
        # Setup hazards without covariates for this test
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
        
        # Test with only the basic 6 columns - should use else branch and return only subjdat_lik
        dat_no_cov = make_subjdat_baseline_panel()
        
        model = multistatemodel(h12, h21; data = dat_no_cov)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 8.01, 20.02, 25.0], [1, 2, 1, 1])
        
        result = make_subjdat(path, subjectdata)
        
    end
    
    @testset "Edge cases" begin
        # Setup hazards with covariates for this test
        h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 1)
        
        # Test single observation with covariates
        dat_single = make_subjdat_single_observation_panel()
        
        model = multistatemodel(h12, h21; data = dat_single)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 5.05, 10.0], [1, 2, 1])
        
        result = make_subjdat(path, subjectdata)
        
        
        # Test when path times exactly match covariate change times
        dat_exact = make_subjdat_exact_match_panel()
        
        model = multistatemodel(h12, h21; data = dat_exact)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 5.0, 10.0], [1, 2, 1])  # state change at same time as covariate change
        
        result = make_subjdat(path, subjectdata)

       
    end
    
    @testset "Constant covariates (no changes)" begin
        # Setup hazards with covariates for this test
        h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 1)
        
        # Test case where covariates are present but don't change
        dat_constant = make_subjdat_constant_covariates_panel()
        
        model = multistatemodel(h12, h21; data = dat_constant)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 7.04, 18.03, 20.0], [1, 2, 1, 1])
        
        result = make_subjdat(path, subjectdata)
        
    end
    
    @testset "Single row with covariates" begin
        # Setup hazards with covariates for this test
        h12 = Hazard(@formula(0 ~ 1 + trt + age), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt + age), "wei", 2, 1)
        
        # Test the specific case that was causing the bug: covariates present but only one row
        # This should now use the else branch (path.times only) rather than trying to find covariate changes
        dat_single_cov = make_subjdat_single_row_full_panel()
        
        model = multistatemodel(h12, h21; data = dat_single_cov)
        subjectdata = view(model.data, model.data.id .== 1, :)
        path = SamplePath(1, [0.0, 7.05, 12.02, 15.0], [1, 2, 1, 2])
        
        result = make_subjdat(path, subjectdata)
        
    end
    
    @testset "Sojourn time calculations" begin
        # Setup hazards with covariates for this test
        h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 1)
        
        # Test specific sojourn time calculations
        dat = make_subjdat_sojourn_panel()
        
        model = multistatemodel(h12, h21; data = dat)
        subjectdata = view(model.data, model.data.id .== 1, :)
        # Path: state 1 from 0-5, state 2 from 5-25, state 1 from 25-30
        path = SamplePath(1, [0.0, 5.02, 25.05, 30.0], [1, 2, 1, 1])
        
        result = make_subjdat(path, subjectdata)

    end
end