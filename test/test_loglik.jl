# Test that log-likelihoods are correctly calculated. Note that both hazards are technically exponential even though we specify weibull-PH for the 2->1 transition.

msm = msm_2state_trans

@testset "test_loglik_2state_trans" begin
    
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
    @test MultistateModels.loglik(path1, msm) ≈ ll1
    @test MultistateModels.loglik(path2, msm) ≈ ll2

end