# Test name-based covariate matching in runtime-generated hazard functions
# This addresses the architectural issue where different hazards may have different covariates

using MultistateModels
using DataFrames
using Test

@testset "Name-Based Covariate Matching" begin
    
    @testset "Helper Functions" begin
        # Test extract_covar_names
        parnames = [:h12_Intercept, :h12_age, :h12_trt]
        covar_names = MultistateModels.extract_covar_names(parnames)
        @test covar_names == [:age, :trt]
        
        # Test with no covariates
        parnames_nocov = [:h12_Intercept]
        covar_names_nocov = MultistateModels.extract_covar_names(parnames_nocov)
        @test isempty(covar_names_nocov)
        
        # Test extract_covariates
        df = DataFrame(id=1, tstart=0.0, tstop=10.0, statefrom=1, stateto=2, obstype=1, age=50, trt=1)
        row = df[1, :]
        covars = MultistateModels.extract_covariates(row, parnames)
        @test covars == (age=50, trt=1)
        
        # Test with no covariates
        covars_nocov = MultistateModels.extract_covariates(row, parnames_nocov)
        @test covars_nocov == NamedTuple()
    end
    
    @testset "Exponential Hazard - Name-Based" begin
        # Create hazard with covariates
        parnames = [:h12_Intercept, :h12_age, :h12_trt]
        hazard_fn, cumhaz_fn = MultistateModels.generate_exponential_hazard(true, parnames)
        
        # Test with NamedTuple covariates
        pars = [log(2.0), 0.5, -0.3]  # log_baseline, age_coef, trt_coef
        covars = (age=10.0, trt=1.0)
        
        # Expected: exp(log(2.0) + 0.5*10 + (-0.3)*1) = exp(log(2) + 5 - 0.3) = 2 * exp(4.7)
        expected_haz = exp(log(2.0) + 0.5*10.0 + (-0.3)*1.0)
        @test hazard_fn(1.0, pars, covars) ≈ expected_haz
        
        # Cumulative hazard: constant rate * duration
        expected_cumhaz = expected_haz * 5.0
        @test cumhaz_fn(0.0, 5.0, pars, covars) ≈ expected_cumhaz
    end
    
    @testset "Weibull Hazard - Name-Based" begin
        # Create hazard with covariates
        parnames = [:h12_shape, :h12_scale, :h12_age]
        hazard_fn, cumhaz_fn = MultistateModels.generate_weibull_hazard(true, parnames)
        
        pars = [log(2.0), log(1.5), 0.1]  # log_shape, log_scale, age_coef
        covars = (age=20.0,)
        t = 3.0
        
        # h(t) = shape * t^(shape-1) * scale * exp(age_coef * age)
        # = 2.0 * 3.0^(2-1) * 1.5 * exp(0.1*20)
        shape = 2.0
        scale = 1.5
        expected_haz = shape * t^(shape-1) * scale * exp(0.1*20.0)
        @test hazard_fn(t, pars, covars) ≈ expected_haz
    end
    
    @testset "Gompertz Hazard - Name-Based" begin
        # Create hazard with covariates
        parnames = [:h12_shape, :h12_scale, :h12_trt, :h12_age]
        hazard_fn, cumhaz_fn = MultistateModels.generate_gompertz_hazard(true, parnames)
        
        pars = [log(0.5), log(2.0), -0.2, 0.05]  # log_shape, log_scale, trt_coef, age_coef
        covars = (trt=1.0, age=30.0)
        t = 2.0
        
        # h(t) = scale * shape * exp(shape*t + trt_coef*trt + age_coef*age)
        shape = 0.5
        scale = 2.0
        expected_haz = scale * shape * exp(shape*t + (-0.2)*1.0 + 0.05*30.0)
        @test hazard_fn(t, pars, covars) ≈ expected_haz
    end
    
    @testset "Different Covariates Per Hazard" begin
        # THIS IS THE KEY TEST: Different hazards with different covariates
        
        # Hazard 1->2 depends on age only
        h12 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 2)
        # Hazard 2->1 depends on trt and sex
        h21 = Hazard(@formula(0 ~ 1 + trt + sex), "exp", 2, 1)
        
        # Create data with all covariates
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [10.0],
            statefrom = [1],
            stateto = [1],
            obstype = [2],
            age = [50],
            trt = [1],
            sex = [0]
        )
        
        # Build model
        model = multistatemodel(h12, h21; data = dat)
        
        # Check that hazards have correct parameter names
        @test model.hazards[1].parnames == [:h12_Intercept, :h12_age]
        @test model.hazards[2].parnames == [:h21_Intercept, :h21_trt, :h21_sex]
        
        # Test covariate extraction for each hazard
        row = dat[1, :]
        covars_h12 = MultistateModels.extract_covariates(row, model.hazards[1].parnames)
        covars_h21 = MultistateModels.extract_covariates(row, model.hazards[2].parnames)
        
        @test covars_h12 == (age=50,)
        @test covars_h21 == (trt=1, sex=0)
        
        # Test hazard evaluation with correct covariates
        pars_h12 = [log(2.0), 0.01]  # baseline, age_coef
        pars_h21 = [log(3.0), -0.5, 0.2]  # baseline, trt_coef, sex_coef
        
        haz_h12 = model.hazards[1](1.0, pars_h12, covars_h12)
        haz_h21 = model.hazards[2](1.0, pars_h21, covars_h21)
        
        @test haz_h12 ≈ exp(log(2.0) + 0.01*50)
        @test haz_h21 ≈ exp(log(3.0) + (-0.5)*1 + 0.2*0)
    end
    
    @testset "Backward Compatibility - No Covariates" begin
        # Hazards without covariates should still work
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [10.0],
            statefrom = [1],
            stateto = [1],
            obstype = [2]
        )
        
        model = multistatemodel(h12, h21; data = dat)
        
        # Should have only intercept parameters
        @test model.hazards[1].parnames == [:h12_Intercept]
        @test model.hazards[2].parnames == [:h21_Intercept]
        
        # Covariate extraction should return empty NamedTuple
        row = dat[1, :]
        covars_h12 = MultistateModels.extract_covariates(row, model.hazards[1].parnames)
        covars_h21 = MultistateModels.extract_covariates(row, model.hazards[2].parnames)
        
        @test covars_h12 == NamedTuple()
        @test covars_h21 == NamedTuple()
    end
    
    @testset "Index-Based Backward Compatibility" begin
        # Test that old index-based code path still works (when parnames not provided)
        hazard_fn_old, cumhaz_fn_old = MultistateModels.generate_exponential_hazard(true)
        
        pars = [log(2.0), 0.5, -0.3]
        covars_vector = [10.0, 1.0]  # Old style: Vector instead of NamedTuple
        
        # Should work with vector covariates
        expected_haz = exp(log(2.0) + 0.5*10.0 + (-0.3)*1.0)
        @test hazard_fn_old(1.0, pars, covars_vector) ≈ expected_haz
    end
end

println("\n✓ All name-based covariate matching tests passed!")
