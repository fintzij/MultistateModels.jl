# =============================================================================
# Surrogate Fitting Tests
# =============================================================================
#
# Tests for:
# - Markov surrogate Q-matrix validity (negative diagonals, row sum = 0)
# - Phase-type surrogate Q-matrix validity
# - MLE produces non-negative rates
# - MLE log-likelihood >= heuristic log-likelihood
# - `fit_surrogate` keyword argument behavior
# - `is_surrogate_fitted` function
using Test
using MultistateModels
using Random
using DataFrames
using Statistics

@testset "Surrogate Fitting" begin

    # Setup: Panel data for testing
    function create_test_data(; n_subj = 30, seed = 12345)
        Random.seed!(seed)
        dat = DataFrame(
            id = repeat(1:n_subj, inner = 3),
            tstart = repeat([0.0, 1.0, 2.0], n_subj),
            tstop = repeat([1.0, 2.0, 3.0], n_subj),
            statefrom = repeat([1, 1, 1], n_subj),
            stateto = vcat([[rand() < 0.3 ? 2 : 1, rand() < 0.5 ? 2 : 1, 2] for _ in 1:n_subj]...),
            obstype = repeat([2, 2, 2], n_subj)
        )
        return dat
    end
    
    @testset "fit_surrogate default behavior" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        # Default: fit_surrogate=true - surrogate should be fitted during model creation
        model_default = multistatemodel(h12; data = dat, surrogate = :markov)
        @test is_surrogate_fitted(model_default)
        @test model_default.markovsurrogate.fitted == true
        
        # Explicit fit_surrogate=false - surrogate exists but not fitted
        model_deferred = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        @test !is_surrogate_fitted(model_deferred)
        @test model_deferred.markovsurrogate.fitted == false
        
        # No surrogate - is_surrogate_fitted returns false
        model_none = multistatemodel(h12; data = dat)
        @test !is_surrogate_fitted(model_none)
    end
    
    @testset "set_surrogate! marks as fitted" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        # Start with unfitted surrogate
        model = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        @test !is_surrogate_fitted(model)
        
        # set_surrogate! should fit and mark as fitted
        set_surrogate!(model; verbose = false)
        @test is_surrogate_fitted(model)
    end
    
    @testset "Markov MLE produces positive rates" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        surrogate = MultistateModels._fit_markov_surrogate(model; method = :mle, verbose = false)
        
        # Rates must be positive (or model is broken)
        rates = values(surrogate.parameters.natural)
        @test all(r[1] > 0 for r in rates)
        
        # Surrogate should be marked as fitted
        @test surrogate.fitted == true
    end
    
    @testset "Phase-Type Q-matrix validity" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        surrogate = MultistateModels._fit_phasetype_surrogate(model; 
            method = :mle, n_phases = 2, verbose = false)
        
        Q = surrogate.expanded_Q
        
        # Diagonals must be non-positive
        for i in 1:size(Q, 1)
            @test Q[i, i] <= 0.0
        end
        
        # Rows must sum to 0 (generator property)
        for i in 1:size(Q, 1)
            @test isapprox(sum(Q[i, :]), 0.0, atol = 1e-10)
        end
    end
    
    @testset "MLE >= heuristic log-likelihood" begin
        dat = create_test_data(n_subj = 100)
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        surrogate_mle = MultistateModels._fit_markov_surrogate(model; method = :mle, verbose = false)
        surrogate_heur = MultistateModels._fit_markov_surrogate(model; method = :heuristic, verbose = false)
        
        ll_mle = MultistateModels.compute_markov_marginal_loglik(model, surrogate_mle)
        ll_heur = MultistateModels.compute_markov_marginal_loglik(model, surrogate_heur)
        
        # MLE is optimal, so its log-likelihood should be >= heuristic
        @test ll_mle >= ll_heur - 1e-6
        
        # Verify the actual log-likelihoods are finite and reasonable in magnitude
        # For n=100 subjects with 3 observations each, expect total ll in [-500, -10] range
        @test isfinite(ll_mle)
        @test isfinite(ll_heur)
        @test ll_mle > -1000.0  # Not implausibly negative
        @test ll_mle < -5.0     # Not implausibly close to zero for 100 subjects
    end
    
    @testset "set_surrogate! auto-fits unfitted surrogate" begin
        dat = create_test_data(n_subj = 20)
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # Create model with unfitted surrogate
        model = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        @test !is_surrogate_fitted(model)
        
        # set_surrogate! should fit the surrogate (this is what fit() calls internally)
        MultistateModels.set_surrogate!(model; verbose = false)
        
        # After fitting, the model's surrogate should be fitted
        @test is_surrogate_fitted(model)
        @test model.markovsurrogate.fitted == true
        
        # Rates should be positive
        rates = values(model.markovsurrogate.parameters.natural)
        @test all(r[1] > 0 for r in rates)
    end
    
    @testset "Input validation" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        @test_throws ErrorException MultistateModels._validate_surrogate_inputs(:invalid, :mle)
        @test_throws ErrorException MultistateModels._validate_surrogate_inputs(:markov, :invalid)
    end

end
