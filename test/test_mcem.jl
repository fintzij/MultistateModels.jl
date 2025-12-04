# =============================================================================
# Test suite for MCEM algorithm and related functions
# =============================================================================
#
# Tests for:
# - MCEM helper function correctness (mcem_mll, mcem_lml, mcem_ase)
# - SQUAREM acceleration helpers
# - ForwardDiff gradient compatibility

using Test
using MultistateModels
using DataFrames
using LinearAlgebra
using Random
using ForwardDiff

@testset "MCEM Tests" begin
    
    @testset "ForwardDiff gradient compatibility" begin
        # Critical: If gradients are wrong, optimization silently fails
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        nsubj = 3
        dat = DataFrame(
            id = repeat(1:nsubj, inner=2),
            tstart = repeat([0.0, 1.0], outer=nsubj),
            tstop = repeat([1.0, 2.0], outer=nsubj),
            statefrom = repeat([1, 2], outer=nsubj),
            stateto = repeat([2, 3], outer=nsubj),
            obstype = fill(1, 2*nsubj)
        )
        
        model = multistatemodel(h12, h23; data=dat)
        params = MultistateModels.get_parameters_flat(model)
        
        # Test ExactData gradient
        samplepaths = [MultistateModels.SamplePath(i, [0.0, 1.0, 2.0], [1, 2, 3]) for i in 1:nsubj]
        exactdata = MultistateModels.ExactData(model, samplepaths)
        
        grad = ForwardDiff.gradient(p -> MultistateModels.loglik_exact(p, exactdata; neg=true), params)
        @test length(grad) == length(params)
        @test all(isfinite.(grad))
        
        # Test SMPanelData gradient
        samplepaths_nested = [[MultistateModels.SamplePath(i, [0.0, 1.0, 2.0], [1, 2, 3])] for i in 1:nsubj]
        weights = [[1.0] for _ in 1:nsubj]
        smdata = MultistateModels.SMPanelData(model, samplepaths_nested, weights)
        
        grad_sm = ForwardDiff.gradient(p -> MultistateModels.loglik_semi_markov(p, smdata; neg=true), params)
        @test length(grad_sm) == length(params)
        @test all(isfinite.(grad_sm))
    end
    
    @testset "MCEM helper functions" begin
        # Test mcem_mll, mcem_ase, mcem_lml, mcem_lml_subj with known values
        logliks = [[-1.0, -2.0], [-1.5, -2.5]]
        ImportanceWeights = [[0.6, 0.4], [0.5, 0.5]]
        SubjectWeights = [1.0, 1.0]
        
        # mcem_mll: weighted average of log-likelihoods
        mll = MultistateModels.mcem_mll(logliks, ImportanceWeights, SubjectWeights)
        expected = (0.6*(-1.0) + 0.4*(-2.0)) + (0.5*(-1.5) + 0.5*(-2.5))
        @test mll ≈ expected
        
        # mcem_mll with non-unit subject weights
        SubjectWeights2 = [2.0, 0.5]
        mll2 = MultistateModels.mcem_mll(logliks, ImportanceWeights, SubjectWeights2)
        expected2 = 2.0*(0.6*(-1.0) + 0.4*(-2.0)) + 0.5*(0.5*(-1.5) + 0.5*(-2.5))
        @test mll2 ≈ expected2
        
        # mcem_lml: log marginal likelihood = log(sum(exp(ll) * w))
        lml = MultistateModels.mcem_lml(logliks, ImportanceWeights, SubjectWeights)
        expected_lml = log(0.6*exp(-1.0) + 0.4*exp(-2.0)) + log(0.5*exp(-1.5) + 0.5*exp(-2.5))
        @test lml ≈ expected_lml
        
        # mcem_lml_subj: per-subject log marginal likelihoods
        lml_subj = MultistateModels.mcem_lml_subj(logliks, ImportanceWeights)
        @test length(lml_subj) == 2
        @test lml_subj[1] ≈ log(0.6*exp(-1.0) + 0.4*exp(-2.0))
        @test lml_subj[2] ≈ log(0.5*exp(-1.5) + 0.5*exp(-2.5))
        
        # mcem_ase with identical logliks (no variance) should be zero
        ase_zero = MultistateModels.mcem_ase(logliks, logliks, ImportanceWeights, SubjectWeights)
        @test ase_zero ≈ 0.0
        
        # var_ris with all zeros should return 0
        w = [0.3, 0.3, 0.4]
        v_zero = MultistateModels.var_ris([0.0, 0.0, 0.0], w)
        @test v_zero ≈ 0.0
    end
    
    @testset "SQUAREM acceleration helpers" begin
        θ0 = [0.0, 0.0]
        θ1 = [1.0, 1.0]
        θ2 = [1.5, 1.5]
        
        α, r, v = MultistateModels.squarem_step_length(θ0, θ1, θ2)
        @test r == [1.0, 1.0]
        @test v == [-0.5, -0.5]
        @test α < 0.0
        @test α >= -1.0
        
        # squarem_accelerate: θ_acc = θ0 - 2αr + α²v
        θ_acc = MultistateModels.squarem_accelerate(θ0, r, v, α)
        expected_acc = θ0 .- 2 * α .* r .+ α^2 .* v
        @test θ_acc ≈ expected_acc
        
        # squarem_should_accept: acc better than start
        @test MultistateModels.squarem_should_accept(-10.0, -12.0, -15.0) == true
        @test MultistateModels.squarem_should_accept(-20.0, -12.0, -15.0) == false
        
        # SquaremState constructor
        state = MultistateModels.SquaremState(5)
        @test length(state.θ0) == 5
        @test state.step == 0
        @test state.n_accelerations == 0
    end
    
end
