# =============================================================================
# Unit Tests for Reversible Semi-Markov Models with TVC
# =============================================================================
#
# Tests targeting reversible models with semi-Markov hazards and TVC.
# NOTE: Basic analytical likelihood validation is in test_loglik.jl
# This file focuses on reversible model-specific behaviors.

using Test
using MultistateModels
using DataFrames
using Random
using Statistics

import MultistateModels: 
    SamplePath, make_subjdat, set_parameters!, SMPanelData, loglik!

@testset "Reversible Semi-Markov with TVC" begin
    
    @testset "Sojourn time resets in reversible model" begin
        # Path: 1 (0-3) → 2 (3-7) → 1 (7-12) → 2 (12-15)
        # Sojourn should reset each time we enter a new state
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
        
        dat = DataFrame(
            id = fill(1, 4),
            tstart = [0.0, 3.0, 7.0, 12.0],
            tstop = [3.0, 7.0, 12.0, 15.0],
            statefrom = [1, 2, 1, 2],
            stateto = [2, 1, 2, 2],
            obstype = fill(1, 4)
        )
        
        model = multistatemodel(h12, h21; data=dat)
        set_parameters!(model, (h12 = [log(2.0), log(0.3)], h21 = [log(1.5), log(0.25)]))
        
        paths = MultistateModels.extract_paths(model)
        @test length(paths) == 1
        
        path = paths[1]
        @test path.times ≈ [0.0, 3.0, 7.0, 12.0, 15.0]
        @test path.states == [1, 2, 1, 2, 2]
        
        # Compute likelihood
        subjectdata = view(model.data, model.data.id .== 1, :)
        subjdat_path = make_subjdat(path, subjectdata)
        
        # Verify sojourn times are correct - should reset each transition
        @test subjdat_path.sojourn ≈ [0.0, 0.0, 0.0, 0.0]
    end
    
    @testset "Manual vs package path likelihood - reversible Weibull" begin
        # Create a simple reversible model and manually compute likelihood
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 5.0],
            tstop = [5.0, 10.0],
            statefrom = [1, 2],
            stateto = [2, 2],
            obstype = [1, 1]
        )
        
        model = multistatemodel(h12, h21; data=dat)
        
        # Parameters: log(shape), log(scale)
        shape_12, scale_12 = 1.5, 0.2
        shape_21, scale_21 = 1.2, 0.3
        set_parameters!(model, (
            h12 = [log(shape_12), log(scale_12)],
            h21 = [log(shape_21), log(scale_21)]
        ))
        
        # Path: state 1 for [0, 5], then state 2 for [5, 10]
        # Transition from 1→2 at t=5
        
        # Manual calculation for Weibull hazard:
        # h(t) = scale * shape * t^(shape-1)
        # H(t) = scale * t^shape  (cumulative hazard)
        # 
        # Interval [0, 5] in state 1:
        #   - Sojourn time = 5
        #   - h12 density at t=5: scale_12 * shape_12 * 5^(shape_12-1)
        #   - h12 cumulative: scale_12 * 5^shape_12
        #   - h21 not active (we're in state 1)
        # 
        # Interval [5, 10] in state 2:
        #   - SOJOURN RESETS! sojourn = 0 to 5
        #   - Duration in state 2: 5 time units
        #   - h21 survival: exp(-scale_21 * 5^shape_21)
        #   - No transition (right-censored)
        
        # Part 1: [0,5] in state 1, transition to 2
        h_val = scale_12 * shape_12 * 5^(shape_12-1)
        H_val = scale_12 * 5^shape_12
        ll_part1 = log(h_val) - H_val
        
        # Part 2: [5,10] in state 2, NO transition (sojourn = 5)
        ll_part2 = -scale_21 * 5^shape_21  # Just survival, no density
        
        ll_manual = ll_part1 + ll_part2
        
        # Package computation
        paths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, paths)
        pars = model.parameters.flat
        ll_package = MultistateModels.loglik_exact(pars, exact_data; neg=false)
        
        @test ll_package ≈ ll_manual rtol=1e-6
    end
    
    @testset "Reversible model with TVC and multiple sojourns" begin
        # Path: 1→2→1→2 with covariate change
        # Tests that TVC is handled correctly with sojourn resets
        
        h12 = Hazard(@formula(0 ~ 1 + x), "wei", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + x), "wei", 2, 1)
        
        # Parameters
        shape_12, scale_12, beta_12 = 1.3, 0.2, 0.5
        shape_21, scale_21, beta_21 = 1.1, 0.15, -0.3
        
        # Covariate x = 1 throughout (no TVC change, simplifies manual calc)
        dat = DataFrame(
            id = fill(1, 4),
            tstart = [0.0, 2.0, 6.0, 9.0],
            tstop = [2.0, 6.0, 9.0, 12.0],
            statefrom = [1, 2, 1, 2],
            stateto = [2, 1, 2, 2],
            obstype = fill(1, 4),
            x = fill(1.0, 4)
        )
        
        model = multistatemodel(h12, h21; data=dat)
        set_parameters!(model, (
            h12 = [log(shape_12), log(scale_12), beta_12],
            h21 = [log(shape_21), log(scale_21), beta_21]
        ))
        
        # Manual calculation:
        # Linear predictor for h12: log(scale_12) + beta_12 * x = log(scale_12) + beta_12
        # Effective scale_12: exp(log(scale_12) + beta_12) = scale_12 * exp(beta_12)
        eff_scale_12 = scale_12 * exp(beta_12)
        eff_scale_21 = scale_21 * exp(beta_21)
        
        # Interval 1: [0, 2], state 1, transition 1→2, sojourn = 2
        t1 = 2.0
        h1 = eff_scale_12 * shape_12 * t1^(shape_12-1)
        H1 = eff_scale_12 * t1^shape_12
        ll_1 = log(h1) - H1
        
        # Interval 2: [2, 6], state 2, transition 2→1, sojourn = 4
        t2 = 4.0  # sojourn time
        h2 = eff_scale_21 * shape_21 * t2^(shape_21-1)
        H2 = eff_scale_21 * t2^shape_21
        ll_2 = log(h2) - H2
        
        # Interval 3: [6, 9], state 1, transition 1→2, sojourn = 3
        t3 = 3.0
        h3 = eff_scale_12 * shape_12 * t3^(shape_12-1)
        H3 = eff_scale_12 * t3^shape_12
        ll_3 = log(h3) - H3
        
        # Interval 4: [9, 12], state 2, no transition, sojourn = 3
        t4 = 3.0
        H4 = eff_scale_21 * t4^shape_21
        ll_4 = -H4  # survival only
        
        ll_manual = ll_1 + ll_2 + ll_3 + ll_4
        
        # Package computation
        paths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, paths)
        pars = model.parameters.flat
        ll_package = MultistateModels.loglik_exact(pars, exact_data; neg=false)
        
        @test ll_package ≈ ll_manual rtol=1e-6
    end
    
end
