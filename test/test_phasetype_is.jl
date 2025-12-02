# Unit tests for phase-type importance sampling
using Test
using MultistateModels
using Random
using DataFrames
using Statistics
using LinearAlgebra

@testset "Phase-Type Importance Sampling" begin

    #==========================================================================
    Test 1: IS weights average to 1 when target = proposal (Markov model)
    
    When the target model is Markov (exponential hazards) AND the target
    parameters equal the Markov surrogate MLE, importance weights should
    average to 1.
    ==========================================================================#
    @testset "IS weights = 1 when target = proposal (Markov)" begin
        Random.seed!(12345)
        
        # Simple 2-state model with exponential hazards (Markov)
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # Panel data - larger dataset for stability
        n_subj = 50
        dat = DataFrame(
            id = repeat(1:n_subj, inner=3),
            tstart = repeat([0.0, 1.0, 2.0], n_subj),
            tstop = repeat([1.0, 2.0, 3.0], n_subj),
            statefrom = repeat([1, 1, 1], n_subj),
            stateto = vcat([[rand() < 0.3 ? 2 : 1, rand() < 0.5 ? 2 : 1, 2] for _ in 1:n_subj]...),
            obstype = repeat([2, 2, 2], n_subj)
        )
        
        model = multistatemodel(h12; data=dat)
        
        # Fit Markov surrogate to get the MLE
        surrogate_fitted = MultistateModels.fit_surrogate(model; verbose=false)
        
        # Set target parameters to EXACTLY match the Markov surrogate MLE
        # This makes target = proposal
        MultistateModels.set_parameters!(model, [collect(surrogate_fitted.parameters[1])])
        
        # Build phase-type with 1 phase per state (equivalent to Markov)
        tmat = model.tmat
        phasetype_config = MultistateModels.PhaseTypeConfig(n_phases=[1, 1])
        surrogate = MultistateModels.build_phasetype_surrogate(tmat, phasetype_config)
        
        # Update phase-type Q matrix to match the fitted Markov surrogate
        # The phase-type rate should equal the Markov rate
        markov_rate = exp(surrogate_fitted.parameters[1][1])
        surrogate.expanded_Q[1, 1] = -markov_rate
        surrogate.expanded_Q[1, 2] = markov_rate
        
        # Build infrastructure
        emat_ph = MultistateModels.build_phasetype_emat_expanded(model, surrogate)
        books = MultistateModels.build_tpm_mapping(model.data)
        absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
        tpm_book_ph, hazmat_book_ph = MultistateModels.build_phasetype_tpm_book(surrogate, books, model.data)
        fbmats_ph = MultistateModels.build_fbmats_phasetype(model, surrogate)
        
        # Compute marginal likelihood under phase-type
        ll_marginal = MultistateModels.compute_phasetype_marginal_loglik(model, surrogate, emat_ph)
        
        # Sample many paths and compute importance weights for subject 1
        n_paths = 500
        log_weights = Float64[]
        
        for _ in 1:n_paths
            path_result = MultistateModels.draw_samplepath_phasetype(
                1, model, tpm_book_ph, hazmat_book_ph, books[2], 
                fbmats_ph, emat_ph, surrogate, absorbingstates)
            
            params = MultistateModels.get_log_scale_params(model.parameters)
            ll_target = MultistateModels.loglik(params, path_result.collapsed, model.hazards, model)
            ll_surrog = MultistateModels.loglik_phasetype_expanded(path_result.expanded, surrogate)
            push!(log_weights, ll_target - ll_surrog)
        end
        
        weights = exp.(log_weights)
        mean_weight = mean(weights)
        
        # IS estimate for subject 1
        ll_is = ll_marginal + log(mean_weight)
        
        # When target = proposal, mean weight should be exactly 1
        @test mean_weight ≈ 1.0 atol=0.05  # Should be very close to 1
        
        # IS estimate should equal marginal likelihood
        @test ll_is ≈ ll_marginal atol=0.1
        
        # Log-likelihood should be negative
        @test ll_is < 0
    end

    #==========================================================================
    Test 2: collapse_phasetype_path correctness
    ==========================================================================#
    @testset "collapse_phasetype_path" begin
        # Create a simple surrogate for testing
        tmat = [0 1 0; 0 0 1; 0 0 0]
        config = MultistateModels.PhaseTypeConfig(n_phases=[2, 2, 1])
        surrogate = MultistateModels.build_phasetype_surrogate(tmat, config)
        absorbingstates = [3]
        
        # Test 1: Path that stays in one observed state (phases 1→2→1)
        expanded = MultistateModels.SamplePath(1, [0.0, 0.5, 1.0], [1, 2, 1])
        collapsed = MultistateModels.collapse_phasetype_path(expanded, surrogate, absorbingstates)
        @test collapsed.states == [1]  # All phases map to state 1
        @test collapsed.times == [0.0]
        
        # Test 2: Path with transition between observed states
        expanded = MultistateModels.SamplePath(1, [0.0, 0.5, 1.0, 1.5], [1, 2, 3, 5])
        collapsed = MultistateModels.collapse_phasetype_path(expanded, surrogate, absorbingstates)
        @test collapsed.states == [1, 2, 3]
        @test collapsed.times == [0.0, 1.0, 1.5]
        
        # Test 3: Path that ends in absorbing state - should truncate
        expanded = MultistateModels.SamplePath(1, [0.0, 1.0, 2.0], [1, 3, 5])
        collapsed = MultistateModels.collapse_phasetype_path(expanded, surrogate, absorbingstates)
        @test collapsed.states[end] == 3  # Absorbing state
        @test length(collapsed.states) == 3
    end

    #==========================================================================
    Test 3: loglik_phasetype_expanded correctness
    ==========================================================================#
    @testset "loglik_phasetype_expanded" begin
        # Create surrogate
        tmat = [0 1 0; 0 0 1; 0 0 0]
        config = MultistateModels.PhaseTypeConfig(n_phases=[1, 1, 1])
        surrogate = MultistateModels.build_phasetype_surrogate(tmat, config)
        
        # Set Q matrix manually for testing
        # Q = [-1  1  0]
        #     [ 0 -2  2]
        #     [ 0  0  0]
        Q = [-1.0 1.0 0.0; 0.0 -2.0 2.0; 0.0 0.0 0.0]
        surrogate.expanded_Q .= Q
        
        # Simple path: state 1 for 0.5 time, then transition to state 2
        path = MultistateModels.SamplePath(1, [0.0, 0.5, 1.0], [1, 2, 3])
        
        # Manual calculation:
        # Interval 1: state 1 for 0.5 time, then transition to 2
        #   survival: exp(-1 * 0.5) → log = -0.5
        #   transition: log(1) = 0
        # Interval 2: state 2 for 0.5 time, then transition to 3
        #   survival: exp(-2 * 0.5) → log = -1.0
        #   transition: log(2)
        # Total: -0.5 + 0 - 1.0 + log(2) = -1.5 + 0.693 = -0.807
        
        ll = MultistateModels.loglik_phasetype_expanded(path, surrogate)
        expected = -0.5 + 0.0 - 1.0 + log(2.0)
        @test ll ≈ expected atol=1e-10
        
        # Path with no transitions
        path_none = MultistateModels.SamplePath(1, [0.0], [1])
        ll_none = MultistateModels.loglik_phasetype_expanded(path_none, surrogate)
        @test ll_none == 0.0  # No transitions = log(1) = 0
    end

    #==========================================================================
    Test 4: draw_samplepath_phasetype produces valid paths
    ==========================================================================#
    @testset "draw_samplepath_phasetype validity" begin
        Random.seed!(54321)
        
        # Illness-death model
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        # Mix of exact and panel observations
        dat = DataFrame(
            id = [1, 1, 1, 1],
            tstart = [0.0, 0.5, 1.0, 1.5],
            tstop = [0.5, 1.0, 1.5, 2.0],
            statefrom = [1, 1, 2, 2],
            stateto = [1, 2, 2, 3],
            obstype = [2, 1, 2, 1]  # Panel, Exact, Panel, Exact
        )
        
        model = multistatemodel(h12, h13, h23; data=dat)
        
        # Build phase-type
        tmat = model.tmat
        config = MultistateModels.PhaseTypeConfig(n_phases=[2, 2, 1])
        surrogate = MultistateModels.build_phasetype_surrogate(tmat, config)
        
        emat_ph = MultistateModels.build_phasetype_emat_expanded(model, surrogate)
        books = MultistateModels.build_tpm_mapping(model.data)
        absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
        tpm_book_ph, hazmat_book_ph = MultistateModels.build_phasetype_tpm_book(surrogate, books, model.data)
        fbmats_ph = MultistateModels.build_fbmats_phasetype(model, surrogate)
        
        # Sample paths and verify validity
        for _ in 1:20
            result = MultistateModels.draw_samplepath_phasetype(
                1, model, tpm_book_ph, hazmat_book_ph, books[2],
                fbmats_ph, emat_ph, surrogate, absorbingstates)
            
            # Collapsed path should start at initial state
            @test result.collapsed.states[1] == 1
            
            # Collapsed path times should be monotonically increasing
            @test issorted(result.collapsed.times)
            
            # Expanded path times should be monotonically increasing
            @test issorted(result.expanded.times)
            
            # Expanded states should be valid phase indices
            @test all(1 .<= result.expanded.states .<= surrogate.n_expanded_states)
            
            # Collapsed states should match observed transitions
            # (path must pass through observed states at observed times)
            collapsed = result.collapsed
            
            # For exact observations, the transition should be recorded
            # Check that state 2 appears after state 1
            idx_state2 = findfirst(==(2), collapsed.states)
            if !isnothing(idx_state2)
                @test all(collapsed.states[1:idx_state2-1] .== 1)
            end
        end
    end

    #==========================================================================
    Test 5: IS produces negative log-likelihood for semi-Markov model
    
    When using phase-type IS for a Weibull (semi-Markov) model, the IS
    estimate should be a reasonable negative log-likelihood.
    ==========================================================================#
    @testset "IS estimate negative for semi-Markov" begin
        Random.seed!(99999)
        
        # Weibull model (semi-Markov)
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        # Panel data
        n_subj = 30
        dat = DataFrame(
            id = repeat(1:n_subj, inner=2),
            tstart = repeat([0.0, 1.0], n_subj),
            tstop = repeat([1.0, 2.0], n_subj),
            statefrom = repeat([1, 1], n_subj),
            stateto = vcat([[rand() < 0.4 ? 2 : 1, 2] for _ in 1:n_subj]...),
            obstype = repeat([2, 2], n_subj)
        )
        
        model = multistatemodel(h12; data=dat)
        
        # Build phase-type surrogate with 2 phases
        tmat = model.tmat
        config = MultistateModels.PhaseTypeConfig(n_phases=[2, 1])
        surrogate = MultistateModels.build_phasetype_surrogate(tmat, config)
        
        emat_ph = MultistateModels.build_phasetype_emat_expanded(model, surrogate)
        books = MultistateModels.build_tpm_mapping(model.data)
        absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
        tpm_book_ph, hazmat_book_ph = MultistateModels.build_phasetype_tpm_book(surrogate, books, model.data)
        fbmats_ph = MultistateModels.build_fbmats_phasetype(model, surrogate)
        
        ll_marginal = MultistateModels.compute_phasetype_marginal_loglik(model, surrogate, emat_ph)
        
        # Sample paths for first subject
        n_paths = 300
        log_weights = Float64[]
        
        for _ in 1:n_paths
            result = MultistateModels.draw_samplepath_phasetype(
                1, model, tpm_book_ph, hazmat_book_ph, books[2],
                fbmats_ph, emat_ph, surrogate, absorbingstates)
            
            params = MultistateModels.get_log_scale_params(model.parameters)
            ll_target = MultistateModels.loglik(params, result.collapsed, model.hazards, model)
            ll_surrog = MultistateModels.loglik_phasetype_expanded(result.expanded, surrogate)
            push!(log_weights, ll_target - ll_surrog)
        end
        
        weights = exp.(log_weights)
        ll_is = ll_marginal + log(mean(weights))
        
        # IS estimate should be negative (it's a log-likelihood)
        @test ll_is < 0
        
        # IS estimate should be finite
        @test isfinite(ll_is)
        
        # Marginal should be negative
        @test ll_marginal < 0
    end

end
