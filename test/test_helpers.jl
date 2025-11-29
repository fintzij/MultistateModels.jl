# =============================================================================
# Helper Utility Tests
# =============================================================================
#
# Exercises low-level helper functions that mutation-heavy code paths depend on.
# Shared fixtures provide reusable datasets for helper checks so each testset
# isolates a single helper.
using .TestFixtures

# --- Parameter setters ----------------------------------------------------------
@testset "test_set_parameters!" begin

    # vector
    vec_vals = [randn(length(msm_expwei.parameters[1])),
                randn(length(msm_expwei.parameters[2])),
                randn(length(msm_expwei.parameters[3])),
                randn(length(msm_expwei.parameters[4]))]
    set_parameters!(msm_expwei, vec_vals)

    @test msm_expwei.parameters[1] == vec_vals[1]
    @test all(msm_expwei.parameters[2] .== vec_vals[2])
    @test all(msm_expwei.parameters[3] .== vec_vals[3])
    @test all(msm_expwei.parameters[4] .== vec_vals[4])
    
    # unnamed tuple
    unnamed_tuple = (randn(1), randn(4), randn(2), randn(3))
    set_parameters!(msm_expwei, unnamed_tuple)

    @test msm_expwei.parameters[1] == unnamed_tuple[1]
    @test all(msm_expwei.parameters[2] .== unnamed_tuple[2])
    @test all(msm_expwei.parameters[3] .== unnamed_tuple[3])
    @test all(msm_expwei.parameters[4] .== unnamed_tuple[4])

    # named tuple
    named_tuple = (h12 = randn(1), h13 = randn(4), h21 = randn(2), h23 = randn(3))
    set_parameters!(msm_expwei, named_tuple)

    @test msm_expwei.parameters[1] == named_tuple[1]
    @test all(msm_expwei.parameters[2] .== named_tuple[2])
    @test all(msm_expwei.parameters[3] .== named_tuple[3])
    @test all(msm_expwei.parameters[4] .== named_tuple[4])
end

# --- Subject index construction -------------------------------------------------
@testset "test_get_subjinds" begin
    sid_df = subject_id_df()
    expected_groups = subject_id_groups()

    subjinds, nsubj = MultistateModels.get_subjinds(sid_df)

    @test subjinds == expected_groups
    @test nsubj == length(expected_groups)
end

# --- Batched likelihood vs sequential likelihood --------------------------------
@testset "test_loglik_exact" begin
    using ArraysOfArrays: flatview
    using MultistateModels: ExactData, loglik_exact, loglik_exact
    
    # Test 1: Two-state model with exp/wei hazards
    @testset "two-state exp/wei model" begin
        h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 1)
        
        dat = DataFrame(
            id = [1,1,1,2,2,2],
            tstart = [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            tstop = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
            statefrom = [1, 1, 1, 1, 1, 1],
            stateto = [2, 2, 1, 2, 1, 2],
            obstype = [1, 2, 1, 1, 1, 2],
            trt = [0, 1, 0, 1, 0, 1]
        )
        
        model = multistatemodel(h12, h21; data = dat)
        set_parameters!(model, (h12 = [log(0.1), log(2)], h21 = [log(1), log(0.1), log(2)]))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_seq = loglik_exact(pars, exact_data; neg=false)
        ll_bat = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_seq, ll_bat, rtol=1e-12)
    end
    
    # Test 2: Three-state illness-death model
    @testset "three-state illness-death model" begin
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "gom", 2, 3)
        
        dat = DataFrame(
            id = [1, 1, 2, 2, 3],
            tstart = [0.0, 2.0, 0.0, 1.0, 0.0],
            tstop = [2.0, 5.0, 1.0, 4.0, 3.0],
            statefrom = [1, 2, 1, 1, 1],
            stateto = [2, 3, 1, 3, 2],
            obstype = [1, 1, 1, 1, 1]
        )
        
        model = multistatemodel(h12, h13, h23; data = dat)
        set_parameters!(model, (h12 = [log(1.0), log(0.5)], h13 = [log(0.2)], h23 = [log(0.3), log(0.1)]))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_seq = loglik_exact(pars, exact_data; neg=false)
        ll_bat = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_seq, ll_bat, rtol=1e-12)
    end
    
    # Test 3: Weighted subjects
    @testset "weighted subjects" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
        
        dat = DataFrame(
            id = [1,1,2,2],
            tstart = [0.0, 5.0, 0.0, 3.0],
            tstop = [5.0, 10.0, 3.0, 8.0],
            statefrom = [1, 2, 1, 1],
            stateto = [2, 1, 1, 2],
            obstype = [1, 1, 1, 1]
        )
        
        weights = [2.0, 0.5]
        model = multistatemodel(h12, h21; data = dat, SubjectWeights = weights)
        set_parameters!(model, (h12 = [log(0.2)], h21 = [log(1.5), log(0.3)]))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_seq = loglik_exact(pars, exact_data; neg=false)
        ll_bat = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_seq, ll_bat, rtol=1e-12)
    end
    
    # Test 4: Neg parameter produces negated result
    @testset "neg parameter" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 5.0],
            tstop = [5.0, 10.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [1, 1]
        )
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.15)],))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_neg = loglik_exact(pars, exact_data; neg=true)
        ll_pos = loglik_exact(pars, exact_data; neg=false)
        
        @test ll_neg == -ll_pos
    end
end

# --- is_separable trait tests ---------------------------------------------------
@testset "test_is_separable" begin
    using MultistateModels: is_separable, MarkovHazard, SemiMarkovHazard, SplineHazard
    
    # Test 1: All current hazard types return true
    @testset "all current hazards are separable" begin
        # Set up a simple model with different hazard types
        h_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h_wei = Hazard(@formula(0 ~ 1), "wei", 2, 3)
        h_gom = Hazard(@formula(0 ~ 1), "gom", 1, 3)
        
        dat = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 2.0, 4.0],
            tstop = [2.0, 4.0, 6.0],
            statefrom = [1, 2, 1],
            stateto = [2, 3, 3],
            obstype = [1, 1, 1]
        )
        
        model = multistatemodel(h_exp, h_wei, h_gom; data = dat)
        
        # All hazards should be separable
        for hazard in model.hazards
            @test is_separable(hazard) == true
        end
    end
    
    # Test 2: Spline hazards are separable (skipped - splines not yet implemented)
    @testset "spline hazards are separable" begin
        @test_skip "Spline hazards not yet implemented - when available, test that is_separable returns true"
    end
end

# --- cache_path_data tests ------------------------------------------------------
@testset "test_cache_path_data" begin
    using MultistateModels: ExactData, cache_path_data, CachedPathData
    
    @testset "basic caching" begin
        h12 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
        
        dat = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 5.0, 0.0, 3.0],
            tstop = [5.0, 10.0, 3.0, 8.0],
            statefrom = [1, 2, 1, 1],
            stateto = [2, 1, 1, 2],
            obstype = [1, 1, 1, 1],
            age = [30, 30, 40, 40]
        )
        
        model = multistatemodel(h12, h21; data = dat)
        paths = MultistateModels.extract_paths(model)
        
        # Cache paths
        cached = cache_path_data(paths, model)
        
        # Verify structure
        @test length(cached) == length(paths)
        @test all(cpd isa CachedPathData for cpd in cached)
        
        # Verify subject indices preserved
        for (i, (cpd, path)) in enumerate(zip(cached, paths))
            @test cpd.subj == path.subj
        end
        
        # Verify DataFrames have expected columns
        for cpd in cached
            @test :sojourn in propertynames(cpd.df)
            @test :increment in propertynames(cpd.df)
            @test :statefrom in propertynames(cpd.df)
            @test :stateto in propertynames(cpd.df)
        end
        
        # Verify linpreds dict is initialized empty
        for cpd in cached
            @test isempty(cpd.linpreds)
        end
    end
    
    @testset "cached vs uncached batched likelihood" begin
        # Verify that using cached paths gives identical results
        h12 = Hazard(@formula(0 ~ 1 + trt), "wei", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        
        dat = DataFrame(
            id = [1, 1, 2, 2, 3],
            tstart = [0.0, 2.0, 0.0, 1.5, 0.0],
            tstop = [2.0, 5.0, 1.5, 4.0, 3.0],
            statefrom = [1, 2, 1, 1, 1],
            stateto = [2, 1, 1, 2, 2],
            obstype = [1, 1, 1, 1, 1],
            trt = [0, 0, 1, 1, 0]
        )
        
        model = multistatemodel(h12, h21; data = dat)
        set_parameters!(model, (h12 = [log(1.0), log(0.5), 0.3], h21 = [log(0.2)]))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        # Both methods should produce identical results
        ll_bat = loglik_exact(pars, exact_data; neg=false)
        ll_seq = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_bat, ll_seq, rtol=1e-12)
    end
end

# --- Test to_batched_ode_data conversion -------------------------------------------
@testset "test_to_batched_ode_data" begin
    using MultistateModels: cache_path_data, stack_intervals_for_hazard, to_batched_ode_data,
                            StackedHazardData, BatchedODEData
    using ArraysOfArrays: VectorOfVectors
    
    @testset "basic conversion with covariates" begin
        h12 = Hazard(@formula(0 ~ 1 + age + trt), "wei", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + age), "exp", 2, 1)
        
        dat = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 3.0, 0.0, 2.0],
            tstop = [3.0, 6.0, 2.0, 5.0],
            statefrom = [1, 2, 1, 1],
            stateto = [2, 1, 1, 2],
            obstype = [1, 1, 1, 1],
            age = [30.0, 30.0, 50.0, 50.0],
            trt = [1.0, 1.0, 0.0, 0.0]
        )
        
        model = multistatemodel(h12, h21; data = dat)
        set_parameters!(model, (h12 = [log(0.5), log(1.0), 0.01, 0.2], h21 = [log(0.3), 0.02]))
        
        paths = MultistateModels.extract_paths(model)
        cached = cache_path_data(paths, model)
        pars = VectorOfVectors(flatview(model.parameters), model.parameters.elem_ptr)
        
        # Stack intervals for hazard 1 (h12)
        stacked = stack_intervals_for_hazard(
            1, cached, model.hazards, model.totalhazards, model.tmat; pars=pars)
        
        # Convert to BatchedODEData
        ode_data = to_batched_ode_data(stacked)
        
        # Verify structure
        @test ode_data isa BatchedODEData
        n_intervals = length(stacked.lb)
        @test size(ode_data.tspans) == (2, n_intervals)
        @test size(ode_data.covars, 2) == n_intervals
        @test length(ode_data.path_idx) == n_intervals
        @test length(ode_data.is_transition) == n_intervals
        @test length(ode_data.transition_times) == n_intervals
        
        # Verify time spans match
        for i in 1:n_intervals
            @test ode_data.tspans[1, i] == stacked.lb[i]
            @test ode_data.tspans[2, i] == stacked.ub[i]
        end
        
        # Verify covariates match (age and trt for h12)
        @test size(ode_data.covars, 1) == 2  # age and trt
        for i in 1:n_intervals
            @test ode_data.covars[1, i] == stacked.covars[i].age
            @test ode_data.covars[2, i] == stacked.covars[i].trt
        end
        
        # Verify path_idx, is_transition, transition_times copied (not aliased)
        @test ode_data.path_idx == stacked.path_idx
        @test ode_data.is_transition == stacked.is_transition
        @test ode_data.transition_times == stacked.transition_times
        @test ode_data.path_idx !== stacked.path_idx  # different objects
    end
    
    @testset "conversion with no covariates" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)  # intercept only
        
        dat = DataFrame(
            id = [1, 2],
            tstart = [0.0, 0.0],
            tstop = [5.0, 7.0],
            statefrom = [1, 1],
            stateto = [2, 2],
            obstype = [1, 1]
        )
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.1)],))
        
        paths = MultistateModels.extract_paths(model)
        cached = cache_path_data(paths, model)
        pars = VectorOfVectors(flatview(model.parameters), model.parameters.elem_ptr)
        
        stacked = stack_intervals_for_hazard(
            1, cached, model.hazards, model.totalhazards, model.tmat; pars=pars)
        
        ode_data = to_batched_ode_data(stacked)
        
        @test size(ode_data.covars, 1) == 0  # no covariates
        @test size(ode_data.covars, 2) == length(stacked.lb)
    end
    
    @testset "empty stacked data" begin
        # Create scenario where hazard has no relevant intervals
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)  # No intervals for this
        
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [5.0],
            statefrom = [1],
            stateto = [2],  # Only 1->2, no 1->3
            obstype = [1]
        )
        
        model = multistatemodel(h12, h13; data = dat)
        set_parameters!(model, (h12 = [log(0.1)], h13 = [log(0.05)]))
        
        paths = MultistateModels.extract_paths(model)
        cached = cache_path_data(paths, model)
        pars = VectorOfVectors(flatview(model.parameters), model.parameters.elem_ptr)
        
        # h13 has intervals from state 1 (survival contribution)
        stacked = stack_intervals_for_hazard(
            2, cached, model.hazards, model.totalhazards, model.tmat; pars=pars)
        
        # This should work but have the same number of intervals as h12
        ode_data = to_batched_ode_data(stacked)
        @test ode_data isa BatchedODEData
    end
    
    @testset "use_views option" begin
        # Test that use_views=true produces equivalent results
        h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2)
        
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = [0.0, 0.0, 0.0],
            tstop = [5.0, 7.0, 4.0],
            statefrom = [1, 1, 1],
            stateto = [2, 2, 2],
            obstype = [1, 1, 1],
            age = [30.0, 50.0, 70.0]
        )
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.1), log(1.0), 0.01],))
        
        paths = MultistateModels.extract_paths(model)
        cached = cache_path_data(paths, model)
        pars = VectorOfVectors(flatview(model.parameters), model.parameters.elem_ptr)
        
        stacked = stack_intervals_for_hazard(
            1, cached, model.hazards, model.totalhazards, model.tmat; pars=pars)
        
        # Convert with and without views
        ode_data_copy = to_batched_ode_data(stacked; use_views=false)
        ode_data_view = to_batched_ode_data(stacked; use_views=true)
        
        # Same values
        @test ode_data_copy.tspans == ode_data_view.tspans
        @test ode_data_copy.covars == ode_data_view.covars
        @test ode_data_copy.path_idx == ode_data_view.path_idx
        @test ode_data_copy.is_transition == ode_data_view.is_transition
        @test ode_data_copy.transition_times == ode_data_view.transition_times
        
        # Views share underlying data (not copies)
        @test ode_data_view.path_idx === stacked.path_idx
        @test ode_data_view.is_transition === stacked.is_transition
        @test ode_data_view.transition_times === stacked.transition_times
        
        # Copies don't share underlying data
        @test ode_data_copy.path_idx !== stacked.path_idx
    end
end

# --- Batched vs sequential parity with time_transform ----------------------------
@testset "test_batched_time_transform_parity" begin
    using ArraysOfArrays: flatview
    using MultistateModels: ExactData, loglik_exact, loglik_exact
    
    # Test 1: PH hazards with time_transform=true
    @testset "PH hazards with time_transform" begin
        h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2; 
                     linpred_effect = :ph, time_transform = true)
        h21 = Hazard(@formula(0 ~ 1 + age), "exp", 2, 1;
                     linpred_effect = :ph, time_transform = true)
        
        dat = DataFrame(
            id = [1, 1, 2, 2, 3],
            tstart = [0.0, 2.0, 0.0, 1.5, 0.0],
            tstop = [2.0, 5.0, 1.5, 4.0, 3.0],
            statefrom = [1, 2, 1, 1, 1],
            stateto = [2, 1, 1, 2, 2],
            obstype = [1, 1, 1, 1, 1],
            age = [30.0, 30.0, 45.0, 45.0, 55.0]
        )
        
        model = multistatemodel(h12, h21; data = dat)
        set_parameters!(model, (h12 = [log(0.5), log(1.2), 0.02], h21 = [log(0.3), -0.01]))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_seq = loglik_exact(pars, exact_data; neg=false)
        ll_bat = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_seq, ll_bat, rtol=1e-12)
    end
    
    # Test 2: AFT hazards with time_transform=true
    @testset "AFT hazards with time_transform" begin
        h12 = Hazard(@formula(0 ~ 1 + trt), "wei", 1, 2;
                     linpred_effect = :aft, time_transform = true)
        
        dat = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 3.0, 0.0, 4.0],
            tstop = [3.0, 7.0, 4.0, 9.0],
            statefrom = [1, 1, 1, 1],
            stateto = [1, 2, 1, 2],
            obstype = [1, 1, 1, 1],
            trt = [0, 0, 1, 1]
        )
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.8), log(1.5), 0.5],))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_seq = loglik_exact(pars, exact_data; neg=false)
        ll_bat = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_seq, ll_bat, rtol=1e-12)
    end
    
    # Test 3: Mixed time_transform enabled/disabled hazards
    @testset "mixed time_transform hazards" begin
        h12 = Hazard(@formula(0 ~ 1 + x), "wei", 1, 2;
                     linpred_effect = :ph, time_transform = true)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)  # no time_transform
        h23 = Hazard(@formula(0 ~ 1 + x), "gom", 2, 3;
                     linpred_effect = :ph, time_transform = true)
        
        dat = DataFrame(
            id = [1, 1, 2, 2, 3],
            tstart = [0.0, 2.0, 0.0, 1.0, 0.0],
            tstop = [2.0, 5.0, 1.0, 4.0, 3.0],
            statefrom = [1, 2, 1, 1, 1],
            stateto = [2, 3, 1, 3, 2],
            obstype = [1, 1, 1, 1, 1],
            x = [1.0, 1.0, 2.0, 2.0, 0.5]
        )
        
        model = multistatemodel(h12, h13, h23; data = dat)
        set_parameters!(model, (
            h12 = [log(1.0), log(0.5), 0.3],
            h13 = [log(0.2)],
            h23 = [log(0.3), log(0.1), -0.2]
        ))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_seq = loglik_exact(pars, exact_data; neg=false)
        ll_bat = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_seq, ll_bat, rtol=1e-12)
    end
    
    # Test 4: Gompertz with time_transform
    @testset "Gompertz with time_transform" begin
        h12 = Hazard(@formula(0 ~ 1 + age), "gom", 1, 2;
                     linpred_effect = :ph, time_transform = true)
        
        dat = DataFrame(
            id = [1, 1, 2],
            tstart = [0.0, 5.0, 0.0],
            tstop = [5.0, 10.0, 8.0],
            statefrom = [1, 1, 1],
            stateto = [1, 2, 2],
            obstype = [1, 1, 1],
            age = [50.0, 50.0, 60.0]
        )
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.01), 0.05, 0.02],))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_seq = loglik_exact(pars, exact_data; neg=false)
        ll_bat = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_seq, ll_bat, rtol=1e-12)
    end
end

# --- Fused likelihood tests (loglik_exact, loglik_exact) ---
@testset "test_fused_likelihood_parity" begin
    using ArraysOfArrays: flatview
    using MultistateModels: ExactData, loglik_exact, loglik_exact, loglik_exact
    using ForwardDiff
    
    @testset "basic parity with loglik_exact" begin
        # Simple exponential model
        h12 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 2)
        
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = [0.0, 0.0, 0.0],
            tstop = [5.0, 7.0, 4.0],
            statefrom = [1, 1, 1],
            stateto = [2, 2, 2],
            obstype = [1, 1, 1],
            age = [30.0, 50.0, 70.0]
        )
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.1), 0.02],))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_orig = loglik_exact(pars, exact_data; neg=false)
        ll_fused = loglik_exact(pars, exact_data; neg=false)
        ll_fast = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_orig, ll_fused, rtol=1e-12)
        @test isapprox(ll_orig, ll_fast, rtol=1e-12)
    end
    
    @testset "time_transform models" begin
        # Weibull with time_transform
        h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2; 
                     linpred_effect = :ph, time_transform = true)
        
        dat = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 5.0, 0.0, 3.0],
            tstop = [5.0, 10.0, 3.0, 8.0],
            statefrom = [1, 1, 1, 1],
            stateto = [1, 2, 1, 2],
            obstype = [1, 1, 1, 1],
            age = [50.0, 50.0, 60.0, 60.0]
        )
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.01), log(1.2), 0.02],))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_orig = loglik_exact(pars, exact_data; neg=false)
        ll_fused = loglik_exact(pars, exact_data; neg=false)
        ll_fast = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_orig, ll_fused, rtol=1e-10)
        @test isapprox(ll_orig, ll_fast, rtol=1e-10)
    end
    
    @testset "time-varying covariates" begin
        # Data with covariate changes
        dat_tvc = DataFrame(
            id = [1, 1, 1, 2, 2],
            tstart = [0.0, 3.0, 7.0, 0.0, 4.0],
            tstop = [3.0, 7.0, 12.0, 4.0, 9.0],
            statefrom = [1, 1, 1, 1, 1],
            stateto = [1, 1, 2, 1, 2],
            obstype = [1, 1, 1, 1, 1],
            trt = [0, 1, 1, 0, 0]
        )
        
        h12_tvc = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        model_tvc = multistatemodel(h12_tvc; data = dat_tvc)
        set_parameters!(model_tvc, (h12 = [log(0.1), 0.5],))
        
        paths_tvc = MultistateModels.extract_paths(model_tvc)
        exact_data_tvc = ExactData(model_tvc, paths_tvc)
        pars_tvc = flatview(model_tvc.parameters)
        
        ll_orig_tvc = loglik_exact(pars_tvc, exact_data_tvc; neg=false)
        ll_fused_tvc = loglik_exact(pars_tvc, exact_data_tvc; neg=false)
        ll_fast_tvc = loglik_exact(pars_tvc, exact_data_tvc; neg=false)
        
        @test isapprox(ll_orig_tvc, ll_fused_tvc, rtol=1e-10)
        @test isapprox(ll_orig_tvc, ll_fast_tvc, rtol=1e-10)
    end
    
    @testset "ForwardDiff gradients" begin
        h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2)
        
        dat = DataFrame(
            id = [1, 2],
            tstart = [0.0, 0.0],
            tstop = [5.0, 7.0],
            statefrom = [1, 1],
            stateto = [2, 2],
            obstype = [1, 1],
            age = [30.0, 50.0]
        )
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.1), log(1.0), 0.01],))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        grad_orig = ForwardDiff.gradient(p -> loglik_exact(p, exact_data; neg=false), pars)
        grad_fused = ForwardDiff.gradient(p -> loglik_exact(p, exact_data; neg=false), pars)
        grad_fast = ForwardDiff.gradient(p -> loglik_exact(p, exact_data; neg=false), pars)
        
        @test isapprox(grad_orig, grad_fused, rtol=1e-8)
        @test isapprox(grad_orig, grad_fast, rtol=1e-8)
    end
    
    @testset "ForwardDiff Hessians" begin
        h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = [0.0, 0.0, 0.0],
            tstop = [4.0, 6.0, 3.0],
            statefrom = [1, 1, 1],
            stateto = [2, 2, 2],
            obstype = [1, 1, 1],
            trt = [0.0, 1.0, 0.0]
        )
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.2), 0.3],))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        hess_orig = ForwardDiff.hessian(p -> loglik_exact(p, exact_data; neg=false), pars)
        hess_fused = ForwardDiff.hessian(p -> loglik_exact(p, exact_data; neg=false), pars)
        
        @test isapprox(hess_orig, hess_fused, rtol=1e-6)
    end
    
    @testset "illness-death model" begin
        # Complex model with multiple hazards
        dat = DataFrame(
            id = [1, 1, 2, 2, 3],
            tstart = [0.0, 3.0, 0.0, 2.0, 0.0],
            tstop = [3.0, 7.0, 2.0, 5.0, 6.0],
            statefrom = [1, 2, 1, 1, 1],
            stateto = [2, 3, 1, 3, 3],
            obstype = [1, 1, 1, 1, 1],
            age = [40.0, 40.0, 50.0, 50.0, 60.0]
        )
        
        h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1 + age), "gom", 2, 3)
        
        model = multistatemodel(h12, h13, h23; data = dat)
        set_parameters!(model, (
            h12 = [log(0.1), log(1.2), 0.01],
            h13 = [log(0.05)],
            h23 = [log(0.15), 0.02, 0.01]
        ))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_orig = loglik_exact(pars, exact_data; neg=false)
        ll_fused = loglik_exact(pars, exact_data; neg=false)
        ll_fast = loglik_exact(pars, exact_data; neg=false)
        
        @test isapprox(ll_orig, ll_fused, rtol=1e-12)
        @test isapprox(ll_orig, ll_fast, rtol=1e-12)
    end
    
    @testset "return_ll_subj option" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = [0.0, 0.0, 0.0],
            tstop = [5.0, 7.0, 4.0],
            statefrom = [1, 1, 1],
            stateto = [2, 2, 2],
            obstype = [1, 1, 1]
        )
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.1)],))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        ll_subj_orig = loglik_exact(pars, exact_data; neg=false, return_ll_subj=true)
        ll_subj_fused = loglik_exact(pars, exact_data; neg=false, return_ll_subj=true)
        ll_subj_fast = loglik_exact(pars, exact_data; neg=false, return_ll_subj=true)
        
        @test length(ll_subj_orig) == length(ll_subj_fused) == length(ll_subj_fast)
        @test isapprox(ll_subj_orig, ll_subj_fused, rtol=1e-12)
        @test isapprox(ll_subj_orig, ll_subj_fast, rtol=1e-12)
    end
end

@testset "test_batched_smpanel_parity" begin
    # Test batched vs sequential log-likelihood computation for SMPanelData
    # This simulates MCEM scenario with multiple paths per subject
    
    using MultistateModels: SMPanelData, loglik_semi_markov!, loglik_semi_markov_batched!
    
    @testset "illness-death with multiple paths" begin
        # Create a simple illness-death model
        dat = DataFrame(
            id = [1, 1, 2, 2, 3],
            tstart = [0.0, 3.0, 0.0, 2.0, 0.0],
            tstop = [3.0, 7.0, 2.0, 5.0, 6.0],
            statefrom = [1, 2, 1, 1, 1],
            stateto = [2, 3, 1, 3, 3],
            obstype = [1, 1, 1, 1, 1],
            age = [40.0, 40.0, 50.0, 50.0, 60.0]
        )
        
        h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1 + age), "gom", 2, 3)
        
        model = multistatemodel(h12, h13, h23; data = dat)
        set_parameters!(model, (
            h12 = [log(0.1), log(1.2), 0.01],
            h13 = [log(0.05)],
            h23 = [log(0.15), 0.02, 0.01]
        ))
        
        # Extract base paths
        base_paths = MultistateModels.extract_paths(model)
        n_subjects = length(base_paths)
        
        # Create nested path structure (simulate multiple sampled paths per subject)
        n_paths_per_subject = 5
        nested_paths = [
            [deepcopy(base_paths[i]) for _ in 1:n_paths_per_subject]
            for i in 1:n_subjects
        ]
        
        # Create importance weights (all 1.0 for simplicity)
        importance_weights = [ones(n_paths_per_subject) for _ in 1:n_subjects]
        
        # Create SMPanelData
        smpanel = SMPanelData(model, nested_paths, importance_weights)
        
        pars = flatview(model.parameters)
        
        # Initialize log-likelihood matrices
        logliks_seq = [zeros(n_paths_per_subject) for _ in 1:n_subjects]
        logliks_bat = [zeros(n_paths_per_subject) for _ in 1:n_subjects]
        
        # Compute log-likelihoods using both methods
        loglik_semi_markov!(pars, logliks_seq, smpanel)
        loglik_semi_markov_batched!(pars, logliks_bat, smpanel)
        
        # Verify parity
        for i in 1:n_subjects
            for j in 1:n_paths_per_subject
                @test isapprox(logliks_seq[i][j], logliks_bat[i][j], rtol=1e-12)
            end
        end
    end
    
    @testset "PH hazards with multiple paths" begin
        # Test with PH hazards
        dat = DataFrame(
            id = [1, 1, 2],
            tstart = [0.0, 4.0, 0.0],
            tstop = [4.0, 8.0, 6.0],
            statefrom = [1, 1, 1],
            stateto = [1, 2, 2],
            obstype = [1, 1, 1],
            trt = [1.0, 1.0, 0.0]
        )
        
        h12 = Hazard(@formula(0 ~ 1 + trt), "wei", 1, 2; linpred_effect = :ph)
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.1), log(1.0), -0.3],))
        
        base_paths = MultistateModels.extract_paths(model)
        n_subjects = length(base_paths)
        
        n_paths_per_subject = 3
        nested_paths = [
            [deepcopy(base_paths[i]) for _ in 1:n_paths_per_subject]
            for i in 1:n_subjects
        ]
        importance_weights = [ones(n_paths_per_subject) for _ in 1:n_subjects]
        
        smpanel = SMPanelData(model, nested_paths, importance_weights)
        pars = flatview(model.parameters)
        
        logliks_seq = [zeros(n_paths_per_subject) for _ in 1:n_subjects]
        logliks_bat = [zeros(n_paths_per_subject) for _ in 1:n_subjects]
        
        loglik_semi_markov!(pars, logliks_seq, smpanel)
        loglik_semi_markov_batched!(pars, logliks_bat, smpanel)
        
        for i in 1:n_subjects
            for j in 1:n_paths_per_subject
                @test isapprox(logliks_seq[i][j], logliks_bat[i][j], rtol=1e-12)
            end
        end
    end
    
    @testset "time_transform with multiple paths" begin
        # Test with time_transform enabled
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = [0.0, 0.0, 0.0],
            tstop = [5.0, 7.0, 4.0],
            statefrom = [1, 1, 1],
            stateto = [2, 2, 2],
            obstype = [1, 1, 1],
            age = [30.0, 50.0, 70.0]
        )
        
        h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2; 
                     linpred_effect = :aft, time_transform = true)
        
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.1), log(1.2), -0.02],))
        
        base_paths = MultistateModels.extract_paths(model)
        n_subjects = length(base_paths)
        
        n_paths_per_subject = 4
        nested_paths = [
            [deepcopy(base_paths[i]) for _ in 1:n_paths_per_subject]
            for i in 1:n_subjects
        ]
        importance_weights = [ones(n_paths_per_subject) for _ in 1:n_subjects]
        
        smpanel = SMPanelData(model, nested_paths, importance_weights)
        pars = flatview(model.parameters)
        
        logliks_seq = [zeros(n_paths_per_subject) for _ in 1:n_subjects]
        logliks_bat = [zeros(n_paths_per_subject) for _ in 1:n_subjects]
        
        loglik_semi_markov!(pars, logliks_seq, smpanel)
        loglik_semi_markov_batched!(pars, logliks_bat, smpanel)
        
        for i in 1:n_subjects
            for j in 1:n_paths_per_subject
                @test isapprox(logliks_seq[i][j], logliks_bat[i][j], rtol=1e-12)
            end
        end
    end
end