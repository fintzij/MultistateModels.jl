# =============================================================================
# test_parallel_likelihood.jl
# =============================================================================
#
# Unit tests for parallel likelihood evaluation functions.
# Verifies that threaded versions produce identical results to sequential.
#
# Run with: julia --threads=4 --project test/test_parallel_likelihood.jl
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random

@testset "Parallel Likelihood Evaluation" begin
    
    @testset "Threading Utilities" begin
        # Test physical core detection
        @test get_physical_cores() >= 1
        @test get_physical_cores() <= Sys.CPU_THREADS
        
        # Test recommended threads
        @test recommended_nthreads() >= 1
        @test recommended_nthreads(task_count=1) == 1
        @test recommended_nthreads(task_count=1000) >= 1
        
        # Test ThreadingConfig
        config_disabled = ThreadingConfig(parallel=false)
        @test !config_disabled.enabled
        @test config_disabled.nthreads == 1
        
        config_enabled = ThreadingConfig(parallel=true)
        @test config_enabled.enabled || Threads.nthreads() == 1
        @test config_enabled.nthreads >= 1
        
        config_explicit = ThreadingConfig(parallel=true, nthreads=2)
        @test config_explicit.nthreads == 2 || Threads.nthreads() < 2
        
        # Test should_parallelize
        @test !should_parallelize(config_disabled, 1000)
        if Threads.nthreads() > 1
            @test should_parallelize(ThreadingConfig(parallel=true, min_batch_size=1), 100)
            @test !should_parallelize(ThreadingConfig(parallel=true, min_batch_size=100), 10)
        end
        
        # Test global config
        old_config = get_threading_config()
        set_threading_config!(parallel=true, nthreads=2)
        new_config = get_threading_config()
        @test new_config.nthreads == 2 || Threads.nthreads() < 2
        set_threading_config!(parallel=old_config.enabled, nthreads=old_config.nthreads)
    end
    
    @testset "loglik_exact: Sequential vs Parallel" begin
        # Setup: 2-state exponential model with exact observations
        Random.seed!(12345)
        n_subjects = 200
        
        # Generate simple exact observation data
        data = DataFrame(
            id = repeat(1:n_subjects, inner=2),
            tstart = repeat([0.0], n_subjects * 2),
            tstop = repeat([1.0], n_subjects * 2),
            statefrom = repeat([1, 2], n_subjects),
            stateto = repeat([2, 2], n_subjects),
            obstype = repeat([1, 1], n_subjects)
        )
        
        # Vary stop times for realism
        for i in 1:nrow(data)
            if data.statefrom[i] == 1
                data.tstop[i] = rand() * 2.0
            else
                data.tstart[i] = data.tstop[i-1]
                data.tstop[i] = data.tstart[i] + rand() * 2.0
            end
        end
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        model = multistatemodel(h12; data=data)
        initialize_parameters!(model)
        
        # Get flat parameters
        params = get_parameters_flat(model)
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        # Compute sequential likelihood
        ll_seq = loglik_exact(params, exact_data; neg=false)
        
        # Compute parallel likelihood (even if single-threaded, should work)
        ll_par = loglik_exact(params, exact_data; neg=false, parallel=true)
        
        @test isapprox(ll_seq, ll_par, rtol=1e-12)
        
        # Test with multiple threads if available
        if Threads.nthreads() > 1
            ll_par_multi = loglik_exact(params, exact_data; neg=false, parallel=true)
            @test isapprox(ll_seq, ll_par_multi, rtol=1e-12)
        end
        
        # Test neg=true
        ll_seq_neg = loglik_exact(params, exact_data; neg=true)
        ll_par_neg = loglik_exact(params, exact_data; neg=true, parallel=true)
        @test isapprox(ll_seq_neg, ll_par_neg, rtol=1e-12)
        @test ll_seq_neg ≈ -ll_seq
    end
    
    @testset "loglik_exact: 3-state model" begin
        # Setup: 3-state illness-death model (1->2, 1->3, 2->3)
        Random.seed!(54321)
        n_subjects = 50
        
        # Create valid illness-death data 
        # Each subject has a single observation interval with exact observation
        rows = []
        for subj in 1:n_subjects
            t_obs = rand() * 5.0 + 0.5
            from_state = 1
            to_state = rand([2, 3])  # Either progresses to 2 or directly to 3
            push!(rows, (id=subj, tstart=0.0, tstop=t_obs, statefrom=from_state, stateto=to_state, obstype=1))
        end
        data = DataFrame(rows)
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h13, h23; data=data)
        initialize_parameters!(model)
        
        params = get_parameters_flat(model)
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        ll_seq = loglik_exact(params, exact_data; neg=false)
        ll_par = loglik_exact(params, exact_data; neg=false, parallel=true)
        
        @test isapprox(ll_seq, ll_par, rtol=1e-12)
        
        if Threads.nthreads() > 1
            ll_par_multi = loglik_exact(params, exact_data; neg=false, parallel=true)
            @test isapprox(ll_seq, ll_par_multi, rtol=1e-12)
        end
    end
    
    @testset "loglik_exact: Weibull hazards" begin
        Random.seed!(99999)
        n_subjects = 100
        
        data = DataFrame(
            id = repeat(1:n_subjects, inner=2),
            tstart = zeros(n_subjects * 2),
            tstop = zeros(n_subjects * 2),
            statefrom = repeat([1, 2], n_subjects),
            stateto = repeat([2, 2], n_subjects),
            obstype = repeat([1, 1], n_subjects)
        )
        
        current_time = 0.0
        current_id = 0
        for i in 1:nrow(data)
            if data.id[i] != current_id
                current_id = data.id[i]
                current_time = 0.0
            end
            data.tstart[i] = current_time
            data.tstop[i] = current_time + rand() * 2.0 + 0.1
            current_time = data.tstop[i]
        end
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        model = multistatemodel(h12; data=data)
        initialize_parameters!(model)
        
        params = get_parameters_flat(model)
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        ll_seq = loglik_exact(params, exact_data; neg=false)
        ll_par = loglik_exact(params, exact_data; neg=false, parallel=true)
        
        @test isapprox(ll_seq, ll_par, rtol=1e-12)
    end
    
    @testset "loglik_exact: With covariates" begin
        Random.seed!(11111)
        n_subjects = 120
        
        data = DataFrame(
            id = repeat(1:n_subjects, inner=2),
            tstart = zeros(n_subjects * 2),
            tstop = zeros(n_subjects * 2),
            statefrom = repeat([1, 2], n_subjects),
            stateto = repeat([2, 2], n_subjects),
            obstype = repeat([1, 1], n_subjects),
            x = repeat(randn(n_subjects), inner=2)
        )
        
        current_time = 0.0
        current_id = 0
        for i in 1:nrow(data)
            if data.id[i] != current_id
                current_id = data.id[i]
                current_time = 0.0
            end
            data.tstart[i] = current_time
            data.tstop[i] = current_time + rand() * 2.0 + 0.1
            current_time = data.tstop[i]
        end
        
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        
        model = multistatemodel(h12; data=data)
        initialize_parameters!(model)
        
        params = get_parameters_flat(model)
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        ll_seq = loglik_exact(params, exact_data; neg=false)
        ll_par = loglik_exact(params, exact_data; neg=false, parallel=true)
        
        @test isapprox(ll_seq, ll_par, rtol=1e-12)
        
        if Threads.nthreads() > 1
            ll_par_multi = loglik_exact(params, exact_data; neg=false, parallel=true)
            @test isapprox(ll_seq, ll_par_multi, rtol=1e-12)
        end
    end
    
    @testset "fit() with parallel=true" begin
        Random.seed!(22222)
        n_subjects = 80
        
        data = DataFrame(
            id = repeat(1:n_subjects, inner=2),
            tstart = zeros(n_subjects * 2),
            tstop = zeros(n_subjects * 2),
            statefrom = repeat([1, 2], n_subjects),
            stateto = repeat([2, 2], n_subjects),
            obstype = repeat([1, 1], n_subjects)
        )
        
        current_time = 0.0
        current_id = 0
        for i in 1:nrow(data)
            if data.id[i] != current_id
                current_id = data.id[i]
                current_time = 0.0
            end
            data.tstart[i] = current_time
            data.tstop[i] = current_time + rand() * 2.0 + 0.1
            current_time = data.tstop[i]
        end
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # Fit without parallel
        model_seq = multistatemodel(h12; data=data)
        fitted_seq = fit(model_seq; parallel=false, verbose=false, 
                         compute_vcov=false, compute_ij_vcov=false)
        
        # Fit with parallel (should give same result even if single-threaded)
        model_par = multistatemodel(h12; data=data)
        fitted_par = fit(model_par; parallel=true, verbose=false,
                         compute_vcov=false, compute_ij_vcov=false)
        
        # Compare parameter estimates
        params_seq = get_parameters_flat(fitted_seq)
        params_par = get_parameters_flat(fitted_par)
        
        @test isapprox(params_seq, params_par, rtol=1e-6)
        
        # Compare log-likelihoods
        ll_seq = get_loglik(fitted_seq)
        ll_par = get_loglik(fitted_par)
        
        @test isapprox(ll_seq, ll_par, rtol=1e-6)
    end
    
    @testset "Thread count specification" begin
        Random.seed!(33333)
        n_subjects = 50
        
        data = DataFrame(
            id = repeat(1:n_subjects, inner=2),
            tstart = zeros(n_subjects * 2),
            tstop = rand(n_subjects * 2) .* 2.0 .+ 0.1,
            statefrom = repeat([1, 2], n_subjects),
            stateto = repeat([2, 2], n_subjects),
            obstype = repeat([1, 1], n_subjects)
        )
        
        current_time = 0.0
        current_id = 0
        for i in 1:nrow(data)
            if data.id[i] != current_id
                current_id = data.id[i]
                current_time = 0.0
            end
            data.tstart[i] = current_time
            data.tstop[i] = current_time + rand() * 2.0 + 0.1
            current_time = data.tstop[i]
        end
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=data)
        
        # Test with explicit nthreads=1
        fitted_1 = fit(model; parallel=true, nthreads=1, verbose=false,
                       compute_vcov=false, compute_ij_vcov=false)
        
        # Test with explicit nthreads=2 (if available)
        if Threads.nthreads() >= 2
            model2 = multistatemodel(h12; data=data)
            fitted_2 = fit(model2; parallel=true, nthreads=2, verbose=false,
                           compute_vcov=false, compute_ij_vcov=false)
            
            # Results should be identical
            @test isapprox(get_parameters_flat(fitted_1), get_parameters_flat(fitted_2), rtol=1e-6)
        end
    end
end

println("Parallel likelihood tests completed!")
println("Julia threads available: $(Threads.nthreads())")
println("Physical cores detected: $(get_physical_cores())")
