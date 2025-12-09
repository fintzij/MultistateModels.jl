# =============================================================================
# Simulation Tests
# =============================================================================
#
# Tests verifying:
# 1. Simulation produces statistically correct output (exponential mean = 1/rate)
# 2. Round-trip consistency (simulate -> observe -> extract)
# 3. Edge cases (absorbing states, censoring)
using Optim
using Random
using DataFrames
using MultistateModels: simulate, simulate_data, simulate_paths, simulate_path, observe_path, extract_paths, Hazard, multistatemodel, set_parameters!, _find_jump_time, SamplePath, OptimJumpSolver
using StatsModels: @formula
using .TestFixtures

# --- Optim.jl Brent solver correctness ----------------------------------------
@testset "OptimJumpSolver" begin
    @testset "finds root accurately" begin
        gap_fn = t -> t - 0.5
        os = OptimJumpSolver(abs_tol = 1e-8)
        result = _find_jump_time(os, gap_fn, 0.0, 1.0)
        @test isapprox(result, 0.5; atol = 1e-6)
    end
    
    @testset "nonlinear root" begin
        # Root at t ≈ 0.693 (ln(2))
        gap_fn = t -> exp(t) - 2.0
        os = OptimJumpSolver(abs_tol = 1e-8)
        result = _find_jump_time(os, gap_fn, 0.0, 2.0)
        @test isapprox(result, log(2.0); atol = 1e-5)
    end
end

# --- Statistical correctness --------------------------------------------------
@testset "simulation distribution sanity" begin
    # Verify exponential distribution: E[T] = 1/λ
    fixture = toy_two_state_exp_model(rate = 0.2, horizon = 50.0)
    model = fixture.model
    rate = fixture.rate

    Random.seed!(42)
    nsim = 2000
    durations = Vector{Float64}(undef, nsim)
    for i in 1:nsim
        path = simulate_path(model, 1)
        durations[i] = path.times[end] - path.times[1]
    end

    sample_mean = sum(durations) / nsim
    @test isapprox(sample_mean, 1 / rate; rtol = 0.1, atol = 0.2)
end

# --- Edge cases ---------------------------------------------------------------
@testset "simulate_path edge cases" begin
    @testset "absorbing initial state" begin
        fixture = toy_absorbing_start_model()
        model = fixture.model
        path = simulate_path(model, 1)
        
        @test length(path.times) == 1
        @test length(path.states) == 1
        @test path.states[1] == 3
    end
    
    @testset "censoring branch" begin
        fixture = toy_expwei_model()
        model = deepcopy(fixture.model)
        subj_inds = model.subjectindices[1]
        model.data[subj_inds, :tstop] .= model.data[subj_inds, :tstart] .+ 0.05

        Random.seed!(123)
        path = simulate_path(model, 1)

        @test path.states[end] == path.states[1]
        @test isapprox(path.times[end], model.data[subj_inds[end], :tstop], atol = 1e-12)
    end
end

# --- Round-trip consistency ---------------------------------------------------
@testset "simulate/observe/extract round-trip" begin
    fixture = toy_two_state_exp_model(rate = 0.3, horizon = 30.0)
    model = fixture.model

    for seed in (11, 17, 23)
        Random.seed!(seed)
        path = simulate_path(model, 1)
        obs = observe_path(path, model)
        recovered = extract_paths(obs)[1]

        @test recovered.states == path.states
        @test all(isapprox.(recovered.times, path.times; atol = 1e-10))
    end
end

# --- draw_paths ---------------------------------------------------------------
@testset "draw_paths" begin
    @testset "fixed-count" begin
        fixture = toy_expwei_model()
        model = fixture.model

        result = draw_paths(model; npaths=3, paretosmooth = false, return_logliks = true)

        @test length(result.samplepaths) == length(model.subjectindices)
        @test all(abs(sum(weights) - 1) < 1e-8 for weights in result.ImportanceWeightsNormalized)
    end
    
    @testset "exact-data shortcut" begin
        fixture = toy_fitted_exact_model()
        model = fixture.model

        result = draw_paths(model, 3; paretosmooth = false, return_logliks = true)

        @test result.loglik == fixture.loglik.loglik
        @test result.subj_lml == fixture.loglik.subj_lml
    end
end

# --- Unified simulation API ---------------------------------------------------
@testset "simulate API" begin
    fixture = toy_expwei_model()
    model = fixture.model
    nsubj = length(model.subjectindices)
    nsim = 2

    @testset "simulate_data" begin
        Random.seed!(42)
        data_results = simulate_data(model; nsim = nsim)
        @test length(data_results) == nsim
        @test all(isa(d, DataFrame) for d in data_results)
        @test all("id" in names(d) for d in data_results)
    end

    @testset "simulate_paths" begin
        Random.seed!(42)
        path_results = simulate_paths(model; nsim = nsim)
        @test length(path_results) == nsim
        @test all(length(paths) == nsubj for paths in path_results)
        @test all(sp isa SamplePath for paths in path_results for sp in paths)
    end

    @testset "simulate unified" begin
        # Both data and paths
        Random.seed!(100)
        data_both, paths_both = simulate(model; nsim = nsim, data = true, paths = true)
        @test length(data_both) == nsim
        @test length(paths_both) == nsim
        
        # Error when neither requested
        @test_throws ErrorException simulate(model; data = false, paths = false)
    end
end

# --- newdata, tmax, autotmax arguments ----------------------------------------
@testset "simulate with newdata/tmax/autotmax" begin
    # Simple 2-state exponential model
    h12 = Hazard("exp", 1, 2)
    template = DataFrame(
        id = [1, 1, 2, 2, 3],
        tstart = [0.0, 2.0, 0.0, 3.0, 0.0],
        tstop = [2.0, 5.0, 3.0, 7.0, 4.0],
        statefrom = [1, 1, 1, 1, 1],
        stateto = [1, 1, 1, 1, 1],
        obstype = [1, 1, 1, 1, 1]
    )
    model = multistatemodel(h12; data=template)
    set_parameters!(model, (h12 = [-1.0],))
    
    @testset "autotmax=true (default)" begin
        Random.seed!(12345)
        sim = simulate(model; nsim=1)
        # With autotmax=true, all subjects should have same observation window
        @test all(sim[1].tstop .<= maximum(template.tstop))
        # Should have collapsed to single interval per subject
        @test length(unique(sim[1].id)) == 3
    end
    
    @testset "autotmax=false" begin
        Random.seed!(12345)
        sim = simulate(model; nsim=1, autotmax=false)
        # Original data has multiple rows per subject
        @test nrow(sim[1]) >= 1
    end
    
    @testset "tmax argument" begin
        Random.seed!(12345)
        sim = simulate(model; nsim=1, tmax=10.0)
        @test all(sim[1].tstop .<= 10.0)
        @test length(unique(sim[1].id)) == 3
    end
    
    @testset "newdata argument" begin
        newdata = DataFrame(
            id = 1:5,
            tstart = zeros(5),
            tstop = fill(15.0, 5),
            statefrom = ones(Int, 5),
            stateto = ones(Int, 5),
            obstype = ones(Int, 5)
        )
        Random.seed!(12345)
        sim = simulate(model; nsim=1, newdata=newdata)
        @test length(unique(sim[1].id)) == 5
        @test all(sim[1].tstop .<= 15.0)
    end
    
    @testset "model data restored after simulation" begin
        original_nrow = nrow(model.data)
        original_data = copy(model.data)
        
        # Simulate with tmax (modifies model temporarily)
        sim = simulate(model; nsim=1, tmax=20.0)
        
        # Model should be restored
        @test nrow(model.data) == original_nrow
        @test model.data == original_data
    end
    
    @testset "newdata column validation" begin
        bad_newdata = DataFrame(id = 1:3, tstart = zeros(3), tstop = fill(5.0, 3))
        @test_throws ArgumentError simulate(model; newdata=bad_newdata)
    end
    
    @testset "tmax validation" begin
        @test_throws ArgumentError simulate(model; tmax=-1.0)
    end
    
    @testset "newdata supersedes tmax" begin
        newdata = DataFrame(
            id = 1:2,
            tstart = zeros(2),
            tstop = fill(100.0, 2),
            statefrom = ones(Int, 2),
            stateto = ones(Int, 2),
            obstype = ones(Int, 2)
        )
        Random.seed!(12345)
        # Even though tmax=5.0, newdata should take precedence
        sim = simulate(model; nsim=1, newdata=newdata, tmax=5.0)
        @test length(unique(sim[1].id)) == 2  # 2 subjects from newdata, not 3 from template
    end
end
