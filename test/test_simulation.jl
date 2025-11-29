using Optim
using Random
using DataFrames
using MultistateModels: simulate, simulate_data, simulate_paths, simulate_path, observe_path, extract_paths, Hazard, multistatemodel, set_parameters!, _find_jump_time_bisection, _find_jump_time, SamplePath, BisectionJumpSolver, OptimJumpSolver, CachedTransformStrategy, DirectTransformStrategy
using StatsModels: @formula
using .TestFixtures

# --- Solver type tests --------------------------------------------------------

@testset "solver type construction" begin
    # BisectionJumpSolver with defaults
    bs_default = BisectionJumpSolver()
    @test bs_default.value_tol == 1e-10
    @test bs_default.max_iters == 80

    # BisectionJumpSolver with custom values
    bs_custom = BisectionJumpSolver(value_tol = 1e-12, max_iters = 100)
    @test bs_custom.value_tol == 1e-12
    @test bs_custom.max_iters == 100

    # OptimJumpSolver with defaults
    os_default = OptimJumpSolver()
    @test os_default.rel_tol == sqrt(sqrt(eps()))
    @test os_default.abs_tol == sqrt(eps())

    # OptimJumpSolver with custom values
    os_custom = OptimJumpSolver(rel_tol = 1e-6, abs_tol = 1e-8)
    @test os_custom.rel_tol == 1e-6
    @test os_custom.abs_tol == 1e-8
end

@testset "strategy type construction" begin
    # Types are singletons, just verify they instantiate
    cached = CachedTransformStrategy()
    direct = DirectTransformStrategy()
    @test cached isa CachedTransformStrategy
    @test direct isa DirectTransformStrategy
end

@testset "solver dispatch" begin
    # Test _find_jump_time dispatch for BisectionJumpSolver
    gap_fn = t -> t - 0.5
    bs = BisectionJumpSolver(value_tol = 1e-10)
    result_bs = _find_jump_time(bs, gap_fn, 0.0, 1.0, 1e-8)
    @test isapprox(result_bs, 0.5; atol = 1e-8)

    # Test _find_jump_time dispatch for OptimJumpSolver
    os = OptimJumpSolver()
    result_os = _find_jump_time(os, gap_fn, 0.0, 1.0, 1e-8)
    @test isapprox(result_os, 0.5; atol = 1e-6)
end

@testset "simulate_path with different solvers" begin
    fixture = toy_two_state_exp_model(rate = 0.2, horizon = 50.0)
    model = fixture.model

    # Test with BisectionJumpSolver (default)
    Random.seed!(42)
    path_bisection = simulate_path(model, 1, sqrt(eps()), sqrt(eps()); solver = BisectionJumpSolver())
    @test path_bisection isa SamplePath

    # Test with OptimJumpSolver
    Random.seed!(42)
    path_optim = simulate_path(model, 1, sqrt(eps()), sqrt(eps()); solver = OptimJumpSolver())
    @test path_optim isa SamplePath

    # Both should produce similar paths (same RNG seed)
    @test path_bisection.states == path_optim.states
    @test all(isapprox.(path_bisection.times, path_optim.times; rtol = 1e-4))
end

@testset "simulate_path absorbing initial state" begin
    fixture = toy_absorbing_start_model()
    model = fixture.model

    path = simulate_path(model, 1, sqrt(eps()), sqrt(eps()))

    @test length(path.times) == 1
    @test length(path.states) == 1
    @test path.states[1] == 3
    @test path.times[1] == model.data[model.subjectindices[1][1], :tstart]
end

@testset "simulate_path censoring branch" begin
    fixture = toy_expwei_model()
    model = deepcopy(fixture.model)
    subj_inds = model.subjectindices[1]
    model.data[subj_inds, :tstop] .= model.data[subj_inds, :tstart] .+ 0.05

    Random.seed!(123)
    path = simulate_path(model, 1, 0.99, sqrt(eps()))

    @test path.states[end] == path.states[1]
    @test isapprox(path.times[end], model.data[subj_inds[end], :tstop], atol = 1e-12)
    @test length(path.times) == 2
end

@testset "simulate_path distribution sanity" begin
    fixture = toy_two_state_exp_model(rate = 0.2, horizon = 50.0)
    model = fixture.model
    rate = fixture.rate

    Random.seed!(42)
    nsim = 2000
    durations = Vector{Float64}(undef, nsim)
    for i in 1:nsim
        path = simulate_path(model, 1, sqrt(eps()), sqrt(eps()))
        durations[i] = path.times[end] - path.times[1]
    end

    sample_mean = sum(durations) / nsim
    @test isapprox(sample_mean, 1 / rate; rtol = 0.1, atol = 0.2)
end

@testset "simulate strategy toggle" begin
    data = DataFrame(
        id = [1],
        tstart = [0.0],
        tstop = [25.0],
        statefrom = [1],
        stateto = [1],
        obstype = [1],
        x = [0.2]
    )

    h12_tt = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2; time_transform = true)
    h13_tt = Hazard(@formula(0 ~ 1), "exp", 1, 3; time_transform = true)
    h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

    model = multistatemodel(h12_tt, h13_tt, h21, h23; data = data)
    set_parameters!(
        model,
        (h12 = [log(0.15), 0.05],
         h13 = [log(0.1)],
         h21 = [log(1.3), log(0.8)],
         h23 = [log(0.9), log(0.6)]))

    Random.seed!(515)
    tt_path = simulate_path(model, 1, sqrt(eps()), sqrt(eps()); strategy = CachedTransformStrategy())
    Random.seed!(515)
    fallback_path = simulate_path(model, 1, sqrt(eps()), sqrt(eps()); strategy = DirectTransformStrategy())

    @test tt_path.times == fallback_path.times
    @test tt_path.states == fallback_path.states

    Random.seed!(2525)
    dat_tt, paths_tt = simulate(model; nsim = 3, data = true, paths = true, strategy = CachedTransformStrategy())
    Random.seed!(2525)
    dat_plain, paths_plain = simulate(model; nsim = 3, data = true, paths = true, strategy = DirectTransformStrategy())

    @test dat_tt == dat_plain
    @test all(isequal.(paths_tt, paths_plain))
end

@testset "simulate multi-subject determinism" begin
    fixture = toy_expwei_model()
    model = fixture.model
    nsubj = length(model.subjectindices)

    Random.seed!(202)
    dat1, paths1 = simulate(model; nsim = 2, data = true, paths = true)
    Random.seed!(202)
    dat2, paths2 = simulate(model; nsim = 2, data = true, paths = true)

    @test all(isequal.(dat1, dat2))
    @test all(isequal.(paths1, paths2))

    Random.seed!(303)
    _, paths3 = simulate(model; nsim = 2, data = true, paths = true)
    
    # paths is a Matrix{SamplePath} with dims (nsubj, nsim)
    differs = any(!isequal(paths1[subj, sim], paths3[subj, sim]) 
                  for subj in 1:nsubj, sim in 1:size(paths1, 2))
    @test differs

    for sim in 1:size(paths1, 2), subj in 1:nsubj
        @test paths1[subj, sim].subj == subj
    end
end

@testset "simulate/observe/extract round-trip" begin
    fixture = toy_two_state_exp_model(rate = 0.3, horizon = 30.0)
    model = fixture.model

    for seed in (11, 17, 23)
        Random.seed!(seed)
        path = simulate_path(model, 1, sqrt(eps()), sqrt(eps()))
        obs = observe_path(path, model)
        recovered = extract_paths(obs)[1]

        @test recovered.states == path.states
        @test all(isapprox.(recovered.times, path.times; atol = 1e-10))
    end
end

@testset "draw_paths fixed-count" begin
    fixture = toy_expwei_model()
    model = fixture.model

    result = draw_paths(model, 3; paretosmooth = false, return_logliks = true)

    @test length(result.samplepaths) == length(model.subjectindices)
    @test all(length(paths) == 3 || length(paths) == 1 for paths in result.samplepaths)
    @test all(abs(sum(weights) - 1) < 1e-8 for weights in result.ImportanceWeightsNormalized)
    @test all(length(ll) == length(paths) for (ll, paths) in zip(result.loglik_target, result.samplepaths))
end

@testset "draw_paths exact-data shortcut" begin
    fixture = toy_fitted_exact_model()
    model = fixture.model

    result = draw_paths(model, 3; paretosmooth = false, return_logliks = true)

    @test result.loglik == fixture.loglik.loglik
    @test result.subj_lml == fixture.loglik.subj_lml
end

# --- New simulation API tests -------------------------------------------------

@testset "simulate_data direct API" begin
    fixture = toy_expwei_model()
    model = fixture.model

    Random.seed!(42)
    data_results = simulate_data(model; nsim = 3)

    @test length(data_results) == 3
    @test all(isa(d, DataFrame) for d in data_results)
    @test all(nrow(d) >= 1 for d in data_results)
    @test all("id" in names(d) for d in data_results)
    @test all("statefrom" in names(d) for d in data_results)
    @test all("stateto" in names(d) for d in data_results)

    # Determinism check
    Random.seed!(42)
    data_results2 = simulate_data(model; nsim = 3)
    @test all(isequal.(data_results, data_results2))
end

@testset "simulate_paths direct API" begin
    fixture = toy_expwei_model()
    model = fixture.model
    nsubj = length(model.subjectindices)
    nsim = 2

    Random.seed!(42)
    path_results = simulate_paths(model; nsim = nsim)

    # Results is a Matrix{SamplePath} with dims (nsubj, nsim) after mapslices with dims=[1,]
    # Actually, it becomes (nsubj, nsim) since mapslices reduces by concatenating subjects
    # Let's verify actual structure
    @test size(path_results, 2) == nsim  # Number of simulations
    @test size(path_results, 1) == nsubj  # Number of subjects
    
    # Each element is a SamplePath
    for sim in 1:nsim
        for subj in 1:nsubj
            sp = path_results[subj, sim]
            @test sp isa SamplePath
            @test sp.subj == subj
        end
    end

    # Determinism check
    Random.seed!(42)
    path_results2 = simulate_paths(model; nsim = nsim)
    @test all(isequal.(path_results, path_results2))
end

@testset "simulate unified wrapper" begin
    fixture = toy_expwei_model()
    model = fixture.model
    nsubj = length(model.subjectindices)
    nsim = 2

    # Data only - returns a result from mapslices that collapses subject dimension
    Random.seed!(100)
    data_only = simulate(model; nsim = nsim, data = true, paths = false)
    @test size(data_only, 2) == nsim
    @test all(isa(d, DataFrame) for d in data_only)

    # Paths only - returns Matrix{SamplePath} with dims (nsubj, nsim)
    Random.seed!(100)
    paths_only = simulate(model; nsim = nsim, data = false, paths = true)
    @test size(paths_only) == (nsubj, nsim)
    @test all(isa(p, SamplePath) for p in paths_only)

    # Both data and paths
    Random.seed!(100)
    data_both, paths_both = simulate(model; nsim = nsim, data = true, paths = true)
    @test size(data_both, 2) == nsim
    @test size(paths_both) == (nsubj, nsim)

    # Error when neither requested
    @test_throws ErrorException simulate(model; data = false, paths = false)
end

# --- Bisection solver edge case tests -----------------------------------------

@testset "bisection solver edge cases" begin
    # Zero-width bracket returns hi
    gap_fn = t -> t - 0.5
    result = _find_jump_time_bisection(gap_fn, 0.5, 0.5, 1e-8)
    @test result == 0.5

    # Immediate root at lower bound (gap_fn(lo) > 0) returns lo
    # If lo_val > 0, the function returns lo immediately
    gap_fn_lo_positive = t -> 1.0 - t  # At t=0.0: 1.0 > 0, so returns 0.0
    result_lo = _find_jump_time_bisection(gap_fn_lo_positive, 0.0, 1.0, 1e-8)
    @test result_lo == 0.0

    # Normal bisection finds root (lo_val < 0, hi_val > 0)
    gap_fn_normal = t -> t - 0.7  # At t=0: -0.7, at t=1: 0.3
    result_normal = _find_jump_time_bisection(gap_fn_normal, 0.0, 1.0, 1e-8; value_tol = 1e-10)
    @test isapprox(result_normal, 0.7; atol = 1e-8)

    # Bracket failure when hi_val < 0
    gap_fn_bad = t -> -1.0  # Always negative
    @test_throws ErrorException _find_jump_time_bisection(gap_fn_bad, 0.0, 1.0, 1e-8)

    # Non-finite value should error
    gap_fn_nan = t -> t < 0.3 ? -0.1 : NaN
    @test_throws ErrorException _find_jump_time_bisection(gap_fn_nan, 0.0, 1.0, 1e-8)

    # Max iterations exceeded (use very tight tolerance and few iters)
    gap_fn_slow = t -> t - 0.123456789
    @test_throws ErrorException _find_jump_time_bisection(gap_fn_slow, 0.0, 1.0, 1e-20; value_tol = 1e-20, max_iters = 3)
end
