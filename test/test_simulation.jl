using Optim
using Random
using DataFrames
using MultistateModels: simulate, simulate_path, observe_path, extract_paths, Hazard, multistatemodel, set_parameters!
using StatsModels: @formula
using .TestFixtures

struct FakeOptimResult
    minimizer::Float64
    converged::Bool
    iterations::Int
end

Optim.converged(res::FakeOptimResult) = res.converged
Optim.iterations(res::FakeOptimResult) = res.iterations

fake_optimize(::Function, ::Float64, ::Float64, ::Optim.Brent; rel_tol = nothing, abs_tol = nothing) = FakeOptimResult(0.0, false, 7)

function fake_optimize_success(::Function, lower::Float64, upper::Float64, ::Optim.Brent; rel_tol = nothing, abs_tol = nothing)
    return FakeOptimResult((lower + upper) / 2, true, 3)
end

@testset "simulate_path error handling" begin
    fixture = toy_expwei_model()
    model = fixture.model

    err = @test_throws ErrorException simulate_path(model, 1, sqrt(eps()), sqrt(eps()); optimize_fn = fake_optimize)
    @test occursin("failed to locate jump time", sprint(showerror, err.value))
    @test occursin("7 iterations", sprint(showerror, err.value))
end

@testset "simulate_path success" begin
    fixture = toy_expwei_model()
    model = fixture.model

    path = simulate_path(model, 1, sqrt(eps()), sqrt(eps()); optimize_fn = fake_optimize_success)

    @test path.subj == 1
    @test length(path.times) == length(path.states)
    @test length(path.times) >= 2
    @test all(diff(path.times) .>= 0)
    @test first(path.states) == fixture.data.statefrom[1]
    @test first(path.times) == fixture.data.tstart[1]
    @test path.times[end] <= maximum(fixture.data.tstop) + 1e-8
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

@testset "simulate time_transform toggle" begin
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
    tt_path = simulate_path(model, 1, sqrt(eps()), sqrt(eps()))
    Random.seed!(515)
    fallback_path = simulate_path(model, 1, sqrt(eps()), sqrt(eps()); time_transform = false)

    @test tt_path.times == fallback_path.times
    @test tt_path.states == fallback_path.states

    Random.seed!(2525)
    dat_tt, paths_tt = simulate(model; nsim = 3, data = true, paths = true, time_transform = true)
    Random.seed!(2525)
    dat_plain, paths_plain = simulate(model; nsim = 3, data = true, paths = true, time_transform = false)

    @test dat_tt == dat_plain
    @test all(isequal.(paths_tt, paths_plain))
end

@testset "simulate multi-subject determinism" begin
    fixture = toy_expwei_model()
    model = fixture.model

    Random.seed!(202)
    dat1, paths1 = simulate(model; nsim = 2, data = true, paths = true)
    Random.seed!(202)
    dat2, paths2 = simulate(model; nsim = 2, data = true, paths = true)

    @test all(isequal.(dat1, dat2))
    @test all(isequal.(paths1, paths2))

    Random.seed!(303)
    _, paths3 = simulate(model; nsim = 2, data = true, paths = true)
    differs = any(!isequal(paths1[i, j], paths3[i, j]) for i in axes(paths1, 1), j in axes(paths1, 2))
    @test differs

    for subj in axes(paths1, 1), sim in axes(paths1, 2)
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
