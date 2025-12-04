# =============================================================================
# Helper Utility Tests
# =============================================================================
#
# Tests that verify critical algorithmic correctness:
# 1. ForwardDiff compatibility (gradients/Hessians work correctly)
# 2. Batched vs sequential likelihood parity (optimization bugs)
using .TestFixtures
using ForwardDiff

# --- ForwardDiff compatibility -------------------------------------------------
# Critical: If gradients/Hessians are wrong, optimization silently fails
@testset "ForwardDiff compatibility" begin
    using MultistateModels: ExactData, loglik_exact
    
    @testset "gradient computation" begin
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
        pars = model.parameters.flat
        
        grad = ForwardDiff.gradient(p -> loglik_exact(p, exact_data; neg=false), pars)
        @test length(grad) == length(pars)
        @test all(isfinite.(grad))
    end
    
    @testset "Hessian computation" begin
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
        pars = model.parameters.flat
        
        hess = ForwardDiff.hessian(p -> loglik_exact(p, exact_data; neg=false), pars)
        @test size(hess) == (length(pars), length(pars))
        @test all(isfinite.(hess))
        @test issymmetric(hess)
    end
end

# --- Batched vs sequential parity ----------------------------------------------
# Critical: Batched optimization must give same answer as sequential
@testset "batched_vs_sequential_parity" begin
    using MultistateModels: SMPanelData, loglik_semi_markov!, loglik_semi_markov_batched!
    
    # Illness-death model tests batched path likelihood
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
    
    base_paths = MultistateModels.extract_paths(model)
    n_subjects = length(base_paths)
    n_paths = 3
    nested_paths = [[deepcopy(base_paths[i]) for _ in 1:n_paths] for i in 1:n_subjects]
    weights = [ones(n_paths) for _ in 1:n_subjects]
    smpanel = SMPanelData(model, nested_paths, weights)
    pars = model.parameters.flat
    
    logliks_seq = [zeros(n_paths) for _ in 1:n_subjects]
    logliks_bat = [zeros(n_paths) for _ in 1:n_subjects]
    loglik_semi_markov!(pars, logliks_seq, smpanel)
    loglik_semi_markov_batched!(pars, logliks_bat, smpanel)
    
    for i in 1:n_subjects, j in 1:n_paths
        @test isapprox(logliks_seq[i][j], logliks_bat[i][j], rtol=1e-12)
    end
end
