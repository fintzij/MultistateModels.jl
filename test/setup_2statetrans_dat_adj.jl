# set up a MultistateModel object
using Chain
using DataFrames
using Distributions
using MultistateModels
using StatsBase

h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)

nsubj = Int64(200)

dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = 5),
              tstart = repeat(collect(0.0:2.0:8.0), outer = nsubj),
              tstop = repeat(collect(2.0:2.0:10.0), outer = nsubj),
              statefrom = fill(1, (5*nsubj)),
              stateto = fill(2, (5*nsubj)),
              obstype = fill(2, (5*nsubj)),
              trt = reduce(vcat, [sample([0,1], 1)[1] * ones(5) for i in 1:nsubj]),
              male = reduce(vcat, [repeat([sample(["m", "f"]),], 5) for i in 1:nsubj]))

# want different gap times
# dat.tstop[1:2:(2*nsubj-1)] = sample([4.0,5.0,6.0], nsubj)
# dat.tstart[2:2:(2*nsubj)] = dat.tstop[1:2:(2*nsubj-1)]

# append!(dat, DataFrame(id=1,tstart=10.0,tstop=20.0,statefrom=2,stateto=1,obstype=2))
# append!(dat, DataFrame(id=1,tstart=20.0,tstop=30.0,statefrom=2,stateto=1,obstype=3))
# sort!(dat, [:id,])

# create multistate model object
msm_2state_transadj = multistatemodel(h12, h21; data = dat)

# set model parameters
# want mean time to event of 5 without treatment
# treatment speeds 1->2 and slows 2->1
set_parameters!(
    msm_2state_transadj, 
    (h12 = [log(0.2),],
     h21 = [log(0.2),]))
    # (h12 = [0.0, log(0.2)],
    #  h21 = [0.0, log(0.2)]))
    # (h12 = [0.0, log(0.2), log(1.5)],
    #  h21 = [0.0, log(0.2), log(2/3)]))

simdat, paths = simulate(msm_2state_transadj; paths = true, data = true);

# create multistate model object with the simulated data
msm_2state_transadj = multistatemodel(h12, h21; data = simdat[1])
set_parameters!(
    msm_2state_transadj, 
    (h12 = [log(0.2)] .+ rand(Normal(), 1)[1],
     h21 = [log(0.2)] .+ rand(Normal(), 1)[1]))
    # (h12 = [0.0, log(0.2)] .+ rand(Normal(), 2)[1:2],
    #  h21 = [0.0, log(0.2)] .+ rand(Normal(), 2)[1:2]))
    # (h12 = [0.0, log(0.2), log(1.5)] .+ rand(Normal(), 3)[1:3],
    #  h21 = [0.0, log(0.2), log(2/3)] .+ rand(Normal(), 3)[1:3]))

msm_2state_transadj.markovsurrogate.parameters[1][1] =  
log(0.2)
# [log(0.2), log(1.5)]

msm_2state_transadj.markovsurrogate.parameters[2][1] =  
log(0.2)
# [log(0.2), log(2/3)]

model = msm_2state_transadj
using ArraysOfArrays, Optimization, OptimizationOptimJL, DifferentialEquations, ExponentialUtilities, ElasticArrays, ForwardDiff, LinearAlgebra, OptimizationOptimisers, Zygote
using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData

nparticles = 10; maxiter = 100; tol = 1e-2; α = 0.1; β = 0.3; γ = 0.05; κ = 3; verbose = true
