# Test fit() with different gamma values
using Random, DataFrames, Distributions, MultistateModels
using StatsModels: @formula

println("Setting up test...")
Random.seed!(77777)
nsubj = 30
dat = DataFrame(
    id = 1:nsubj, 
    tstart = zeros(nsubj), 
    tstop = rand(Exponential(2.0), nsubj),
    statefrom = ones(Int, nsubj),
    stateto = fill(2, nsubj),
    obstype = ones(Int, nsubj)
)

h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
model = multistatemodel(h12; data=dat)

println("\n=== Standard PIJCV (γ=1.0) ===")
result1 = fit(model; verbose=true, select_lambda=:pijcv)
lambda1 = result1.smoothing_parameters
println("\nlog(λ) = ", round(log(lambda1[1]), digits=2))

println("\n=== Robust PIJCV (γ=1.4) ===")  
result2 = fit(model; verbose=true, select_lambda=:pijcv_robust)
lambda2 = result2.smoothing_parameters
println("\nlog(λ) = ", round(log(lambda2[1]), digits=2))

println("\n=== Summary ===")
println("Standard: log(λ) = ", round(log(lambda1[1]), digits=2))
println("Robust:   log(λ) = ", round(log(lambda2[1]), digits=2))
