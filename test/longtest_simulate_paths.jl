using MultistateModels
using Distributions
using Plots

include("test/setup_2state_exp.jl")

# simulate a single path
MultistateModels.simulate_path(msm, 1)

# simulate a collection of sample paths
paths = simulate(msm; nsim = 10000, paths = true, data = false)

# person 1 has an exponential hazard with rate 0.2
# person 2 has an exponential hazard with rate 0.4
etimes1 = map(x -> x.times[2], paths[1,:])
etimes2 = map(x -> x.times[2], paths[2,:])
mean(etimes1)
mean(etimes2)

# plot histogram of event times
histogram(etimes1, 
          normalize = true,
          label = "Simulated",
          bins = 0:50)
plot!(collect(0:50),
      pdf(Exponential(1/0.2), collect(0:50)),
      lw = 3,
      colour = :red,
      label = "Analytic")
title!("Simulated vs. analytic event times")