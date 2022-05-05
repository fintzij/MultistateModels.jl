using MultistateModels
using Distributions
using Plots

include("test/setup_2state_exp_trans.jl")

# simulate a single path
samplepath = MultistateModels.simulate_path(msm, 1)

# simulate a collection of sample paths
paths = simulate(msm; nsim = 100, paths = true, data = true)

# person 1 has an exponential hazard with rate 0.2
# person 2 has an exponential hazard with rate 0.4
etimes1 = map(x -> x.times[2], paths[1,:])
etimes2 = map(x -> x.times[2], paths[2,:])
mean(etimes1) # there's still a bug
mean(etimes2)

# plot histogram of event times

h1 = histogram(etimes1, 
          normalize = true,
          label = "Simulated",
          bins = 0:50);
plot!(h1, collect(0:50),
      pdf(Exponential(1/0.2), collect(0:50)),
      lw = 3,
      colour = :blue,
      label = "Analytic",
      layout = (2,1));
title!(h1, "Simulated vs. analytic event times, control group");

h2 = histogram(etimes2, 
          normalize = true,
          label = "Simulated",
          bins = 0:50,
          colour = :orange);
plot!(h2,collect(0:50),
      pdf(Exponential(1/0.4), collect(0:50)),
      lw = 3,
      colour = :orange,
      label = "Analytic");
title!(h2, "Simulated vs. analytic event times, treatment group");

plot(h1, h2, layout = (2,1))