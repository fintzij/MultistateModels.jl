using ModelingToolkit
using Catalyst
using Distributions
using DataFrames
using DifferentialEquations
using JumpProcesses
using Plots
using MultistateModels: sample_ecctmc 

# helpers
function census_path(times, timeseq, stateseq)
      # get indices for states
      indseq = map(x -> searchsortedlast(timeseq, x), times)

      return map(x -> x[2] + 1, stateseq[indseq])
end

# for simulating ecctmc
function sim_path(times)
      # simulate jump chain and expand
      jumpchain = MultistateModels.sample_ecctmc(P,Q,1,2,0.0,5.0)
      timeseq = [0.0;jumpchain[1];5.0]
      stateseq = [1;jumpchain[2];2]

      # get indices for states
      indseq = map(x -> searchsortedlast(timeseq, x), times)

      return stateseq[indseq]
end

# Using the SciML ecosystem - rejection sampling
# simulate from a MJP using DifferentialEquations.jl
twostaterecur = @reaction_network begin
    β, S --> D
    μ, D --> S
end β μ

nsim = 100000
p = (0.4, 0.6)
u0 = [1, 0]
tspan = (0.0, 5.0)
times = collect(0.0:0.1:5.0)
prob = DiscreteProblem(twostaterecur, u0, tspan, p)
jump_prob = JumpProblem(twostaterecur, prob, Direct())
sol = [solve(jump_prob, SSAStepper()) for t in 1:(3 * nsim)]

endpoints = map(x -> x.u[end][end], sol)
whichkeep = findall(endpoints .== 1)
keepers = sol[whichkeep]

sciml_paths = reduce(hcat, map(y -> census_path(times, y.t, y.u), keepers))[:,1:nsim]

# using MultistateModels
Q = [-p[1] p[1]; p[2] -p[2]]
P = exp(Q * 5.0)

ecctmc_paths = reduce(hcat,[sim_path(times) for t in 1:nsim])

# summarize results
sciml_props = mean(sciml_paths .== 2, dims = 2)
ecctmc_props = mean(ecctmc_paths .== 2, dims = 2)

plot(times, sciml_props, label = "SciML via rejection sampling")
plot!(times, ecctmc_props, label = "MultistateModels via uniformization")
title!("Proportion in state 2 given X(\$t_0\$ = 1) and X(\$t_1\$=2)\n Huzzah!")