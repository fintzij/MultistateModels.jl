# ECCTMC transition time distribution diagnostic
# Understanding what transition times are sampled by ECCTMC

using Random
Random.seed!(12345)

println("=" ^ 60)
println("ECCTMC TRANSITION TIME DISTRIBUTION")
println("=" ^ 60)
println()

# For a Markov process with exponential(λ) holding times,
# given a transition occurs in [0, T], where is it?
# 
# For exponential(λ), the transition time T_trans conditioned on T_trans ∈ [0, T] 
# has density:
#   f(t | t < T) = λ exp(-λt) / (1 - exp(-λT))  for t ∈ [0, T]
#
# This is NOT uniform! It's skewed towards earlier times.
# 
# But ECCTMC samples transition times uniformly!

λ = 0.15  # rate
T = 2.5   # interval length

# True conditional density (exponential truncated to [0, T])
truncated_exp_density(t, λ, T) = λ * exp(-λ * t) / (1 - exp(-λ * T))

# Sample from true conditional distribution
function sample_truncated_exp(λ, T)
    # Inverse CDF sampling
    u = rand()
    # CDF: F(t) = (1 - exp(-λt)) / (1 - exp(-λT))
    # Inverse: t = -log(1 - u * (1 - exp(-λT))) / λ
    return -log(1 - u * (1 - exp(-λ * T))) / λ
end

# Compare distributions
n_samples = 100000

# ECCTMC uniform samples
uniform_samples = rand(n_samples) .* T

# True exponential samples (conditioned on occurring before T)
exp_samples = [sample_truncated_exp(λ, T) for _ in 1:n_samples]

println("Comparison for λ=$λ, T=$T:")
println()
println("Uniform (ECCTMC):")
println("  Mean: $(sum(uniform_samples)/n_samples) (expected: $(T/2))")
println("  StdDev: $(std(uniform_samples))")
println()
println("Truncated Exponential (true):")
println("  Mean: $(sum(exp_samples)/n_samples)")
println("  StdDev: $(std(exp_samples))")
println()

# Theoretical mean of truncated exponential
# E[T | T < T_max] = ∫₀^T t f(t) dt = 1/λ - T * exp(-λT) / (1 - exp(-λT))
exp_mean = 1/λ - T * exp(-λ * T) / (1 - exp(-λ * T))
println("  Expected mean: $exp_mean")
println()

# For the last panel interval [12.5, 15.0]:
println("=" ^ 60)
println("Analysis for last panel interval [12.5, 15.0]")
println("=" ^ 60)
println()

T_last = 2.5  # interval length
λ_true = 0.18  # approximate true hazard at late times

exp_mean_last = 1/λ_true - T_last * exp(-λ_true * T_last) / (1 - exp(-λ_true * T_last))

println("If true hazard ≈ $λ_true:")
println("  Expected transition time (from interval start): $(round(exp_mean_last, digits=3))")
println("  Uniform mean: $(T_last/2)")
println("  Difference: $(round(T_last/2 - exp_mean_last, digits=3))")
println()

println("=" ^ 60)
println("KEY INSIGHT")
println("=" ^ 60)
println("""
The ECCTMC sampler draws transition times UNIFORMLY within each panel interval.

For a constant-rate (Markov) process, this IS correct because:
  - Under a homogeneous Poisson process with known number of events,
    the event times are uniformly distributed.

But for a NON-constant hazard (spline), the transition times should NOT
be uniform! Higher hazard regions should have more transitions.

However, ECCTMC is sampling from the Markov SURROGATE, not the target.
The importance weights should correct for this difference.

The question is: are the importance weights correctly accounting for
the difference in transition time distributions?
""")
