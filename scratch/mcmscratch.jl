# for implementing some toy MCMC examples
using Distributions
using Plots

# Example 1 - Sample from Normal(μ, σ^2) via random walk Metropolis-Hastings
μ = 0.0; σ = 1.0

# log likelihood of x
loglik(x) = logpdf(Normal(μ, σ), x[1])

# probability of going from y to z
# first just happens to be normal but doesn't have to be - could be e.g. exponential
propose(y) = rand(Normal(y, 1))[1]
prop1(y,z) = logpdf(Normal(y, 1), z)

# intialize MCMC
val = rand(Normal(0,1))[1]
iters = 10000

loglik_cur = loglik(val)

# container for storing samples
samples = zeros(Float64, iters)

# run that MCMC
for k in Base.OneTo(iters)
    
    # propose new value
    prop = propose(val)
    
    # parts of the MH acceptance probability
    loglik_prop = loglik(prop) # P(x') in wikipedia    
    loglik_cur2prop = prop1(val, prop) # g(x' | x_t) in wikipedia
    loglik_prop2cur = prop1(prop,val) # g(x_t | x') in wikipedia

    # compute MH acceptance probability
    acc_prob = loglik_prop - loglik_cur + loglik_prop2cur - loglik_cur2prop

    # decide whether to accept/reject and make the update
    if(acc_prob > 0 || acc_prob > log(rand(1)[1]))
        # update the state and log likelihood
        val = prop
        loglik_cur = loglik_prop 
    end

    # save the value
    samples[k] = val
end

histogram(samples; n = 50, normalize=true)