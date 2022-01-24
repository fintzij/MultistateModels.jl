# For simulating sample paths
# 1. total hazard for each transient state
# 2. cumulative hazard for each transient state
#       - write/find general hazard integrator function
# 3. solver for next event time
# 4. wrappers/callers
# 5. state initializers? maybe we should insist on the initial state at t0 being known for now? how would it change the structure of the code if we allowed the initial state to be random?
# 6. gamma and gg and semipar
# 7. Function to set model parameters

# For simulating data
# ??? I guess we need to parse the user-supplied dataset for observation schema. This will be clearer after we simulate sample paths since it's a matter of caching the state at observation times/paths over observation intervals and recording states

# Likelihood functions
# 1. Likelihood of sample paths
# 2. Likelihood of data given sample path

# Unit tests
# 1. Check accuracy of hazards, cumulative hazards, total hazard
#       - Check for non Float64 stuff
#       - Edge cases? Zero hazard, infinite hazard, negative hazard (should throw error)
#       - Test for numerical problems (see Distributions.jl for ideas)
# 2. function to validate data
# 3. validate MultistateModel object
# 

# Document internal functions

# Eventually, validation of MCMC
# 1. "Minweke" test 