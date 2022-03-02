# simulation functions
# simone_path: simulates a single sample path
# sim_paths: wrapper around simeone_path to simulate nsamples x nids number of paths
# simulate: wrapper around sim_paths and maybe other stuff to simulate new data, also incorporates censoring, etc.

"""
    simulate(model::MultistateModel; n = 1, data = true, paths = false, ...)

Simulate `n` datasets or collections of sample paths from a multistate model. If `data = true` (the default) discretely observed sample paths are returned, possibly subject to measurement error. If `paths = false` (the default), continuous-time sample paths are not returned.

# Arguments
- `model::MultistateModel`: object created by multistatemodel()
- `n`: number of sample paths to simulate
- `data`: boolean; if true then return discretely observed sample paths
- `paths`: boolean; if false then continuous-time sample paths not returned
"""
function simulate(model::MultistateModel; n = 1, data = true, paths = false, ...)

    # number of subjects
    nsubj = length(model.subjectindices)

    # initialize container if data is to be returned...

    # loop 1:nsim
        # loop 1:nsubj
        # end
    # end
end

"""
    simulate_paths(model::MultistateModel)

Simulates one sample path per subject.

# Arguments
- `model::MultistateModel`: object created by multistatemodel()
"""


"""
    simulate_path()

Simulate a single sample path.

# Arguments 
- starting state
- transition matrix
- set of hazards    
"""


