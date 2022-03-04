# simulation functions
# simone_path: simulates a single sample path
# sim_paths: wrapper around simeone_path to simulate nsamples x nids number of paths
# simulate: wrapper around sim_paths and maybe other stuff to simulate new data, also incorporates censoring, etc.

"""
    simulate(model::MultistateModel; n = 1, data = true, paths = false, ...)

Simulate `n` datasets or collections of sample paths from a multistate model. If `data = true` (the default) discretely observed sample paths are returned, possibly subject to measurement error. If `paths = false` (the default), continuous-time sample paths are not returned.

# Arguments
- `model::MultistateModel`: object created by multistatemodel()
- `nsim`: number of sample paths to simulate
- `data`: boolean; if true then return discretely observed sample paths
- `paths`: boolean; if false then continuous-time sample paths not returned
"""
function simulate(model::MultistateModel; nsim = 1, data = true, paths = false, ...)

    # throw an error if neither paths nor data are asked for
    if(paths == false & data == false)
        error("Why are you calling `simulate` if you don't want sample paths or data? Stop wasting my time.")
    end

    # number of subjects
    nsubj = length(model.subjectindices)

    # initialize array for simulated paths 
    if paths == true
        samplepaths = Array{SamplePath}(undef, nsubj, nsim)
    end

    # initialize container for simulated data
    # if data == true

    # end 

    for i in Base.OneTo(nsim)
        for j in Base.OneTo(nsubj)
            
            # simulate a path for subject j
            samplepath = simulate_path(model, j)

            # save path if requested
            if path == true
                samplepaths[j, i] = samplepath
            end

            # simulate data
            if data == true
                # simulate data
                # sampledata = simulate_data(samplepath)

                # save data
            end
        end
    end

    # return paths and data
    if paths == false & data == true
    elseif paths == true & data == true
    elseif paths == true & data == false
    end
end

"""
    simulate_path()

Simulate a single sample path.

# Arguments 
- starting state
- transition matrix
- set of hazards    
"""


