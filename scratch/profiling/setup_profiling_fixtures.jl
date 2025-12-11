# Profiling Fixtures for MultistateModels.jl
# Creates test models for benchmarking different configurations

module ProfilingFixtures

using MultistateModels
using DataFrames
using Random

export create_markov_2state, create_markov_3state, 
       create_semimarkov_2state, create_semimarkov_3state,
       create_semimarkov_with_covariates, create_spline_model

# -----------------------------------------------------------------------------
# 2-State Markov Model (Exponential)
# -----------------------------------------------------------------------------
function create_markov_2state(; nsubj=100, seed=12345)
    Random.seed!(seed)
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    
    # Generate panel data with 2 observations per subject
    data = DataFrame(
        id = repeat(1:nsubj, inner=2),
        tstart = repeat([0.0, 5.0], nsubj),
        tstop = repeat([5.0, 10.0], nsubj),
        statefrom = repeat([1, 1], nsubj),
        stateto = repeat([1, 2], nsubj),
        obstype = repeat([2, 2], nsubj)
    )
    
    # Randomize outcomes
    for i in 1:nsubj
        if rand() < 0.3
            data[(i-1)*2 + 1, :stateto] = 2
            data[(i-1)*2 + 2, :statefrom] = 2
            data[(i-1)*2 + 2, :stateto] = 2
        end
    end
    
    model = multistatemodel(h12; data=data)
    set_parameters!(model, (h12 = [log(0.1)],))
    
    return model
end

# -----------------------------------------------------------------------------
# 3-State Progressive Markov Model
# -----------------------------------------------------------------------------
function create_markov_3state(; nsubj=100, seed=12345)
    Random.seed!(seed)
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    data = DataFrame(
        id = repeat(1:nsubj, inner=3),
        tstart = repeat([0.0, 3.0, 6.0], nsubj),
        tstop = repeat([3.0, 6.0, 10.0], nsubj),
        statefrom = repeat([1, 1, 1], nsubj),
        stateto = repeat([1, 1, 1], nsubj),
        obstype = repeat([2, 2, 2], nsubj)
    )
    
    # Randomize progressions
    for i in 1:nsubj
        base = (i-1)*3
        if rand() < 0.4
            data[base + 1, :stateto] = 2
            data[base + 2, :statefrom] = 2
            if rand() < 0.5
                data[base + 2, :stateto] = 3
                data[base + 3, :statefrom] = 3
                data[base + 3, :stateto] = 3
            else
                data[base + 2, :stateto] = 2
                data[base + 3, :statefrom] = 2
                data[base + 3, :stateto] = rand() < 0.5 ? 3 : 2
            end
        end
    end
    
    model = multistatemodel(h12, h23; data=data)
    set_parameters!(model, (h12 = [log(0.15)], h23 = [log(0.2)]))
    
    return model
end

# -----------------------------------------------------------------------------
# 2-State Semi-Markov Model (Weibull)
# -----------------------------------------------------------------------------
function create_semimarkov_2state(; nsubj=100, seed=12345)
    Random.seed!(seed)
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    
    data = DataFrame(
        id = repeat(1:nsubj, inner=2),
        tstart = repeat([0.0, 5.0], nsubj),
        tstop = repeat([5.0, 10.0], nsubj),
        statefrom = repeat([1, 1], nsubj),
        stateto = repeat([1, 2], nsubj),
        obstype = repeat([2, 2], nsubj)
    )
    
    for i in 1:nsubj
        if rand() < 0.3
            data[(i-1)*2 + 1, :stateto] = 2
            data[(i-1)*2 + 2, :statefrom] = 2
            data[(i-1)*2 + 2, :stateto] = 2
        end
    end
    
    model = multistatemodel(h12; data=data)
    set_parameters!(model, (h12 = [log(1.5), log(0.1)],))  # shape=1.5, scale=0.1
    
    return model
end

# -----------------------------------------------------------------------------
# 3-State Semi-Markov with Covariates
# -----------------------------------------------------------------------------
function create_semimarkov_with_covariates(; nsubj=200, seed=12345)
    Random.seed!(seed)
    
    h12 = Hazard(@formula(0 ~ 1 + age + trt), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1 + age), "gom", 2, 3)
    
    data = DataFrame(
        id = repeat(1:nsubj, inner=3),
        tstart = repeat([0.0, 3.0, 6.0], nsubj),
        tstop = repeat([3.0, 6.0, 10.0], nsubj),
        statefrom = repeat([1, 1, 1], nsubj),
        stateto = repeat([1, 1, 1], nsubj),
        obstype = repeat([2, 2, 2], nsubj),
        age = repeat(randn(nsubj), inner=3),
        trt = repeat(rand([0, 1], nsubj), inner=3)
    )
    
    # Randomize outcomes
    for i in 1:nsubj
        base = (i-1)*3
        if rand() < 0.4
            data[base + 1, :stateto] = 2
            data[base + 2, :statefrom] = 2
            if rand() < 0.5
                data[base + 2, :stateto] = 3
                data[base + 3, :statefrom] = 3
                data[base + 3, :stateto] = 3
            else
                data[base + 2, :stateto] = 2
                data[base + 3, :statefrom] = 2
                data[base + 3, :stateto] = rand() < 0.5 ? 3 : 2
            end
        end
    end
    
    model = multistatemodel(h12, h23; data=data)
    set_parameters!(model, (
        h12 = [log(1.2), log(0.15), 0.1, -0.3],  # shape, scale, age, trt
        h23 = [log(0.05), log(0.1), 0.2]          # shape, rate, age
    ))
    
    return model
end

# -----------------------------------------------------------------------------
# Spline Model
# -----------------------------------------------------------------------------
function create_spline_model(; nsubj=100, nknots=3, seed=12345)
    Random.seed!(seed)
    
    h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                 degree=3, extrapolation=:flat, knots=[2.0, 5.0, 8.0])
    
    data = DataFrame(
        id = repeat(1:nsubj, inner=2),
        tstart = repeat([0.0, 5.0], nsubj),
        tstop = repeat([5.0, 10.0], nsubj),
        statefrom = repeat([1, 1], nsubj),
        stateto = repeat([1, 2], nsubj),
        obstype = repeat([2, 2], nsubj)
    )
    
    for i in 1:nsubj
        if rand() < 0.3
            data[(i-1)*2 + 1, :stateto] = 2
            data[(i-1)*2 + 2, :statefrom] = 2
            data[(i-1)*2 + 2, :stateto] = 2
        end
    end
    
    model = multistatemodel(h12; data=data)
    
    return model
end

end # module
