using MultistateModels, DataFrames, StatsModels, Symbolics

#### Minimal model
h12 = Hazard(@formula(lambda12 ~ 1), "exp", 1, 2);
h13 = Hazard(@formula(lambda13 ~ 1), "exp", 1, 3);
h23 = Hazard(@formula(lambda23 ~ 1), "wei", 2, 3);

# data from user can be null, four parameters: 
    # λ12_intercept, λ13_intercept, λ23_intercept_scale, λ23_intercept_shape

# if any hazards depend on covariates, require data. need to check for this.
# if so, a minimal dataset contains enough information to apply_schema

# before applying schema, remove lhs from each formula and save it somewhere.
# if we don't remove it then StatsModels looks for lhs in the dataset

# two objects, θ (parameters) and X (data)
# LHS of the following are log-hazards, RHS parameterizes the log-hazards
# want the following behavior:
# log(λ_12) ~ θ12_intercept + θ12_trt * X_trt + θ12 * BP
# log(λ_13) ~ θ13_intercept
# log(λ_23_scale) ~ θ23_scale_intercept + θ23_scale_trt * X_trt
# log(λ_23_shape) ~ θ_shape_intercept + θ23_scale_trt * X_trt

# Q: how do we get StatsModels to give us the correct design matrix for each hazard? What does the user need to give in terms of data?
# Q: what do we parse out to Symbolics? 

hazards = (h12, h23, h13)

# exponential case
function hazard_exp(t, parameters, data; loghaz = true)

    log_haz = data * parameters

    if loghaz == true 
        return exp(log_haz)
    end

    return log_haz
end

# weibull case
function hazard_weibull(t, parameters, data; loghaz = true, scale_inds, shape_inds)

    # compute parameters
    log_shape = data * parameters[shape_inds] # log(p)
    log_scale = data * parameters[scale_inds] # log(lambda)

    # calculate hazard
    # p=shape, lambda=scale, 
    # h(t)= p * lambda^p* t^(p-1)
    # log(h(t)) = log(p) + p * log(lambda) + (p-1) * log(t)
    log_haz = log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t)

    if loghaz != true 
        return exp(log_haz)
    end

    return log_haz
end

function total_haz(log_hazards...; logtothaz = true)

    log_tot_haz = logsumexp(log_hazards)

    if logtothaz != true
        return exp(log_tot_haz)
    end

    return log_tot_haz
end

i = (a = 5, b = [1, 2, 3])

v = [1 2 3 4]
function changevec(v) 
    return exp(v[2])
end

# some experiments with Julia
using StableRNGs; rng = StableRNG(1);

f = @formula(y ~ 1)
df = DataFrame(y = rand(rng, 9), a = 1:9, b = rand(rng, 9), c = repeat(["a","b","c"], 3), e = rand(rng, 9) * 10)

# going to need to append the variable names to the data frame 
# StatsModels is expecting outcome data
df_blank = DataFrame(xblank = 1.0)
df_blank[:,f.lhs] = 0.0

schema(df)
schema(f, df_blank) 

f = apply_schema(f, schema(f, df)) 

# to get coeficient names as symbols
Meta.parse.(coefnames(f)[2])

while t < tmax
    hazards(pars, data)
    lik = hazards(event) * S(hazards, t0, t1, pars)
    statenext = rand(hazards(tnext) / sum(hazards(tnext)))
end

"""
    parse_hazard(hazards::Hazard)

Takes the formula in a Hazard object and modifies it to have the specified baseline hazard function. 

For exponential hazards, maps @formula(lambda ~ trt) -> @formula(lambda ~ 1 + trt)

For Weibull hazards, maps @formula(lambda ~ trt) -> (@formula(shape ~ 1 + trt), @formula(scale ~ 1 + trt))

shapes = exp.(modelmatrix(weibull_formula_1) \beta_shapes)
scales = exp.(modelmatrix(weibull_formula_2) \beta_scales)

function weibull_hazard(t, shapes, scales) 
    shapes .* (scales .^ shapes) .* (t .^ (shapes .- 1))
end

# f1 is a hazard
function f1(t, x)
    ...
end

# cumulative hazard
function F1(f, t0, t1, x)
    integrate(f, t0, t1, x)
end 

# total hazard
function T(F..., t0, t1, x)
    sum(F...)
end

# survival function
function S(T)
    exp(-T)
end
"""