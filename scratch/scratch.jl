using DataFrames
using Distributions
using StatsFuns
using StatsModels
using MultistateModels

#### Minimal model
h12 = Hazard(@formula(0 ~ trt), "exp", 1, 2);
h13 = Hazard(@formula(0 ~ trt*age), "exp", 1, 3);
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3);

hazards = (h12, h23, h13)

# then go from hazards -> _hazards
# _hazards is an array of mutable structs
# in each struct:
# hazard function, data, parameters, ...
# with(_hazards[1], hazfun(t, parameters, data; give_log))

# work towards this
call_hazard(_hazards, which_hazard, t; loghaz = give_log) # returns the hazard

# or in other words
hazards[which_hazard].hazfun(t, parameters, hazards[which_hazard].data; loghaz = give_log, hazards[which_hazard].inds)

# thinking about how data looks
# Q1: what should a dataset from the user look like?
# Q2: if/how a user-supplied dataset should be reshaped internally?

# obstype: observation scheme
# 0: exactly observed data => tstart,tstop are jump times in a sample path
# 1: interval censored panel data => state known at tstart,tstop, but not path in between
# 2: interval censored, measurement error about states at tstart, tstop => standard HMM
# 3: interval censored, measurement error about path between tstart, tstop (e.g., actt worst state) => e.g., know the worst state between tstart, tstop but not the state at the endpoints

# with the following, we always know where covariates start
# minimal dataset, :statefrom[1] is the initial state, :stateto gets ignored
# [:id :tstart :tstop :statefrom :stateto :obstype :x1 :x2 ...] 

# for exactly observes sample paths
# simulation: :tstart and :tstop are t0 and tmax for each person
# inference: :tstart and :tstop are interval endpoints in a jump chain (might need reshaping to get jump chains for sample paths)

# for interval censored panel data (case 1)
# simulation + inference: :tstart and :tstop are times at which the process is observed

dat_exact = 
    DataFrame(id = collect(1:3),
              tstart = [0, 0, 0],
              tstop = [10, 10, 10],
              statefrom = [1, 2, 1],
              stateto = [3, 3, 3],
              obstype = zeros(3))

dat_exact2 = 
    DataFrame(id = collect(1:3),
              tstart = [0, 0, 0],
              tstop = [10, 10, 10],
              statefrom = [1, 2, 1],
              stateto = [3, 3, 3],
              obstype = zeros(3),
              trt = [0, 1, 0],
              age = [23, 32, 50])

dat_interval = 
    DataFrame(id = [1, 1, 2, 2, 3, 3],
              tstart = [0, 1, 0, 1, 0, 1],
              tstop = [1, 2, 1, 2, 1, 2],
              statefrom = [1, 1, 2, 2, 1, 1], # only the first statefrom counts for simulation
              stateto = [1, 1, 1, 1, 1, 1],
              obstype = ones(6),
              trt = [0, 0, 1, 1, 0, 0],
              age = [23, 23, 32, 32, 50, 50])

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

# see here: https://stackoverflow.com/questions/67123916/julia-function-inside-a-struct-and-method-constructor
Base.@kwdef mutable struct Model2
    p::Float64
    n::Int64
    f::Function
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

f = @formula(0 ~ 1)
s = apply_schema(f, schema(f, dat_exact))
modelcols(s, dat_exact) 

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


````
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