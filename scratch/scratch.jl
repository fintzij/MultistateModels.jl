using DataFrames
using Distributions
using StatsFuns
using LinearAlgebra
using StatsModels
using MultistateModels

#### Minimal model
h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2);
h13 = Hazard(@formula(0 ~ 1 + trt*age), "exp", 1, 3);
h23 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 3);

hazards = (h12, h23, h13)


# obstype: observation scheme
# 0: exactly observed data => tstart,tstop are jump times in a sample path
# 1: interval censored panel data => state known at tstart,tstop, but not path in between
# 2: interval censored, measurement error about states at tstart, tstop => standard HMM
# 3: interval censored, measurement error about path between tstart, tstop (e.g., actt worst state) => e.g., know the worst state between tstart, tstop but not the state at the endpoints

# with the following, we always know where covariates start
# minimal dataset, :statefrom[1] is the initial state, :stateto gets ignored
# [:id :tstart :tstop :statefrom :stateto :obstype :x1 :x2 ...] 

# for exactly observed sample paths
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