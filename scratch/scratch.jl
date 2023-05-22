using DataFrames
using Distributions
using StatsFuns
using LinearAlgebra
using StatsModels
using MultistateModels
using Quadrature
using QuadGK

#### Minimal model
h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2);
h13 = Hazard(@formula(0 ~ 1 + trt*age), "exp", 1, 3);
h23 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 3);

hazards = (h12, h23, h13)


# obstype: observation scheme
# 0: censored data => no state observed
# 1: exactly observed data => tstart,tstop are jump times in a sample path
# 2: interval censored panel data => state known at tstart,tstop, but not path in between
# 3: interval censored, measurement error about path between tstart, tstop (e.g., actt worst state) => e.g., know the worst state between tstart, tstop but not the state at the endpoints

# with the following, we always know where covariates start
# minimal dataset, :statefrom[1] is the initial state, :stateto gets ignored
# [:id :tstart :tstop :statefrom :stateto :obstype :x1 :x2 ...] 

# for exactly observed sample paths
# simulation: :tstart and :tstop are t0 and tmax for each person
# inference: :tstart and :tstop are interval endpoints in a jump chain (might need reshaping to get jump chains for sample paths)

# for interval censored panel data (case 1)
# simulation + inference: :tstart and :tstop are times at which the process is observed

### NOTE 8/25/2022 - discuss different observation schemes and how simulated data should be returned. the current configuration doesn't seem to actually work because, e.g., exactly observed sample paths are returned as interval censored at times given by the original dataset that provides simulation context.

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

### Experiments with quadrature

# functions to be integrated have signature f(x,p) or f(dx,x,p)
# x is the variable of integration, p are parameters, dx is in-place syntax

function testhaz(t,p)
    exp.(p[1] + p[2] * t)
end

prob = QuadratureProblem(testhaz, 0, 1, [0.2, 0.5];  nout=1, batch = 0, N = 50)
@time solve(prob, QuadGKJL())[1]

function th(t,p)
    sin(t)
end

prob = QuadratureProblem(th, 0, pi ; nout=1, reltol=1e-6, order = 5)
@time solve(prob, QuadGKJL())

function th10k()
    for i in 1:100000
        solve(prob, QuadGKJL())
    end
end
@time th10k()

# messing around with QuadGK
d = 2.0
p = 1.2
lb = 0.0
ub = 2.3

quadgk(x -> d + p * sin(x), lb, ub)

function k(p::Float64, lb::Float64, ub::Float64; d::Float64 = d)

    _d = d
    _p = p

    solve(QuadratureProblem((x,_p) -> _d + _p * sin(x), lb, ub, _p), QuadGKJL())
end

function g(p::Float64, lb::Float64, ub::Float64; d::Float64 = d)

    g2 = x -> d + p * sin(x)
    quadgk(x -> d + p * sin(x), lb, ub)[1]
end

@time g(1.2, 0.0, 2.3; d = 3.1)

include("src/MultistateModels.jl")
include("src/common.jl")
include("hazards.jl")











    