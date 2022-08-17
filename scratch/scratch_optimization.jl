using Optimization
using OptimizationOptimJL

objfun = function(t, p)
    ((log(p.cuminc + (1 - p.cuminc) * (1 - survprob(p.timeinstate, p.timeinstate + t), p.parameters, p.ind, p.totalhazards[scur], p.hazards))) - log(p.u))^2
end

p = (cuminc = cuminc, timeinstate = timeinstate, parameters = model.parameters, ind = ind, totalhazards = model.totalhazards, scur = scur, u = u, hazards = model.hazards)

p1 = p

optfun = OptimizationFunction(objfun, Optimization.AutoForwardDiff())

prob = OptimizationProblem(optfun, 0.0, p, lb = 0.0, ub = tstop - tcur)




objfun = function(t, u, cuminc, timeinstate, parameters, scur, ind, totalhazards, hazards)
    ((log(cuminc + (1 - cuminc) * (1 - survprob(timeinstate, timeinstate + t), parameters, ind, totalhazards[scur], hazards))) - log(u))^2
end



# this is ok
function expsurv(t, u, λ)
    (-t[1] * exp(λ) - log(u))^2 # need to index into t
end

u = 0.4; λ = log(0.5)
@benchmark optimize(t -> expsurv(t, u, λ), [0.0,], BFGS(); autodiff = :forward)
@benchmark optimize(t -> expsurv2(t, u, λ), [0.0,], BFGS(); autodiff = :forward)
@benchmark optimize(t -> expsurv(t, u, λ), [0.0,])

# try with types
function expsurv2(t, u::Float64, λ::Float64)
    (-t[1] * exp(λ) - log(u))^2
end

u = 0.4; λ = log(0.5)
optimize(t -> expsurv(t, u, λ), [0.0,], BFGS(); autodiff = :forward)