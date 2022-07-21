# optim + autodiff
using Optim 
using ForwardDiff
using QuadGK
using Quadrature
using MultistateModels
using Distributions

# just testing out autodiff
x = rand(Exponential(5), 100000)

exph(x,p) = p
function dexph(dx,x,p) 
    dx .= p
end

qp = QuadratureProblem(exph, 1.0, 2.2, 10.0)
qp2 = QuadratureProblem{true}(dexph, 1.0, 2.2, 10.0)
solve(qp, HCubatureJL())
solve(qp2, HCubatureJL())

@time begin
    for k in 1:1000000
        # remake has minimal overhead
        # solve(qp, QuadGKJL())
        # solve(remake(qp, lb = 1.0, ub = 2.2), QuadGKJL())

        # Making a new QuadratureProblem each time is slowwww
        # qp = QuadratureProblem(exph, 1.0, 2.2, 10.0)
        # solve(qp, QuadGKJL())

        # HCubatureJL also slow, even in-place
        # solve(remake(qp2, lb = 1.0, ub = 2.2), HCubatureJL())

        # HCubatureJL non-in-place is faster than in-place
        # solve(remake(qp, lb = 1.0, ub = 2.2), HCubatureJL())
    end
end

solve(qp, 0, 10, 42)
s = solve(qp, QuadGKJL(), reltol=1e-3, abstol=1e-3,)

function exp_ll(log_theta, x)
    theta = exp(log_theta[1])
    # -length(x) * log_theta[1] + sum(theta .* x)
    -length(x)
end

# calculate exponential MLE
function calc_mle(x)
    exp(optimize(theta -> exp_ll(theta, x), -100.0, 100.0).minimizer)
end

calc_mle(x)

# now with autodiff
function exp_ll2(log_theta)
    exp_ll(log_theta, x)
    # theta = exp.(log_theta[1])
    # -length(x) * log_theta[1] + sum(theta .* x)
end

initial_theta = [1.0]
exp_ll3 = TwiceDifferentiable(exp_ll2, initial_theta; autodiff = :forward)

m = optimize(exp_ll3, initial_theta, BFGS())
ForwardDiff.hessian(exp_ll2,Optim.minimizer(m))

# Weibull example, more similar in structure to MultistateModels
# weibull hazard
function wei_haz(pars,t)
    p = exp(pars[1])
    l = exp(pars[2])
    p * l^p * t^(p-1)
end

function wei_surv(pars, t)

end

# weibull log-likelihood
function wei_ll(log_theta, t, m)
    # set parameters in m
    copyto!(m, log_theta)

    # calculate log-likelihood
end

# quadrature
f(x,p) = sum(sin.(x .* p))
lb = 1
ub = 3
p = 1.5

function testf(p)
    prob = IntegralProblem(f,lb,ub,p)
    sol = solve(prob,QuadGKJL(),reltol=1e-6,abstol=1e-6)[1]
    sin(sol)
end

dp3 = ForwardDiff.derivative(testf,1)


function testfun(p)
    ((p[1] - 1.0))^2
end
td = TwiceDifferentiable(testfun, [1.0]; autodiff = :forward)
Optim.minimizer(optimize(td, [1.0], BFGS()))


function f(x)
    return (1.0 - x[1])^2
end
initial_x = zeros(1)
td = TwiceDifferentiable(f, initial_x; autodiff = :forward)
Optim.minimizer(optimize(td, initial_x, Newton()))

# composing functions
using Optimization
using OptimizationOptimJL
using ForwardDiff


# refactor rosenbrock example
rb1(x,p) = (p[1] - x[1])^2
rb2(x,p) = p[1] * (x[2] - x[1]^2)^2 
rbf(x,p) = rb1(x,p) + rb2(x,p)


rb3(x,p) = (p.vals[p.inds[1]] - x[p.xinds[1]])^2
rb4(x,p) = p.vals[p.inds[2]] * (x[p.xinds[2]] - x[p.xinds[1]]^2)^2
rbf2(x,p) = rb3(x,p) + rb4(x,p)

function rbfw(x,P,i)
    p = P[i,:]
    rbf2(x,p)
end

function rb5(x,p)
    y = view(x, p.xinds)
    (p.vals[p.inds[1]] - y[1])^2
end 
function rb6(x,p)
    y = view(x, p.xinds)
    p.vals[p.inds[2]] * (y[2] - y[1]^2)^2
end

rbf3(x,p) = rb5(x,p) + rb6(x,p)


p1 = (vals = [2.3, 25.0, 40.2], inds = [1,2], xinds = [1,])
p2 = (vals = [2.3, 25.0, 40.2], inds = [2,3], xinds = [1,2])

# function rbf2(x,p)
#     x1 = view(x, 1)
#     x2 = view(x, 1:2)
#     rb1(x1,p) + rb2(x2,p)
# end

x0 = ones(2)
p  = [2.3,25.0,500.0]

ff = OptimizationFunction(rbf3, Optimization.AutoForwardDiff())
prob = OptimizationProblem(ff, x0, p2)
sol = solve(prob, BFGS()) 


##### Exponential example - optimize p to maximize log-likelihood, 
# ll is a composition of hazards and escape probabilities
# first simulate data
using Distributions
using Optimization
using ForwardDiff
using Integrals
using OptimizationOptimJL
XX = rand(Exponential(1), 10000)
p = 1.0; lp = 0.0

# loghaz(t,lp) = lp[1] # or log(total hazard)
loghaz(t, lp) = lp[1]
haz(t, lp) = exp(lp[1])
cumulhaz = IntegralProblem(haz, 0.0, 1.0, lp)

# this solves the integral from 0 to 1 of h(t) = exp(lp[1])
solve(cumulhaz, QuadGKJL())
# this solves the integral from 0 to 2.1 of h(t) = exp(lp[1])
solve(remake(cumulhaz, ub = 2.1, p = 1.2), QuadGKJL())

# how do we pass data with closures?
function loghaz_outer(t, lp, data)
    (t, lp) -> loghaz_inner(t, lp, data)
end

function loghaz_inner(t, lp, data, i)
    lp[1] * data.X[i]
end

ch2 = IntegralProblem((t, lp) -> loghaz_inner(t, lp, data, i), 0.0, 1.0, lp)

# x would be the endpoints of the time interval
# p is anything? or a vector?
function logsurv(parameters, model, rowind) # model object contains data
    -solve(remake(cumulhaz, ub = model.X[rowind], p = parameters[1]), QuadGKJL())[1]
end

# logsurv(x,lp) = -1.0 * x * exp(lp[1]) 

# this works, need to modify this so that p are parameters and X is a model object
function loglik(p, model)
    ll = 0.0
    for k in eachindex(model.X)
        ll += loghaz(model.X[k], p, model.C[k]) + logsurv(p, model, k)
    end
    return(-1.0 * ll)
end

model = (X = XX,)

# now find p using Optimization.jl
optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, [0.1,], model)
sol = solve(prob, BFGS())


haz(x) = 1
solve(IntegralProblem(haz, 0, 10), QuadGKJL(), reltol=1e-3, abstol=1e-3)


# Jon thinks that loghaz can be anything as long as loglik has a strict signature and the integrand in the integrand of the cumulative hazard has a strict signature. 
loghaz(p[MSM.tmat[X.samplepaths[k].statefrom, X.samplepaths[k].stateto]], X.samplepaths[k].tstart, X.samplepaths[k].tstop, otherargs...)
