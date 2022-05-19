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