using LinearAlgebra
using DifferentialEquations

# example of matrix exponential
# stochastic rate matrix
Q = [-2.0 1.0 1.0; 
     1.2 -3.1 1.9;
     0      0   0]

# compute matrix exponential by hand
S = eigen(Q)
Sexp = S.vectors * diagm(exp.(S.values)) * inv(S.vectors)

Q*S.vectors ≈ S.vectors*diagm(S.values)

# what if we want to change the time interval? e.g., t2 - t1 = 5.4
Sexp = S.vectors * diagm(exp.(S.values .* 5.4)) * inv(S.vectors)

# try solving numerically, time homogeneous case
function update_func(du, u, p, t)
     # kolmogorov forward equation
     du = p * u
end

# try solving numerically - time inhomogeneous case
# function update_func(du, u, p, t)
#      # kolmogorov forward equation
#      Q = make_Qmat(p,t) # because Q needs to be updated at each time step, potentially allocating and slow
#      du = u * Q
# end

A = DiffEqArrayOperator(Q)
prob = ODEProblem(A, diagm(ones(3)), (0.0, 5.4), Q )
s = solve(prob, saveat = [1.0, 2.0, 3.0, 4.0, 5.0, 5.4])


# now try as a parameterized function
function update_func(A,u,p,t)
     A[1,2] = p[1]
     A[1,3] = p[2]
     A[1,1] = -p[1] - p[2]
     A[2,1] = p[3]
     A[2,3] = p[4]
     A[2,2] = -p[3] - p[4]
end

# this works
p = [Q[1,2], Q[1,3], Q[2,1], Q[2,3]]
A2 = DiffEqArrayOperator(zeros(3,3), update_func = update_func)
prob2 = ODEProblem(A2, diagm(ones(3)), (0.0, 5.4), p)
s2 = solve(prob2, saveat = [collect(1:5); 5.4])