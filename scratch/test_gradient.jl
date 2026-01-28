# Test script to debug analytical gradient for PIJCV
# Tests the two components: Term 1 (through beta) and Term 2 (direct lambda dependence)

using Random, DataFrames, Distributions, MultistateModels, ForwardDiff, LinearAlgebra
using StatsModels: @formula

import MultistateModels: ExactData, extract_paths, get_parameters_flat,
    build_penalty_config, SplinePenalty, build_implicit_beta_cache, 
    compute_pijcv_with_gradient, _fit_inner_coefficients,
    compute_subject_grads_and_hessians_fast, set_hyperparameters, loglik_exact

println("Setting up test problem...")
Random.seed!(77777)
nsubj = 30
dat = DataFrame(
    id = 1:nsubj, 
    tstart = zeros(nsubj), 
    tstop = [min(rand(Exponential(2.0)), 5.0) for _ in 1:nsubj],
    statefrom = ones(Int, nsubj)
)
dat.stateto = [t < 5.0 ? 2 : 1 for t in dat.tstop]
dat.obstype = [t < 5.0 ? 1 : 2 for t in dat.tstop]

h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
model = multistatemodel(h12; data=dat)
data = ExactData(model, extract_paths(model))
beta_init = get_parameters_flat(model)
penalty = build_penalty_config(model, SplinePenalty())
cache = build_implicit_beta_cache(model, data, penalty, beta_init)
lb, ub = model.bounds.lb, model.bounds.ub

# Test at log(lambda) = 3.0 (near the minimum)
println("\n=== Debugging gradient components at log(λ) = 3.0 ===")
log_lambda = [3.0]
lambda_val = exp.(log_lambda)
penalty_curr = set_hyperparameters(penalty, lambda_val)
beta = _fit_inner_coefficients(model, data, penalty_curr, beta_init; lb=lb, ub=ub, maxiter=50)
g_ll, h_ll = compute_subject_grads_and_hessians_fast(beta, model, data.paths; use_threads=:auto)
subject_grads = -g_ll
subject_hessians = [-H for H in h_ll]
H_unpen = sum(subject_hessians)

n_subjects = size(subject_grads, 2)
n_params = length(beta)

# Build H_lambda
H_lambda = copy(H_unpen)
for term in penalty.terms
    idx = term.hazard_indices
    H_lambda[idx, idx] .+= lambda_val[1] .* term.S
end
S_k = zeros(n_params, n_params)
for term in penalty.terms
    idx = term.hazard_indices
    S_k[idx, idx] .= term.S
end

println("lambda = ", lambda_val[1])
println("beta = ", round.(beta, digits=4))
println("norm(g_total) = ", norm(sum(subject_grads, dims=2)))

# Compute phi_beta = H_lambda^{-1} S_k beta
phi_beta = Symmetric(H_lambda) \ (S_k * beta)
println("norm(phi_beta) = ", norm(phi_beta))
println("S_k * beta = ", round.(S_k * beta, digits=4))

# Compute r_total and each r_i
r_total = zeros(n_params)
term2_components = Float64[]
for i in 1:n_subjects
    g_i = subject_grads[:, i]
    H_i = subject_hessians[i]
    H_lambda_loo = H_lambda - H_i
    delta_i = Symmetric(H_lambda_loo) \ g_i
    r_i = g_i + H_i * delta_i
    r_total .+= r_i
    
    # Term 2 component for this subject
    phi_i = Symmetric(H_lambda_loo) \ (S_k * delta_i)
    term2_i = -lambda_val[1] * dot(r_i, phi_i)
    push!(term2_components, term2_i)
end

println("norm(r_total) = ", norm(r_total))
println("r_total = ", round.(r_total, digits=4))
g_total = vec(sum(subject_grads, dims=2))
println("g_total = ", round.(g_total, digits=4))

# Term 1 (using r_total): -lambda * r_total' * phi_beta
term1_r = -lambda_val[1] * dot(r_total, phi_beta)
println("\nTerm 1 (using r_total) = ", term1_r)

# Term 1 (using g_total): -lambda * g_total' * phi_beta  
term1_g = -lambda_val[1] * dot(g_total, phi_beta)
println("Term 1 (using g_total) = ", term1_g)

# Term 2: sum of per-subject contributions  
term2 = sum(term2_components)
println("Term 2 = ", term2)

# Total analytical gradient (using g_total for Term 1, as in the code now)
grad_anal = term1_g + term2
println("Total analytical gradient (g_total) = ", grad_anal)
grad_anal_r = term1_r + term2
println("Total analytical gradient (r_total) = ", grad_anal_r)

# Now compute FD gradient to compare
eps = 1e-5
lam_p = exp.([3.0 + eps])
pen_p = set_hyperparameters(penalty, lam_p)
beta_p = _fit_inner_coefficients(model, data, pen_p, beta; lb=lb, ub=ub, maxiter=50)
g_p, h_p = compute_subject_grads_and_hessians_fast(beta_p, model, data.paths; use_threads=:auto)
V_p, _ = compute_pijcv_with_gradient(beta_p, [3.0 + eps], cache; 
    subject_grads=-g_p, subject_hessians=[-H for H in h_p], H_unpenalized=sum([-H for H in h_p]))

lam_m = exp.([3.0 - eps])
pen_m = set_hyperparameters(penalty, lam_m)
beta_m = _fit_inner_coefficients(model, data, pen_m, beta; lb=lb, ub=ub, maxiter=50)
g_m, h_m = compute_subject_grads_and_hessians_fast(beta_m, model, data.paths; use_threads=:auto)
V_m, _ = compute_pijcv_with_gradient(beta_m, [3.0 - eps], cache; 
    subject_grads=-g_m, subject_hessians=[-H for H in h_m], H_unpenalized=sum([-H for H in h_m]))

grad_fd = (V_p - V_m) / (2*eps)
println("\nFD gradient = ", grad_fd)
println("Difference = ", grad_anal - grad_fd)

# Also compute partial derivative (beta fixed)
V, _ = compute_pijcv_with_gradient(beta, log_lambda, cache; 
    subject_grads=subject_grads, subject_hessians=subject_hessians, H_unpenalized=H_unpen)
Vp_fixed, _ = compute_pijcv_with_gradient(beta, [3.0 + eps], cache; 
    subject_grads=subject_grads, subject_hessians=subject_hessians, H_unpenalized=H_unpen)
Vm_fixed, _ = compute_pijcv_with_gradient(beta, [3.0 - eps], cache; 
    subject_grads=subject_grads, subject_hessians=subject_hessians, H_unpenalized=H_unpen)
partial_fd = (Vp_fixed - Vm_fixed) / (2*eps)
println("\nPartial FD (beta fixed) = ", partial_fd)
println("This should match Term 2 = ", term2)

# Implied Term 1 from FD
term1_fd = grad_fd - partial_fd
println("Implied Term 1 from FD = ", term1_fd)
println("Our Term 1 (g_total) = ", term1_g)
println("Our Term 1 (r_total) = ", term1_r)

# Let's verify phi_beta = H_lambda^{-1} S_k beta by computing dbeta/dlambda numerically
println("\n=== Verifying d(beta)/d(lambda) ===")
eps_lam = 1e-5
lam_p2 = lambda_val[1] * (1 + eps_lam)
pen_p2 = set_hyperparameters(penalty, [lam_p2])
beta_p2 = _fit_inner_coefficients(model, data, pen_p2, beta; lb=lb, ub=ub, maxiter=50)
lam_m2 = lambda_val[1] * (1 - eps_lam)
pen_m2 = set_hyperparameters(penalty, [lam_m2])
beta_m2 = _fit_inner_coefficients(model, data, pen_m2, beta; lb=lb, ub=ub, maxiter=50)
dbeta_dlambda_fd = (beta_p2 - beta_m2) / (2 * eps_lam * lambda_val[1])
println("dbeta/dlambda (FD): ", round.(dbeta_dlambda_fd, digits=6))
phi_beta_analytical = -Symmetric(H_lambda) \ (S_k * beta)  # Note the minus sign
println("-H_lambda^{-1} S_k beta: ", round.(phi_beta_analytical, digits=6))

# Now let's look at d(beta)/d(rho) = d(beta)/d(lambda) * lambda
dbeta_drho_fd = dbeta_dlambda_fd * lambda_val[1]
println("dbeta/drho (FD): ", round.(dbeta_drho_fd, digits=6))
phi_beta_rho = -lambda_val[1] * (Symmetric(H_lambda) \ (S_k * beta))
println("-lambda * H_lambda^{-1} S_k beta: ", round.(phi_beta_rho, digits=6))

# Now let's test the CONSISTENT gradient (with frozen g/H throughout)
println("\n=== Testing gradient with FROZEN g/H ===")
# When we fix g/H but vary beta and lambda, the gradient should be:
# dV/drho = (dV/dbeta)*(dbeta/drho) + (dV/dlambda)*lambda
# where dV/dbeta = g_total (under frozen g/H) and dV/dlambda is the partial

# FD with frozen g/H: perturb rho, re-fit beta, but keep g/H fixed
function V_frozen_gH(log_lam_val, beta_start, sg_fixed, sh_fixed, H_unpen_fixed)
    lam = exp.([log_lam_val])
    pen = set_hyperparameters(penalty, lam)
    beta_new = _fit_inner_coefficients(model, data, pen, beta_start; lb=lb, ub=ub, maxiter=50)
    V, _ = compute_pijcv_with_gradient(beta_new, [log_lam_val], cache;
        subject_grads=sg_fixed, subject_hessians=sh_fixed, H_unpenalized=H_unpen_fixed)
    return V, beta_new
end

V_frozen0, _ = V_frozen_gH(3.0, beta, subject_grads, subject_hessians, H_unpen)
V_frozen_p, _ = V_frozen_gH(3.0 + eps, beta, subject_grads, subject_hessians, H_unpen)
V_frozen_m, _ = V_frozen_gH(3.0 - eps, beta, subject_grads, subject_hessians, H_unpen)
grad_fd_frozen_gH = (V_frozen_p - V_frozen_m) / (2*eps)

println("V (frozen g/H) = ", V_frozen0)
println("FD gradient (frozen g/H) = ", grad_fd_frozen_gH)
println("Our analytical gradient = ", grad_anal)

println("\n=== Comparing gradients ===")
println("FD full (g/H updated):     ", grad_fd)
println("FD frozen g/H:             ", grad_fd_frozen_gH)
println("Analytical (g_total):      ", grad_anal)
println("Analytical (r_total):      ", grad_anal_r)

# Now let's scan to see where the minima are for each objective
println("\n=== Scanning to find minima ===")
println("log(λ)    V_full       V_frozen     grad_full    grad_frozen")
println("-" ^ 70)
for log_lam_test in 2.0:0.5:5.0
    # Full V (g/H updated)
    lam_t = exp.([log_lam_test])
    pen_t = set_hyperparameters(penalty, lam_t)
    beta_t = _fit_inner_coefficients(model, data, pen_t, beta_init; lb=lb, ub=ub, maxiter=50)
    g_t, h_t = compute_subject_grads_and_hessians_fast(beta_t, model, data.paths; use_threads=:auto)
    V_full_t, grad_full_t = compute_pijcv_with_gradient(beta_t, [log_lam_test], cache;
        subject_grads=-g_t, subject_hessians=[-H for H in h_t], H_unpenalized=sum([-H for H in h_t]))
    
    # Frozen V
    V_frozen_t, grad_frozen_t = compute_pijcv_with_gradient(beta_t, [log_lam_test], cache;
        subject_grads=subject_grads, subject_hessians=subject_hessians, H_unpenalized=H_unpen)
    
    println("$(round(log_lam_test, digits=1))       $(round(V_full_t, digits=3))       $(round(V_frozen_t, digits=3))        $(round(grad_full_t[1], digits=3))        $(round(grad_frozen_t[1], digits=3))")
end

# Test: use gradient with UPDATED g/H at each step
println("\n=== Testing HYBRID gradient: FD for Term1, analytical for Term2 ===")
log_lam_opt = 2.0  # Start at low lambda
beta_curr = copy(beta_init)

# Hybrid gradient: Term2 analytical, Term1 via FD
function compute_hybrid_gradient(log_lam_val, beta_start)
    lam_val = exp.([log_lam_val])
    pen_val = set_hyperparameters(penalty, lam_val)
    beta_opt = _fit_inner_coefficients(model, data, pen_val, beta_start; lb=lb, ub=ub, maxiter=50)
    
    # Compute g/H at current beta
    g_opt, h_opt = compute_subject_grads_and_hessians_fast(beta_opt, model, data.paths; use_threads=:auto)
    sg_opt = -g_opt
    sh_opt = [-H for H in h_opt]
    H_unpen_opt = sum(sh_opt)
    
    # Full V at current point
    V_curr, grad_approx = compute_pijcv_with_gradient(beta_opt, [log_lam_val], cache;
        subject_grads=sg_opt, subject_hessians=sh_opt, H_unpenalized=H_unpen_opt)
    
    # Term 2 (partial, beta fixed): use FD with frozen g/H
    eps_fd = 1e-5
    Vp_partial, _ = compute_pijcv_with_gradient(beta_opt, [log_lam_val + eps_fd], cache;
        subject_grads=sg_opt, subject_hessians=sh_opt, H_unpenalized=H_unpen_opt)
    Vm_partial, _ = compute_pijcv_with_gradient(beta_opt, [log_lam_val - eps_fd], cache;
        subject_grads=sg_opt, subject_hessians=sh_opt, H_unpenalized=H_unpen_opt)
    term2_fd = (Vp_partial - Vm_partial) / (2*eps_fd)
    
    # Term 1 (beta dependence): get from full FD - term2
    # Full FD: perturb lambda, re-fit beta, recompute g/H
    lam_p = exp.([log_lam_val + eps_fd])
    pen_p = set_hyperparameters(penalty, lam_p)
    beta_p = _fit_inner_coefficients(model, data, pen_p, beta_opt; lb=lb, ub=ub, maxiter=50)
    g_p, h_p = compute_subject_grads_and_hessians_fast(beta_p, model, data.paths; use_threads=:auto)
    Vp_full, _ = compute_pijcv_with_gradient(beta_p, [log_lam_val + eps_fd], cache;
        subject_grads=-g_p, subject_hessians=[-H for H in h_p], H_unpenalized=sum([-H for H in h_p]))
    
    lam_m = exp.([log_lam_val - eps_fd])
    pen_m = set_hyperparameters(penalty, lam_m)
    beta_m = _fit_inner_coefficients(model, data, pen_m, beta_opt; lb=lb, ub=ub, maxiter=50)
    g_m, h_m = compute_subject_grads_and_hessians_fast(beta_m, model, data.paths; use_threads=:auto)
    Vm_full, _ = compute_pijcv_with_gradient(beta_m, [log_lam_val - eps_fd], cache;
        subject_grads=-g_m, subject_hessians=[-H for H in h_m], H_unpenalized=sum([-H for H in h_m]))
    
    grad_full_fd = (Vp_full - Vm_full) / (2*eps_fd)
    
    return V_curr, grad_full_fd, beta_opt
end

for iter in 1:15
    global log_lam_opt, beta_curr
    
    V_opt, grad_opt, beta_curr = compute_hybrid_gradient(log_lam_opt, beta_curr)
    
    # Gradient descent step
    step_size = 0.5
    step = -step_size * grad_opt
    step_clamped = clamp(step, -0.5, 0.5)
    
    println("Iter $iter: log(λ)=$(round(log_lam_opt, digits=3)), V=$(round(V_opt, digits=4)), grad=$(round(grad_opt, digits=4)), step=$(round(step_clamped, digits=4))")
    
    if abs(grad_opt) < 0.05
        println("Converged!")
        break
    end
    
    log_lam_opt += step_clamped
    log_lam_opt = clamp(log_lam_opt, 0.0, 8.0)
end
