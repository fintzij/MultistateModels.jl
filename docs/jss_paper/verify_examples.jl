using MultistateModels
using DataFrames
using Random
using LinearAlgebra
using StatsModels

println("Running Example 1: Illness-Death Model with Panel Data")
Random.seed!(12345)

# Define hazards
h12 = Hazard(:wei, 1, 2)  # Healthy → Ill
h23 = Hazard(:wei, 2, 3)  # Ill → Dead  
h13 = Hazard(:wei, 1, 3)  # Healthy → Dead

# Generate panel data template
n_subjects = 200
obs_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
n_obs = length(obs_times) - 1

panel_template = DataFrame(
    id = repeat(1:n_subjects, inner=n_obs),
    tstart = repeat(obs_times[1:end-1], n_subjects),
    tstop = repeat(obs_times[2:end], n_subjects),
    statefrom = 1,
    stateto = 1,
    obstype = 2
)

# Create model for simulation
sim_model = multistatemodel(h12, h23, h13; 
                            data=panel_template, 
                            surrogate=:markov)

# Set true parameters (log-shape, log-scale for Weibull)
set_parameters!(sim_model, (
    h12 = [log(1.5), log(5.0)],   # Shape=1.5, Scale=5.0
    h23 = [log(1.8), log(3.0)],   # Shape=1.8, Scale=3.0
    h13 = [log(1.2), log(10.0)]   # Shape=1.2, Scale=10.0
))

# Simulate data
println("Simulating data...")
# Use autotmax=false to respect the panel structure (multiple intervals per subject)
sim_result = simulate(sim_model; nsim=1, paths=false, data=true, autotmax=false)
panel_data = sim_result[1, 1]
println("Panel data dimensions (autotmax=false): ", size(panel_data))

# Fit model using MCEM
println("Fitting model (this may take a moment)...")
fit_model = multistatemodel(h12, h23, h13;
                            data=panel_data,
                            surrogate=:markov)

fitted = fit(fit_model;
    verbose = true,
    compute_vcov = true,
    vcov_type = :ij,
    maxiter = 10, # Reduced for verification speed
    tol = 0.1,    # Relaxed for verification speed
    ess_target_initial = 10 # Reduced for verification speed
)

# Results
println("Fitted parameters:")
println(get_parameters(fitted))
println("\nStandard errors:")
println(sqrt.(diag(get_vcov(fitted))))

println("\nRunning Example 2: Model with Covariates")
println("Panel data dimensions: ", size(panel_data))

# Hazard with treatment effect
h12_cov = Hazard(@formula(0 ~ treatment), :wei, 1, 2)
h23_cov = Hazard(@formula(0 ~ treatment), :wei, 2, 3)
h13_cov = Hazard(@formula(0 ~ treatment), :wei, 1, 3)

# Data must include treatment column
# Robust assignment of treatment
# panel_data.treatment = repeat([0, 1], inner=n_subjects÷2 * n_obs) # This was causing issues if rows changed
# Assign treatment: first half of subjects = 0, second half = 1
u_ids = unique(panel_data.id)
n_u = length(u_ids)
treat_map = Dict(id => (i <= n_u/2 ? 0 : 1) for (i, id) in enumerate(u_ids))
panel_data.treatment = [treat_map[id] for id in panel_data.id]

model_cov = multistatemodel(h12_cov, h23_cov, h13_cov;
                            data=panel_data,
                            surrogate=:markov)

println("Fitting covariate model...")
fitted_cov = fit(model_cov; 
    verbose=true, 
    compute_vcov=true,
    maxiter = 5, # Reduced for verification speed
    tol = 0.5,   # Relaxed for verification speed
    ess_target_initial = 10
)

# Treatment hazard ratios
params = get_parameters(fitted_cov)
hr_12 = exp(params.h12[end])  # Last parameter is treatment effect
println("Treatment HR for 1→2: ", round(hr_12, digits=3))

println("\nVerification complete!")
