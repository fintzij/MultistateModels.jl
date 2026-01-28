using MultistateModels
using Random
using DataFrames
using LinearAlgebra

# Diagnostic test to check PIJCV correctness

println("="^60)
println("PIJCV CORRECTNESS DIAGNOSTIC")
println("="^60)

# Simple 2-state model
Random.seed!(12345)
n = 100

# Generate data with known true hazard (exponential with rate 0.5)
true_rate = 0.5
survival_data = DataFrame(
    id = 1:n,
    tstart = zeros(n),
    tstop = -log.(rand(n)) / true_rate,  # Exponential survival times
    statefrom = ones(Int, n),
    stateto = fill(2, n),
    obstype = ones(Int, n)
)
# Censor at time 5
for i in 1:n
    if survival_data.tstop[i] > 5.0
        survival_data.tstop[i] = 5.0
        survival_data.stateto[i] = 1
    end
end
println("\nData: n=$n subjects, $(sum(survival_data.stateto .== 2)) events")

# Create spline hazard
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
             degree = 3,
             knots = [1.0, 2.0, 3.0, 4.0],
             boundaryknots = [0.0, 5.0],
             natural_spline = true)
model = multistatemodel(h12; data = survival_data)

npar = model.hazards[1].npar_total
println("Parameters: $npar")

# Fit with different methods
println("\n--- Fitting with EFS ---")
result_efs = fit(model; penalty=:auto, select_lambda=:efs, vcov_type=:none, verbose=true)
println("  λ = $(round.(result_efs.smoothing_parameters, digits=4))")
println("  EDF = $(round(result_efs.edf.total, digits=2))")

println("\n--- Fitting with PIJCV ---")
result_pijcv = fit(model; penalty=:auto, select_lambda=:pijcv, vcov_type=:none, verbose=true)
println("  λ = $(round.(result_pijcv.smoothing_parameters, digits=4))")
println("  EDF = $(round(result_pijcv.edf.total, digits=2))")

# Compare hazard curves
println("\n--- Comparing hazard curves ---")
time_grid = collect(0.1:0.5:4.5)
true_hazards = fill(true_rate, length(time_grid))

# Extract hazard parameters and compute directly using eval_hazard
function get_hazards_at_times(fitted, haz_idx, times)
    haz = fitted.hazards[haz_idx]
    params = MultistateModels.get_hazard_params(fitted.parameters, fitted.hazards)[haz_idx]
    covars = NamedTuple()  # No covariates in this model
    return [MultistateModels.eval_hazard(haz, t, params, covars) for t in times]
end

hazards_efs = get_hazards_at_times(result_efs, 1, time_grid)
hazards_pijcv = get_hazards_at_times(result_pijcv, 1, time_grid)

println("Time | True | EFS | PIJCV")
for (i, t) in enumerate(time_grid)
    println("$(round(t, digits=1)) | $(round(true_hazards[i], digits=3)) | $(round(hazards_efs[i], digits=3)) | $(round(hazards_pijcv[i], digits=3))")
end

# Compute mean squared error
using Statistics: mean
mse_efs = mean((hazards_efs .- true_hazards).^2)
mse_pijcv = mean((hazards_pijcv .- true_hazards).^2)
println("\nMSE EFS: $(round(mse_efs, digits=6))")
println("MSE PIJCV: $(round(mse_pijcv, digits=6))")

# Check if PIJCV is overfitting (should have HIGHER EDF if overfitting)
println("\n--- Overfitting check ---")
println("EDF difference (PIJCV - EFS): $(round(result_pijcv.edf.total - result_efs.edf.total, digits=2))")
if result_pijcv.edf.total > result_efs.edf.total + 1
    println("⚠️  WARNING: PIJCV has significantly higher EDF - may be overfitting!")
else
    println("✓ EDF difference is reasonable")
end
