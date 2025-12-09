using MultistateModels
using DataFrames
using Statistics
using Random
include("../longtest_mcem.jl")

# Reuse hazards and true params from the Weibull no-covariate test
Random.seed!(0xABCD1234 + 10)
true_shape_12, true_scale_12 = 1.3, 0.15
true_shape_23, true_scale_23 = 1.1, 0.12
true_shape_13, true_scale_13 = 1.2, 0.06

true_params = (
    h12 = [log(true_shape_12), log(true_scale_12)],
    h23 = [log(true_shape_23), log(true_scale_23)],
    h13 = [log(true_shape_13), log(true_scale_13)]
)

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)

panel_data = generate_panel_data_illness_death((h12, h23, h13), true_params)

# Count number of observation rows per subject
by_id = combine(groupby(panel_data, :id), nrow => :n_obs)

mean_obs = mean(by_id.n_obs)
sd_obs = std(by_id.n_obs)

println("Mean observations per subject: ", mean_obs)
println("SD observations per subject: ", sd_obs)
