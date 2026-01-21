# Diagnostic: Check simulation from spline model
using MultistateModels
using DataFrames
using Random
using Statistics

Random.seed!(12345)

# Configuration matching the test
N_SUBJECTS = 1000
PANEL_TIMES = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
TRUE_COEFS_H12 = [0.08, 0.10, 0.14, 0.18]

# Create spline hazard
h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
    degree=3,
    knots=[5.0, 10.0],
    boundaryknots=[0.0, 15.0],
    extrapolation="constant")

h23 = Hazard(@formula(0 ~ 1), :sp, 2, 3;
    degree=3,
    knots=[5.0, 10.0],
    boundaryknots=[0.0, 15.0],
    extrapolation="constant")

# Create panel data template
nobs = length(PANEL_TIMES) - 1
template = DataFrame(
    id = repeat(1:N_SUBJECTS, inner=nobs),
    tstart = repeat(PANEL_TIMES[1:end-1], N_SUBJECTS),
    tstop = repeat(PANEL_TIMES[2:end], N_SUBJECTS),
    statefrom = ones(Int, N_SUBJECTS * nobs),
    stateto = ones(Int, N_SUBJECTS * nobs),
    obstype = fill(2, N_SUBJECTS * nobs)
)

# Build model
model = multistatemodel(h12, h23; data=template, initialize=false)
set_parameters!(model, (h12 = TRUE_COEFS_H12, h23 = [0.06, 0.08, 0.11, 0.14]))

# Check the hazard function at various times
println("=== True hazard values from spline model ===")
pars_12 = MultistateModels.get_parameters(model, 1, scale=:log)
println("h12 parameters (log scale): $pars_12")
pars_12_nat = MultistateModels.get_parameters(model, 1, scale=:natural)
println("h12 parameters (natural scale): $pars_12_nat")

for t in [1.0, 5.0, 10.0, 15.0]
    h = model.hazards[1](t, pars_12, NamedTuple())
    println("  h(t=$t) = $(round(h, digits=4))")
end

# Simulate data
println("\n=== Simulating data ===")
obstype_map = Dict(1 => 2, 2 => 1)  # h12 panel, h23 exact
sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=obstype_map)
panel_data = sim_result[1, 1]

# Check state distribution at each observation time
println("\n=== State distribution at panel times ===")
for t in PANEL_TIMES[2:end]
    obs_at_t = panel_data[panel_data.tstop .== t, :]
    if nrow(obs_at_t) > 0
        n_state1 = sum(obs_at_t.stateto .== 1)
        n_state2 = sum(obs_at_t.stateto .== 2)
        n_state3 = sum(obs_at_t.stateto .== 3)
        total = nrow(obs_at_t)
        println("  t=$t: State 1: $n_state1 ($(round(100*n_state1/total, digits=1))%), " *
                "State 2: $n_state2 ($(round(100*n_state2/total, digits=1))%), " *
                "State 3: $n_state3 ($(round(100*n_state3/total, digits=1))%)")
    end
end

# Calculate empirical transition rates
println("\n=== Empirical transition analysis ===")
# Find subjects who transitioned from 1 to 2
trans_12 = panel_data[(panel_data.statefrom .== 1) .& (panel_data.stateto .== 2), :]
println("Number of 1→2 transitions observed: $(nrow(trans_12))")
if nrow(trans_12) > 0
    println("Mean transition time: $(round(mean(trans_12.tstop), digits=2))")
end

# Check data consistency
println("\n=== Data structure check ===")
println("Total rows: $(nrow(panel_data))")
println("Unique subjects: $(length(unique(panel_data.id)))")
println("obstype distribution: $(StatsBase.countmap(panel_data.obstype))")
