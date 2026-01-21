using MultistateModels
using DataFrames
using Random
using Printf

println("="^70)
println("WEIBULL PANEL TEST: Markov vs PhaseType Surrogate Comparison")
println("="^70)

Random.seed!(54321)

n_subj = 200
max_time = 5.0
panel_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# True Weibull parameters (shape=1.5, scale=2.0)
# Weibull hazard h(t) = (shape/scale) * (t/scale)^(shape-1)
true_shape = 1.5
true_scale = 2.0

# Create template data for simulation
template_rows = []
for i in 1:n_subj
    push!(template_rows, (id=i, tstart=0.0, tstop=max_time, statefrom=1, stateto=2, obstype=1))
end
template_data = DataFrame(template_rows)

# Create model with true parameters and simulate exact data
h12_sim = Hazard(@formula(0 ~ 1), "wei", 1, 2)
model_sim = multistatemodel(h12_sim; data=template_data)
# Set true parameters on log scale
set_parameters!(model_sim, (h12=[log(true_shape), log(true_scale)],))

# Simulate exact transition times
sim_data = simulate(model_sim; paths=false, data=true, nsim=1)[1,1]

# Convert to panel data
function make_panel_data(exact_data, times)
    rows = []
    for subj_id in unique(exact_data.id)
        subj_rows = filter(row -> row.id == subj_id, eachrow(exact_data))
        # Find the event time from the exact data
        event_time = Inf
        for row in subj_rows
            if row.statefrom == 1 && row.stateto == 2
                event_time = row.tstop
                break
            end
        end
        
        for j in 1:(length(times)-1)
            t_start = times[j]
            t_stop = times[j+1]
            if event_time > t_stop
                push!(rows, (id=subj_id, tstart=t_start, tstop=t_stop, statefrom=1, stateto=1, obstype=2))
            elseif event_time > t_start && event_time <= t_stop
                push!(rows, (id=subj_id, tstart=t_start, tstop=t_stop, statefrom=1, stateto=2, obstype=2))
                break
            elseif t_start == 0.0 && event_time <= t_stop
                # Event in first interval
                push!(rows, (id=subj_id, tstart=t_start, tstop=t_stop, statefrom=1, stateto=2, obstype=2))
                break
            end
        end
    end
    return DataFrame(rows)
end

data = make_panel_data(sim_data, panel_times)
n_transitions = sum(data.stateto .== 2)
println("Data: $n_subj subjects, $(nrow(data)) obs, $n_transitions transitions")
println("True parameters: shape=$(true_shape), scale=$(true_scale)")

println("\n" * "="^70)
println("Fitting with MARKOV surrogate...")
println("="^70)

h12_m = Hazard(@formula(0 ~ 1), "wei", 1, 2)
model_m = multistatemodel(h12_m; data=data, surrogate=:markov)

t_m = @elapsed fitted_m = fit(model_m; verbose=false, method=:MCEM, tol=0.01, ess_target_initial=50, max_ess=500, maxiter=100)
p_m = fitted_m.parameters.flat
est_shape_m = exp(p_m[1])
est_scale_m = exp(p_m[2])

println("Time: $(round(t_m, digits=1)) sec")
println("Estimated: shape=$(round(est_shape_m, digits=4)), scale=$(round(est_scale_m, digits=4))")
println("Error: shape=$(round(100*(est_shape_m - true_shape)/true_shape, digits=1))%, scale=$(round(100*(est_scale_m - true_scale)/true_scale, digits=1))%")

println("\n" * "="^70)
println("Fitting with PHASETYPE surrogate...")
println("="^70)

h12_pt = Hazard(@formula(0 ~ 1), "wei", 1, 2)
model_pt = multistatemodel(h12_pt; data=data, surrogate=:phasetype)

t_pt = @elapsed fitted_pt = fit(model_pt; verbose=false, method=:MCEM, tol=0.01, ess_target_initial=50, max_ess=500, maxiter=100)
p_pt = fitted_pt.parameters.flat
est_shape_pt = exp(p_pt[1])
est_scale_pt = exp(p_pt[2])

println("Time: $(round(t_pt, digits=1)) sec")
println("Estimated: shape=$(round(est_shape_pt, digits=4)), scale=$(round(est_scale_pt, digits=4))")
println("Error: shape=$(round(100*(est_shape_pt - true_shape)/true_shape, digits=1))%, scale=$(round(100*(est_scale_pt - true_scale)/true_scale, digits=1))%")

println("\n" * "="^70)
println("COMPARISON")
println("="^70)

println(rpad("Param", 10), rpad("True", 12), rpad("Markov", 12), rpad("PhaseType", 12), rpad("Diff(%)", 12))
println("-"^58)

shape_diff = 100 * abs(est_shape_m - est_shape_pt) / true_shape
scale_diff = 100 * abs(est_scale_m - est_scale_pt) / true_scale

@printf "%-10s %-12.4f %-12.4f %-12.4f %-12.1f\n" "shape" true_shape est_shape_m est_shape_pt shape_diff
@printf "%-10s %-12.4f %-12.4f %-12.4f %-12.1f\n" "scale" true_scale est_scale_m est_scale_pt scale_diff

println("-"^58)
println("Max surrogate difference: $(round(max(shape_diff, scale_diff), digits=1))%")

if max(shape_diff, scale_diff) < 15.0
    println("PASS: Surrogates agree (diff < 15%)")
else
    println("WARN: Surrogates differ by > 15%")
end
