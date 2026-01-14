using MultistateModels, DataFrames, Random, Statistics
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate, get_parameters, PhaseTypeProposal, MarkovProposal

println("="^60)
println("TVC Estimation Investigation - Multiple replications")
println("="^60)

true_shape, true_scale, true_beta = 1.3, 0.15, 0.5
n_reps = 5

# Store results
markov_results = [(0.0, 0.0, 0.0) for _ in 1:n_reps]
pt_results = [(0.0, 0.0, 0.0) for _ in 1:n_reps]

for rep in 1:n_reps
    Random.seed!(12345 + rep)
    
    n_subj = 1000
    obs_times = [3.0, 6.0, 9.0, 12.0]
    change_time = 4.0

    rows = []
    for subj in 1:n_subj
        trt = rand() < 0.5 ? [0.0, 1.0] : [0.0, 0.0]
        all_times = sort(unique([0.0; obs_times; change_time]))
        for i in 1:(length(all_times)-1)
            x_val = all_times[i] < change_time ? trt[1] : trt[2]
            push!(rows, (id=subj, tstart=all_times[i], tstop=all_times[i+1], statefrom=1, stateto=1, obstype=2, x=x_val))
        end
    end
    template_data = DataFrame(rows)

    h12 = Hazard(@formula(0 ~ x), "wei", 1, 2)
    model_sim = multistatemodel(h12; data=template_data, surrogate=:markov)
    set_parameters!(model_sim, (h12 = [true_shape, true_scale, true_beta],))

    sim_panel = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false, obstype_by_transition=Dict(1 => 2))
    panel_data = sim_panel[1, 1]
    
    println("\nRep $rep:")
    
    # Markov MCEM
    model_markov = multistatemodel(Hazard(@formula(0 ~ x), "wei", 1, 2); data=panel_data, surrogate=:markov)
    fitted_markov = fit(model_markov; proposal=MarkovProposal(), verbose=false, maxiter=50, tol=0.02, ess_target_initial=50, max_ess=300, compute_vcov=false)
    pm = get_parameters(fitted_markov; scale=:natural)
    markov_results[rep] = (pm.h12[1], pm.h12[2], pm.h12[3])
    println("  Markov: shape=$(round(pm.h12[1],digits=3)), scale=$(round(pm.h12[2],digits=3)), beta=$(round(pm.h12[3],digits=3))")
    
    # PhaseType MCEM 
    model_pt = multistatemodel(Hazard(@formula(0 ~ x), "wei", 1, 2); data=panel_data, surrogate=:markov)
    fitted_pt = fit(model_pt; proposal=PhaseTypeProposal(n_phases=3), verbose=false, maxiter=50, tol=0.02, ess_target_initial=50, max_ess=300, compute_vcov=false)
    pp = get_parameters(fitted_pt; scale=:natural)
    pt_results[rep] = (pp.h12[1], pp.h12[2], pp.h12[3])
    println("  PT:     shape=$(round(pp.h12[1],digits=3)), scale=$(round(pp.h12[2],digits=3)), beta=$(round(pp.h12[3],digits=3))")
end

println("\n" * "="^60)
println("Summary (true: shape=$true_shape, scale=$true_scale, beta=$true_beta)")
println("="^60)

markov_shape = mean([r[1] for r in markov_results])
markov_scale = mean([r[2] for r in markov_results])
markov_beta = mean([r[3] for r in markov_results])
pt_shape = mean([r[1] for r in pt_results])
pt_scale = mean([r[2] for r in pt_results])
pt_beta = mean([r[3] for r in pt_results])

println("Markov avg: shape=$(round(markov_shape,digits=3)) ($(round(100*(markov_shape/true_shape-1),digits=1))%), scale=$(round(markov_scale,digits=3)) ($(round(100*(markov_scale/true_scale-1),digits=1))%), beta=$(round(markov_beta,digits=3)) ($(round(100*(markov_beta/true_beta-1),digits=1))%)")
println("PT avg:     shape=$(round(pt_shape,digits=3)) ($(round(100*(pt_shape/true_shape-1),digits=1))%), scale=$(round(pt_scale,digits=3)) ($(round(100*(pt_scale/true_scale-1),digits=1))%), beta=$(round(pt_beta,digits=3)) ($(round(100*(pt_beta/true_beta-1),digits=1))%)")
