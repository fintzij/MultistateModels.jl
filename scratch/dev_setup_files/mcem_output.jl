# marginal loglikelihood
plot(model_fitted.ConvergenceRecords.mll_trace, title="Marginal Logikelihood", label=nothing, linewidth=3)
    xlabel!("Iteration")
    ylabel!("Marginal loglikelihood")

# ess per subject and per iteration
plot(model_fitted.ConvergenceRecords.ess_trace', title="ESS per subject",legend = :outertopright, linewidth=3)
    xlabel!("Iteration")
    ylabel!("ESS")

# trace of parameters
haznames = map(x -> String(model_fitted.hazards[x].hazname), collect(1:length(model_fitted.hazards)))
plot(model_fitted.ConvergenceRecords.parameters_trace', title="Trace of the parameters", linewidth=3) # label=permutedims(haznames),legend = :outertopright)
    xlabel!("Iteration")
    ylabel!("Parameters")