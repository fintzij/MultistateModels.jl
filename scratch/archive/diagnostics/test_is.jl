using Pkg
Pkg.activate("MultistateModelsTests")
Pkg.develop(path=".")

using MultistateModels
using DataFrames
using Random

import MultistateModels: Hazard, multistatemodel, set_parameters!,
    SamplePath, @formula, _fit_phasetype_surrogate, _fit_markov_surrogate,
    compute_phasetype_marginal_loglik, compute_markov_marginal_loglik

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

panel_data = DataFrame(
    id = [1, 1, 2, 2],
    tstart = [0.0, 2.0, 0.0, 2.0],
    tstop = [2.0, 4.0, 2.0, 4.0],
    statefrom = [1, 2, 1, 1],
    stateto = [2, 3, 1, 2],
    obstype = [2, 2, 2, 2]
)

model = multistatemodel(h12, h23; data=panel_data, initialize=false, surrogate_n_phases=:three)
set_parameters!(model, (h12 = [1.3, 0.15], h23 = [1.1, 0.12]))

pt_surrog = _fit_phasetype_surrogate(model; n_phases=3, verbose=false)
markov_surrog = _fit_markov_surrogate(model; verbose=false)

emat_ph = MultistateModels.build_phasetype_emat_expanded(model, pt_surrog)

nc_phasetype = compute_phasetype_marginal_loglik(model, pt_surrog, emat_ph)
nc_markov = compute_markov_marginal_loglik(model, markov_surrog)

println("NormConstantProposal (PhaseType): ", round(nc_phasetype, digits=4))
println("NormConstantProposal (Markov): ", round(nc_markov, digits=4))
