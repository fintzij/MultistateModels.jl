using Test, MultistateModels, DataFrames, Random, ExponentialUtilities

Random.seed!(11111)

h12 = Hazard(@formula(0 ~ 1), :wei, 1, 2)
h23 = Hazard(@formula(0 ~ 1), :wei, 2, 3)

n_subj = 30
data = DataFrame(
    id = repeat(1:n_subj, inner=4),
    tstart = repeat([0.0, 2.0, 4.0, 6.0], n_subj),
    tstop = repeat([2.0, 4.0, 6.0, 8.0], n_subj),
    statefrom = ones(Int, n_subj * 4),
    stateto = ones(Int, n_subj * 4),
    obstype = fill(2, n_subj * 4)
)

model = multistatemodel(h12, h23; data = data, surrogate = :markov, initialize = false)
println("tmat = ", model.tmat)

# Check surrogate model construction and bounds
surrogate_model = MultistateModels.make_surrogate_model(model)
println("\nSurrogate model bounds:")
println("  lb = ", surrogate_model.bounds.lb)
println("  ub = ", surrogate_model.bounds.ub)

println("\nBefore set_crude_init:")
println("  params = ", surrogate_model.parameters.flat)

# Set crude init manually to see what it produces
MultistateModels.set_crude_init!(surrogate_model)
println("\nAfter set_crude_init:")
println("  params = ", surrogate_model.parameters.flat)

# Check what the crude matrix looks like
crude_mat = MultistateModels.calculate_crude(model)
println("\nCrude matrix:")
display(crude_mat)

# Fit surrogate and check parameters
println("\nFitting surrogate with HEURISTIC (crude rates)...")
markov_surr_crude = MultistateModels.fit_surrogate(model; type=:markov, method=:heuristic, verbose=true)
println("\nMarkov surrogate CRUDE parameters:")
println("  flat: ", markov_surr_crude.parameters.flat)

println("\nFitting surrogate with MLE...")
markov_surr = MultistateModels.fit_surrogate(model; type=:markov, method=:mle, verbose=true)

println("\nMarkov surrogate MLE parameters:")
println("  h12 params: ", markov_surr.parameters.nested.h12)
println("  h23 params: ", markov_surr.parameters.nested.h23)
println("  flat: ", markov_surr.parameters.flat)

# Check if negative values in MLE fit
if any(markov_surr.parameters.flat .< 0)
    println("\n⚠️  WARNING: Negative parameters in MLE fit!")
    println("  This indicates Ipopt is not enforcing lb=0 bounds properly")
end

MultistateModels.fit_surrogate(model; type=:markov, method=:mle, verbose=false)

import MultistateModels: draw_samplepath, build_tpm_mapping, build_hazmat_book, build_tpm_book, compute_hazmat!, compute_tmat!, build_fbmats, get_hazard_params

params_surrog = get_hazard_params(model.markovsurrogate.parameters, model.markovsurrogate.hazards)
hazards_surrog = model.markovsurrogate.hazards

books = build_tpm_mapping(model.data)
hazmat_book = build_hazmat_book(Float64, model.tmat, books[1])
tpm_book = build_tpm_book(Float64, model.tmat, books[1])
cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())

for t in eachindex(books[1])
    compute_hazmat!(hazmat_book[t], params_surrog, hazards_surrog, books[1][t], model.data)
    compute_tmat!(tpm_book[t], hazmat_book[t], books[1][t], cache)
end

fbmats = build_fbmats(model)
abs_states = findall(map(x -> all(x .== 0), eachrow(model.tmat)))

println("Absorbing states = ", abs_states)
println("\nSurrogate hazards: ")
for h in hazards_surrog
    println("  ", h.hazname, ": statefrom=", h.statefrom, ", stateto=", h.stateto)
end
println("\nSurrogate parameters: ", params_surrog)
println("\nQ matrix [1] = ")
display(hazmat_book[1])
println("\nRow sums of Q: ", sum(hazmat_book[1], dims=2))
println("\nTPM [1] = ")
display(tpm_book[1][1])

for i in 1:100
    path = draw_samplepath(1, model, tpm_book, hazmat_book, books[2], fbmats, abs_states)
    n_transitions = length(path.times) - 1
    for j in 1:n_transitions
        sf = path.states[j]
        st = path.states[j+1]
        th = model.tmat[sf, st]
        if th == 0
            println("BAD TRANSITION: path ", i, ", ", sf, " -> ", st)
            println("Full path: ", path)
            error("Found bad transition!")
        end
    end
end
println("No bad transitions found in 100 paths!")
