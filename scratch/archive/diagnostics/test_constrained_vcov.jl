using MultistateModels
using DataFrames
using Test
using Random

Random.seed!(1234)

println("Testing constrained variance with phase-type panel model...")

n = 50
data = DataFrame(
    id = repeat(1:n, inner=2),
    tstart = zeros(2n),
    tstop = zeros(2n),
    statefrom = repeat([1, 1], n),
    stateto = vcat(ones(Int, n), fill(2, n)),
    obstype = repeat([2, 2], n)
)

for i in 1:n
    idx = 2*(i-1) + 1
    t1 = rand() * 2
    t2 = t1 + rand() * 2
    data[idx, :tstart] = 0.0
    data[idx, :tstop] = t1
    data[idx, :stateto] = 1
    data[idx+1, :tstart] = t1
    data[idx+1, :tstop] = t2
    data[idx+1, :statefrom] = 1
    data[idx+1, :stateto] = 2
end

h12 = Hazard(@formula(0 ~ 1), :pt, 1, 2; n_phases=2)

println("Creating phase-type model...")
model = multistatemodel(h12; data = data)
println("  phasetype_expansion: ", !isnothing(model.phasetype_expansion))

println("\nFitting phase-type model...")
fitted = fit(model; compute_ij_vcov=true, verbose=true)

println("\nResults:")
println("  vcov: ", !isnothing(fitted.vcov))
println("  ij_vcov: ", !isnothing(fitted.ij_vcov))

vcov_auto = get_vcov(fitted)
println("  get_vcov(:auto): ", isnothing(vcov_auto) ? "nothing" : "$(size(vcov_auto)) matrix")

println("\nTest completed!")
