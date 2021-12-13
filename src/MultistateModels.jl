module MultistateModels

using DataFrames
using Distributions
using StatsModels
using Symbolics

# Write your package code here.
export
    @formula,
    Hazard,
    MultistateModel

include("hazards.jl")

end
