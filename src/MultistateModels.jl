module MultistateModels

using Chain
using DataFrames
using Distributions
using LinearAlgebra
using StatsModels
using StatsFuns

# Write your package code here.
export
    @formula,
    Hazard,
    MultistateModel

include("hazards.jl")
include("modelgeneration.jl")

end
