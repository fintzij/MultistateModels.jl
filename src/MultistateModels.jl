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
    multistatemodel

include("hazards.jl")
include("modelgeneration.jl")
include("typedefs.jl")

end
