module MultistateModels

using Chain
using DataFrames
using Distributions
using LinearAlgebra
using QuadGK
using StatsModels
using StatsFuns

# Write your package code here.
export
    @formula,
    Hazard,
    multistatemodel

# typedefs
include("common.jl")

# helpers
include("helpers.jl")

# hazard related functions
include("hazards.jl")

# model generation
include("modelgeneration.jl")

end
