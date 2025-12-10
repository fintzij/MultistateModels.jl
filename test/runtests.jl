include(joinpath(@__DIR__, "..", "MultistateModelsTests", "src", "MultistateModelsTests.jl"))
using .MultistateModelsTests

MultistateModelsTests.runtests()
