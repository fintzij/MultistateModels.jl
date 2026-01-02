using MultistateModels
include("../MultistateModelsTests/src/MultistateModelsTests.jl")
using .MultistateModelsTests

# Run unit tests
MultistateModelsTests.runtests()
