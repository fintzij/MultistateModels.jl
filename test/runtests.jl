using Chain
using MultistateModels
using Random
using Test

# set seed
Random.seed!(52787)

# setup file for generating model objects 
# include("setup.jl") # this could be included in the individual .jl files

include("test/setup_3state_expwei.jl")
include("test/setup_3state_weiph.jl")
include("test/setup_2state_trans.jl")

@testset "runtests" begin
    include("test/test_modelgeneration.jl")
    include("test/test_hazards.jl")
    include("test/test_helpers.jl")
    include("test/test_loglik.jl")
end
