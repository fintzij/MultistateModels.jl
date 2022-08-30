# Test observe_path function
using DataFrames
using Random
using MultistateModels: SamplePath, simulate_path

include("setup_2statetrans.jl")

samplepath = simulate_path(model, 1)

# Does it work when samplepath has no transitions between tstart and tstop?
@testset "observe_path_no_transitions" begin
    samplepath.times = [0.0, 5.5, 9.8, 21.4, 25.0, 27.8, 29.9, 31.2, 40]
    samplepath.states = [1, 2, 1, 2, 2, 1, 1, 1, 2]
end

# Does it work when samplepath has transitions between tstart and tstop, AND tstart is tied with a samplepath.times value?
@testset "observe_path_yes_transitions_tstart_tie" begin
    
end

# Does it work when samplepath has transitions between tstart and tstop, AND tstop is tied with a samplepath.times value?
@testset "observe_path_yes_transitions_tstop_tie" begin

end

# Does it work when samplepath has transitions between tstart and tstop, AND tstart is tied with a samplepath.times, AND tstop is tied with a samplepath.times
@testset "observe_path_yes_transitions_both_tie" begin
    
end

# Does it work when samplepath has transitions between tstart and tstop, AND and there are no ties between tstart, tstop, and samplepath.times
@testset "observe_path_yes_transitions_no_tie" begin
    
end
