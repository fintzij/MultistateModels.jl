# Test observe_path function
using DataFrames
using Random
using MultistateModels: SamplePath, simulate_path

include("setup_2state_trans.jl")

samplepath = MultistateModels.simulate_path(model, 1)

# Does it work when samplepath has no transitions between tstart (10.0) and tstop (20.0)?
@testset "observe_path_no_transitions" begin
    samplepath.times = [0.0, 5.5, 9.8, 21.4, 25.0, 27.8, 29.9, 31.2, 40]
    samplepath.states = [1, 2, 1, 2, 2, 1, 1, 1, 2]

    # will need to make sure msm_2state_trans.data for subject 1 has a row with tstart=10.0 and tstop=20.0
    # pass the path, model, and subj to observe_path
    # and check that the output is what we expect
    observed = MultistateModels.observe_path(samplepath, msm_2state_trans, 1)

    @test 1+1 == 2
end

# Does it work when samplepath has transitions between tstart (10.0) and tstop (20.0), AND tstart is tied with a samplepath.times value?
@testset "observe_path_yes_transitions_tstart_tie" begin
    samplepath.times = [0.0, 5.5, 9.8, 10.0, 11.7, 14.6, 21.4, 25.0, 27.8, 29.9, 31.2, 40]
    samplepath.states = [1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2]
    
end

# Does it work when samplepath has transitions between tstart (10.0) and tstop (20.0), AND tstop is tied with a samplepath.times value?
@testset "observe_path_yes_transitions_tstop_tie" begin
    samplepath.times = [0.0, 5.5, 9.8, 11.7, 14.6, 20.0, 21.4, 25.0, 27.8, 29.9, 31.2, 40]
    samplepath.states = [1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2]

end

# Does it work when samplepath has transitions between tstart (10.0) and tstop (20.0), AND tstart is tied with a samplepath.times, AND tstop is tied with a samplepath.times?
@testset "observe_path_yes_transitions_both_tie" begin
    samplepath.times = [0.0, 5.5, 9.8, 10.0, 11.7, 14.6, 20.0, 21.4, 25.0, 27.8, 29.9, 31.2, 40]
    samplepath.states = [1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2]
    
end

# Does it work when samplepath has transitions between tstart (10.0) and tstop (20.0), AND and there are no ties between tstart, tstop, and samplepath.times?
@testset "observe_path_yes_transitions_no_tie" begin
    samplepath.times = [0.0, 5.5, 9.8, 10.1, 11.7, 14.6, 20.4, 21.4, 25.0, 27.8, 29.9, 31.2, 40]
    samplepath.states = [1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2]
    
end

# Does it work when obstype changes from 3 to 1?
