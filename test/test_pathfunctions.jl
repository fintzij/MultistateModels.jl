# Test observe_path function
using DataFrames
using Random
using MultistateModels: SamplePath, simulate_path

include("setup_2statetrans_dat.jl")

# Does it work when samplepath has no transitions between tstart (0.0) and tstop (10.0)?
@testset "observe_path_no_transitions" begin

    path = 
        MultistateModels.SamplePath(
            1, 
            [0.0, 10.0, 11.7, 19.4, 25, 29.4], 
            [1, 1, 2, 1, 2, 1])
    
    expected = 
        DataFrame(id = fill(1, 3),
                  tstart = [0, 10.0, 20.0],
                  tstop = [10.0, 20.0, 30.0],
                  statefrom = [1, 1, 1],
                  stateto = [1, 1, 2],
                  obstype = [1, 2, 3])

    # will need to make sure msm_2state_trans.data for subject 1 has a row with tstart=10.0 and tstop=20.0
    # pass the path, model, and subj to observe_path
    # and check that the output is what we expect
    observed = MultistateModels.observe_path(path, msm_2state_trans, 1)

    @test observed == expected
end

# Does it work when samplepath has transitions between tstart (10.0) and tstop (20.0), AND tstart is tied with a samplepath.times value?
@testset "observe_path_yes_transitions_tstart_tie" begin

    path = 
        MultistateModels.SamplePath(
            1,
            [0.0, 5.5, 9.8, 10.0, 11.7, 14.6, 21.4, 25.0, 27.8, 29.9, 31.2, 40.0],
            [1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2])

    expected = DataFrame(id=fill(1,5),
                         tstart = [0, 5.5, 9.8, 10.0, 20.0],
                         tstop = [5.5, 9.8, 10.0, 20.0, 30.0],
                         statefrom = [1, 2, 1, 2, 1],
                         stateto = [2, 1, 2, 1, 2],
                         obstype = [1, 1, 1, 2, 3])

    observed = MultistateModels.observe_path(path, msm_2state_trans, 1)
    
    @test observed == expected
end

# Does it work when samplepath has transitions between tstart (10.0) and tstop (20.0), AND tstop is tied with a samplepath.times value?
@testset "observe_path_yes_transitions_tstop_tie" begin

    path = MultistateModels.SamplePath(
        1, 
        [0.0, 5.5, 9.8, 11.7, 14.6, 20.0, 21.4, 25.0, 27.8, 29.9, 31.2, 40.0],
        [1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2])

    expected = DataFrame(id = fill(1, 5),
                         tstart = [0, 5.5, 9.8, 10.0, 20.0],
                         tstop = [5.5, 9.8, 10.0, 20.0, 30.0],
                         statefrom = [1, 2, 1, 1, 1],
                         stateto = [2, 1, 1, 1, 2],
                         obstype = [1, 1, 1, 2, 3])

    observed = MultistateModels.observe_path(path, msm_2state_trans, 1)

    @test observed == expected
end

# Does it work when samplepath has transitions between tstart (10.0) and tstop (20.0), AND tstart is tied with a samplepath.times, AND tstop is tied with a samplepath.times?
@testset "observe_path_yes_transitions_both_tie" begin

    path = MultistateModels.SamplePath(
        1, 
        [0.0, 5.5, 9.8, 10.0, 11.7, 14.6, 20.0, 21.4, 25.0, 27.8, 29.9, 31.2, 40.0],
        [1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2])

    expected = DataFrame(id = fill(1, 5),
                         tstart = [0, 5.5, 9.8, 10.0, 20.0],
                         tstop = [5.5, 9.8, 10.0, 20.0, 30.0],
                         statefrom = [1, 2, 1, 2, 1],
                         stateto = [2, 1, 2, 1, 2],
                         obstype = [1, 1, 1, 2, 3])

    observed = MultistateModels.observe_path(path, msm_2state_trans, 1)

    @test observed == expected
end

# Does it work when samplepath has transitions between tstart (10.0) and tstop (20.0), AND and there are no ties between tstart, tstop, and samplepath.times?
@testset "observe_path_yes_transitions_no_tie" begin

    path = MultistateModels.SamplePath(
        1, 
        [0.0, 5.5, 9.8, 10.1, 11.7, 14.6, 20.4, 21.4, 25.0, 27.8, 29.9, 31.2, 40.0],
        [1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2])

    expected = DataFrame(id = fill(1, 5),
                         tstart = [0, 5.5, 9.8, 10.0, 20.0],
                         tstop = [5.5, 9.8, 10.0, 20.0, 30.0],
                         statefrom = [1, 2, 1, 1, 1],
                         stateto = [2, 1, 1, 1, 2],
                         obstype = [1, 1, 1, 2, 3])

    observed = MultistateModels.observe_path(path, msm_2state_trans, 1)

    @test observed == expected
end

# Does it work when obstype changes from 3 to 1?
@testset "observe_path_change_obs_type_3_to_1" begin

    msm_2state_trans.data.obstype[msm_2state_trans.data.obstype.==3] .= 1

    path = MultistateModels.SamplePath(
        1, 
        [0.0, 5.5, 9.8, 10.1, 11.7, 14.6, 20.4, 21.4, 25.0, 27.8, 29.9, 31.2, 40.0],
        [1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2])

    expected = DataFrame(id = fill(1, 10),
                         tstart = [0, 5.5, 9.8, 10.0, 20.0, 20.4, 21.4, 25.0, 27.8, 29.9],
                         tstop = [5.5, 9.8, 10.0, 20.0, 20.4, 21.4, 25.0, 27.8, 29.9, 30.0],
                         statefrom = [1, 2, 1, 1, 1, 1, 2, 2, 1, 1],
                         stateto = [2, 1, 1, 1, 1, 2, 2, 1, 1, 1],
                         obstype = [1, 1, 1, 2, 1, 1, 1, 1, 1, 1])

    observed = MultistateModels.observe_path(path, msm_2state_trans, 1)

    @test observed == expected
end

@testset "test_reduce_jumpchain" begin
    p1 = MultistateModels.SamplePath(1, [0.0, 1.0], [1, 1])
    p2 = MultistateModels.SamplePath(1, [0.0, 1.0], [1, 2])
    p3 = MultistateModels.SamplePath(1, [0.0, 0.5, 1.0], [1, 1, 1])
    p4 = MultistateModels.SamplePath(1, [0.0, 0.5, 1.0], [1, 2, 2])
    p5 = MultistateModels.SamplePath(1, [0.0, 0.3, 0.5, 0.7, 1.0], [1, 1, 1, 1, 1])
    p6 = MultistateModels.SamplePath(1, [0.0, 0.3, 0.5, 0.7, 1.0], [1, 1, 2, 1, 2])

    # reduce
    r1 = MultistateModels.reduce_jumpchain(p1)
    r2 = MultistateModels.reduce_jumpchain(p2)
    r3 = MultistateModels.reduce_jumpchain(p3)
    r4 = MultistateModels.reduce_jumpchain(p4)
    r5 = MultistateModels.reduce_jumpchain(p5)
    r6 = MultistateModels.reduce_jumpchain(p6)

    # test whether reduced jump chains are correct
    @test (r1.times == [0.0, 1.0]) & (r1.states == [1, 1]) 
    @test (r2.times == [0.0, 1.0]) & (r2.states == [1, 2]) 
    @test (r3.times == [0.0, 1.0]) & (r3.states == [1, 1])
    @test (r4.times == [0.0, 0.5, 1.0]) & (r4.states == [1, 2, 2])
    @test (r5.times == [0.0, 1.0]) & (r5.states == [1, 1])
    @test (r6.times == [0.0, 0.5, 0.7, 1.0]) & (r6.states == [1, 2, 1, 2])
end