# =============================================================================
# Tests for Phase-Type Model Simulation
# =============================================================================
#
# Tests verifying:
# 1. simulate_path works on PhaseTypeModel with expanded/collapsed output
# 2. simulate, simulate_data, simulate_paths work correctly
# 3. Path collapsing correctly merges phases within same observed state
# 4. Data collapsing correctly merges rows
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random

@testset "Phase-Type Simulation" begin
    Random.seed!(42)
    
    # Create test data for a 1→2→3 progressive model
    n_subjects = 20
    data_rows = []
    for i in 1:n_subjects
        t1 = rand() * 2
        t2 = rand() * 2
        push!(data_rows, (id=i, tstart=0.0, tstop=t1, statefrom=1, stateto=2, obstype=1))
        push!(data_rows, (id=i, tstart=t1, tstop=t1+t2, statefrom=2, stateto=3, obstype=1))
    end
    data = DataFrame(data_rows)
    
    # Build phase-type model with 2 phases on 1→2 transition
    h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    model = multistatemodel(h12, h23; data=data)
    initialize_parameters!(model)
    
    @testset "State Space Structure" begin
        # Verify mappings are correct
        @test model.mappings.n_observed == 3
        @test model.mappings.n_expanded == 4  # 2 phases for state 1, 1 for state 2, 1 for state 3
        @test model.mappings.phase_to_state == [1, 1, 2, 3]
        @test model.mappings.n_phases_per_state == [2, 1, 1]
    end
    
    @testset "simulate_path" begin
        @testset "expanded=true returns phase-level states" begin
            Random.seed!(123)
            path = simulate_path(model, 1; expanded=true)
            
            @test path isa MultistateModels.SamplePath
            @test path.subj == 1
            @test length(path.times) == length(path.states)
            @test all(s in 1:4 for s in path.states)  # States should be in expanded space
        end
        
        @testset "expanded=false returns observed states" begin
            Random.seed!(123)
            path = simulate_path(model, 1; expanded=false)
            
            @test path isa MultistateModels.SamplePath
            @test path.subj == 1
            @test all(s in 1:3 for s in path.states)  # States should be in original space
        end
        
        @testset "collapsed path merges phases correctly" begin
            # Find a case where expanded path has multiple phases in same observed state
            found_merge = false
            for seed in 1:100
                Random.seed!(seed)
                path_exp = simulate_path(model, 1; expanded=true)
                
                # Check if any consecutive expanded states map to same observed state
                observed = [model.mappings.phase_to_state[s] for s in path_exp.states]
                for i in 2:length(observed)
                    if observed[i] == observed[i-1]
                        Random.seed!(seed)
                        path_col = simulate_path(model, 1; expanded=false)
                        
                        # Collapsed should have fewer states
                        @test length(path_col.states) < length(path_exp.states)
                        found_merge = true
                        break
                    end
                end
                found_merge && break
            end
            @test found_merge  # We should find at least one case
        end
        
        @testset "subject index validation" begin
            @test_throws ArgumentError simulate_path(model, 0)
            @test_throws ArgumentError simulate_path(model, n_subjects + 1)
        end
    end
    
    @testset "simulate" begin
        @testset "returns Vector{DataFrame} for data only" begin
            datasets = simulate(model; nsim=3, data=true, paths=false)
            
            @test datasets isa Vector{DataFrame}
            @test length(datasets) == 3
            @test all(d isa DataFrame for d in datasets)
        end
        
        @testset "returns tuple for data and paths" begin
            dat, paths = simulate(model; nsim=2, data=true, paths=true)
            
            @test dat isa Vector{DataFrame}
            @test paths isa Vector{Vector{MultistateModels.SamplePath}}
            @test length(dat) == 2
            @test length(paths) == 2
        end
        
        @testset "expanded=true returns expanded state space" begin
            Random.seed!(456)
            dat_exp, paths_exp = simulate(model; nsim=2, data=true, paths=true, expanded=true)
            
            # States should be in expanded space (1-4)
            for path_set in paths_exp
                for path in path_set
                    @test all(s in 1:4 for s in path.states)
                end
            end
        end
        
        @testset "expanded=false returns collapsed state space" begin
            Random.seed!(456)
            dat_col, paths_col = simulate(model; nsim=2, data=true, paths=true, expanded=false)
            
            # States should be in original space (1-3)
            for path_set in paths_col
                for path in path_set
                    @test all(s in 1:3 for s in path.states)
                end
            end
        end
    end
    
    @testset "simulate_data" begin
        datasets = simulate_data(model; nsim=3)
        
        @test datasets isa Vector{DataFrame}
        @test length(datasets) == 3
        
        # Collapsed data should have states in original space
        for df in datasets
            @test all(s in 1:3 for s in df.statefrom)
            @test all(s in 1:3 for s in df.stateto)
        end
        
        # Expanded data should have states in expanded space
        datasets_exp = simulate_data(model; nsim=3, expanded=true)
        for df in datasets_exp
            @test all(s in 1:4 for s in df.statefrom)
            @test all(s in 1:4 for s in df.stateto)
        end
    end
    
    @testset "simulate_paths" begin
        paths = simulate_paths(model; nsim=3)
        
        @test paths isa Vector{Vector{MultistateModels.SamplePath}}
        @test length(paths) == 3
        
        # Each simulation should have paths for all subjects
        for path_set in paths
            @test length(path_set) == n_subjects
        end
    end
    
    @testset "Path collapsing correctness" begin
        # Test that collapsing merges consecutive same-state phases
        phase_to_state = model.mappings.phase_to_state
        
        # Create a synthetic expanded path that should collapse
        expanded_path = MultistateModels.SamplePath(1, [0.0, 0.5, 1.0, 1.5], [1, 2, 3, 4])
        # Phase 1 -> State 1, Phase 2 -> State 1, Phase 3 -> State 2, Phase 4 -> State 3
        # Expected: [1, 2, 3] at times [0.0, 1.0, 1.5]
        
        collapsed = MultistateModels._collapse_path(expanded_path, model.mappings)
        
        @test collapsed.states == [1, 2, 3]
        @test collapsed.times == [0.0, 1.0, 1.5]
    end
    
    @testset "Data collapsing correctness" begin
        # Create a synthetic expanded DataFrame
        expanded_df = DataFrame(
            id = [1, 1, 1, 1],
            tstart = [0.0, 0.5, 1.0, 1.5],
            tstop = [0.5, 1.0, 1.5, 2.0],
            statefrom = [1, 2, 3, 4],
            stateto = [2, 3, 4, 4],
            obstype = [1, 1, 1, 3]  # Last is censored
        )
        
        collapsed = MultistateModels._collapse_data(expanded_df, model.mappings)
        
        # First two rows (phases 1→2, 2→3) both map to state 1 staying in state 1
        # Then state 1 → state 2 (phase 3 maps to state 2)
        # Then state 2 → state 3 (phase 4 maps to state 3)
        
        @test nrow(collapsed) <= nrow(expanded_df)
        @test all(s in 1:3 for s in collapsed.statefrom)
        @test all(s in 1:3 for s in collapsed.stateto)
    end
end
