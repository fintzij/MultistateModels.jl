# Benchmark script for batched Fisher information computation
# Compares old O(n_paths) Hessian approach vs new batched approach

using MultistateModels
using Random
using BenchmarkTools
using LinearAlgebra
using ForwardDiff
using DiffResults

"""
Run the original (non-batched) Fisher information computation for MCEM.
This computes O(n_paths) Hessians per subject.
"""
function compute_fisher_original(params, model, samplepaths, ImportanceWeights)
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # set up containers for path and sampling weight
    path = Array{MultistateModels.SamplePath}(undef, 1)
    samplingweight = Vector{Float64}(undef, 1)
    
    # initialize Fisher information matrix containers
    fishinf = zeros(Float64, nparams, nparams)
    fish_i1 = zeros(Float64, nparams, nparams)
    fish_i2 = similar(fish_i1)
    
    # container for gradient and hessian
    diffres = DiffResults.HessianResult(params)

    ll = pars -> (MultistateModels.loglik_AD(pars, MultistateModels.ExactDataAD(path, samplingweight, model.hazards, model); neg=false))

    # accumulate Fisher information
    for i in 1:nsubj
        # set importance weight
        samplingweight[1] = model.SubjectWeights[i]

        # number of paths
        npaths = length(samplepaths[i])

        # for accumulating gradients and hessians
        grads = Array{Float64}(undef, nparams, length(samplepaths[i]))

        # reset matrices for accumulating Fisher info contributions
        fill!(fish_i1, 0.0)
        fill!(fish_i2, 0.0)

        # calculate gradient and hessian for paths
        for j in 1:npaths
            path[1] = samplepaths[i][j]
            diffres = ForwardDiff.hessian!(diffres, ll, params)

            # grab hessian and gradient
            grads[:,j] = DiffResults.gradient(diffres)

            # just to be safe wrt nans or infs
            if !all(isfinite, DiffResults.hessian(diffres))
                fill!(DiffResults.hessian(diffres), 0.0)
            end

            if !all(isfinite, DiffResults.gradient(diffres))
                fill!(DiffResults.gradient(diffres), 0.0)
            end

            fish_i1 .+= ImportanceWeights[i][j] * (-DiffResults.hessian(diffres) - DiffResults.gradient(diffres) * transpose(DiffResults.gradient(diffres)))
        end

        # sum of outer products of gradients
        for j in 1:npaths
            for k in 1:npaths
                fish_i2 .+= ImportanceWeights[i][j] * ImportanceWeights[i][k] * grads[:,j] * transpose(grads[:,k])
            end
        end

        fishinf += fish_i1 + fish_i2
    end

    return fishinf
end

"""
Run the batched Fisher information computation.
Uses 1 Jacobian + 1 Hessian per subject.
"""
function compute_fisher_batched(params, model, samplepaths, ImportanceWeights)
    result = MultistateModels.compute_subject_fisher_louis_batched(params, model, samplepaths, ImportanceWeights)
    return Matrix(result.fishinf)
end

"""
Setup a semi-Markov model for benchmarking.
"""
function setup_benchmark_model(nsubj, npaths_per_subj)
    Random.seed!(12345)
    
    # Simple 2-state illness-death model with exponential hazards
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    # Simulate data
    dat = DataFrame(
        id = repeat(1:nsubj, inner=2),
        tstart = repeat([0.0, 0.5], nsubj),
        tstop = repeat([0.5, 1.0], nsubj),
        statefrom = repeat([1, 1], nsubj),
        stateto = repeat([1, 2], nsubj),
        obstype = repeat([2, 1], nsubj)
    )
    
    # Create model
    model = multistatemodel(h12, h23; data=dat)
    
    # Initialize with simple parameters
    set_parameters!(model, [0.5], [0.3])
    
    # Generate sample paths (mock MCEM-style paths)
    samplepaths = Vector{Vector{MultistateModels.SamplePath}}(undef, nsubj)
    ImportanceWeights = Vector{Vector{Float64}}(undef, nsubj)
    
    for i in 1:nsubj
        # Sample some paths
        samplepaths[i] = Vector{MultistateModels.SamplePath}(undef, npaths_per_subj)
        for j in 1:npaths_per_subj
            # Create a simple sample path
            t = rand() * 0.8
            samplepaths[i][j] = MultistateModels.SamplePath(
                i, 
                (times=[0.0, t, 1.0], states=[1, 1, 2])
            )
        end
        
        # Uniform importance weights
        ImportanceWeights[i] = fill(1.0/npaths_per_subj, npaths_per_subj)
    end
    
    # Get parameters
    params = MultistateModels.flatview(model.parameters)
    
    return model, samplepaths, ImportanceWeights, copy(params)
end

"""
Run benchmarks comparing original vs batched Fisher computation.
"""
function run_fisher_benchmark()
    println("=" ^ 70)
    println("Batched Fisher Information Benchmark")
    println("=" ^ 70)
    
    # Test configurations: (n_subjects, n_paths_per_subject)
    configs = [
        (10, 10),
        (10, 50),
        (20, 20),
        (20, 50),
        (50, 20),
    ]
    
    for (nsubj, npaths) in configs
        println("\nConfiguration: $nsubj subjects × $npaths paths each ($(nsubj * npaths) total paths)")
        println("-" ^ 60)
        
        try
            model, samplepaths, weights, params = setup_benchmark_model(nsubj, npaths)
            
            # Warm up
            fish_old = compute_fisher_original(params, model, samplepaths, weights)
            fish_new = compute_fisher_batched(params, model, samplepaths, weights)
            
            # Check correctness
            max_diff = maximum(abs.(fish_old - fish_new))
            rel_diff = max_diff / max(maximum(abs.(fish_old)), 1e-10)
            
            println("  Max absolute difference: $(round(max_diff, sigdigits=4))")
            println("  Max relative difference: $(round(rel_diff, sigdigits=4))")
            
            if max_diff > 1e-6
                println("  ⚠️  WARNING: Results differ significantly!")
            else
                println("  ✓ Results match")
            end
            
            # Benchmark original
            println("\n  Benchmarking original (O(n_paths) Hessians per subject)...")
            b_old = @benchmark compute_fisher_original($params, $model, $samplepaths, $weights) samples=5 evals=1
            time_old = median(b_old).time / 1e9  # seconds
            
            # Benchmark batched
            println("  Benchmarking batched (1 Jacobian + 1 Hessian per subject)...")
            b_new = @benchmark compute_fisher_batched($params, $model, $samplepaths, $weights) samples=5 evals=1
            time_new = median(b_new).time / 1e9  # seconds
            
            speedup = time_old / time_new
            println("\n  Original time: $(round(time_old, digits=3)) s")
            println("  Batched time:  $(round(time_new, digits=3)) s")
            println("  Speedup:       $(round(speedup, digits=2))×")
            
            # Theoretical complexity
            println("\n  Hessian calls - Original: $(nsubj * npaths), Batched: $nsubj")
            println("  Jacobian calls - Original: 0, Batched: $nsubj")
            
        catch e
            println("  Error: $e")
            @show stacktrace(catch_backtrace())
        end
    end
    
    println("\n" * "=" ^ 70)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_fisher_benchmark()
end
