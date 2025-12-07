
using Test
using MultistateModels
using Distributions
using DataFrames
using Random

# Helper to run all spline tests
function run_all_spline_tests()
    @testset "Spline Tests" begin
        run_spline_nocov_exact()
        run_spline_nocov_panel()
        run_spline_tfc_exact()
        run_spline_tfc_panel()
        run_spline_tvc_exact()
        run_spline_tvc_panel()
        run_spline_monotonic_inc()
        run_spline_monotonic_dec()
        run_spline_aft_exact()
        run_spline_aft_panel()
    end
end

# Test 41: Spline, No Covariates, Exact
function run_spline_nocov_exact()
    @testset "Test 41: Spline, No Covariates, Exact" begin
        println("Running Test 41: Spline, No Covariates, Exact...")
        
        # 1. Setup data - simulate from Weibull(1.5, 0.8)
        Random.seed!(41)
        nsubj = 1000
        
        # Weibull(shape, scale_param) where scale_param = 1/lambda^(1/k)
        # h(t) = 0.8 * 1.5 * t^0.5
        shape = 1.5
        scale = 0.8
        dist = Weibull(shape, 1/scale^(1/shape)) 
        
        dat = DataFrame(id = 1:nsubj, tstart = 0.0, tstop = 0.0, statefrom = 1, stateto = 2, obstype = 1)
        
        for i in 1:nsubj
            t = rand(dist)
            c = 2.0
            if t < c
                dat.tstop[i] = t
                dat.stateto[i] = 2
            else
                dat.tstop[i] = c
                dat.stateto[i] = 1 
            end
        end
        
        # 2. Setup Spline Model
        # Hazard: 1 -> 2, "sp" for spline
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, 
                     knots=[0.5, 1.0, 1.5], 
                     boundaryknots=[0.0, 2.5]) 
        
        model = multistatemodel(h12; data=dat)
        
        # 3. Fit model
        fitted = fit(model; verbose=true)
        
        # 4. Check convergence
        @test fitted.optim_ret.converged
        
        println("Test 41 passed!")
    end
end

# Test 42: Spline, No Covariates, Panel
function run_spline_nocov_panel()
    @testset "Test 42: Spline, No Covariates, Panel" begin
        println("Running Test 42: Spline, No Covariates, Panel...")
        
        # 1. Setup data - simulate from Weibull(1.5, 0.8)
        Random.seed!(42)
        nsubj = 500
        
        shape = 1.5
        scale = 0.8
        dist = Weibull(shape, 1/scale^(1/shape)) 
        
        # Panel data: observe at t=0, 1, 2
        dat = DataFrame()
        
        for i in 1:nsubj
            t_event = rand(dist)
            
            # Observation times
            times = [0.0, 1.0, 2.0]
            
            # Determine state at each time
            states = Int[]
            push!(states, 1) # Start at 1
            
            for t in times[2:end]
                if t_event <= t
                    push!(states, 2)
                else
                    push!(states, 1)
                end
            end
            
            # Create rows
            for j in 1:(length(times)-1)
                push!(dat, (id=i, tstart=times[j], tstop=times[j+1], statefrom=states[j], stateto=states[j+1], obstype=2))
            end
        end
        
        # 2. Setup Spline Model
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, 
                     knots=[0.5, 1.0, 1.5], 
                     boundaryknots=[0.0, 2.5]) 
        
        model = multistatemodel(h12; data=dat)
        
        # 3. Fit model
        fitted = fit(model; verbose=true)
        
        # 4. Check convergence
        @test fitted.optim_ret.converged
        
        println("Test 42 passed!")
    end
end

# Test 43: Spline, TFC, Exact
function run_spline_tfc_exact()
    @testset "Test 43: Spline, TFC, Exact" begin
        println("Running Test 43: Spline, TFC, Exact...")
        
        # 1. Setup data - simulate from Weibull PH
        Random.seed!(43)
        nsubj = 1000
        
        # h(t|x) = h0(t) * exp(beta * x)
        # h0(t) = Weibull(1.5, 0.8)
        shape = 1.5
        scale = 0.8
        beta = 0.5
        
        dat = DataFrame(id = 1:nsubj, tstart = 0.0, tstop = 0.0, statefrom = 1, stateto = 2, obstype = 1, x = randn(nsubj))
        
        for i in 1:nsubj
            # Effective scale: scale * exp(beta * x)
            # S(t) = exp(-scale * t^shape * exp(beta * x))
            #      = exp(-(scale * exp(beta * x)) * t^shape)
            # So effective scale parameter for Weibull distribution is (scale * exp(beta * x))
            # And Julia's scale_param is 1 / (scale * exp(beta * x))^(1/shape)
            
            eff_scale = scale * exp(beta * dat.x[i])
            dist = Weibull(shape, 1/eff_scale^(1/shape))
            
            t = rand(dist)
            c = 2.0
            if t < c
                dat.tstop[i] = t
                dat.stateto[i] = 2
            else
                dat.tstop[i] = c
                dat.stateto[i] = 1 
            end
        end
        
        # 2. Setup Spline Model
        # Hazard: 1 -> 2, "sp" for spline
        h12 = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; 
                     degree=3, 
                     knots=[0.5, 1.0, 1.5], 
                     boundaryknots=[0.0, 2.5]) 
        
        model = multistatemodel(h12; data=dat)
        
        # 3. Fit model
        fitted = fit(model; verbose=true)
        
        # 4. Check convergence
        @test fitted.optim_ret.converged
        
        println("Test 43 passed!")
    end
end

# Test 44: Spline, TFC, Panel
function run_spline_tfc_panel()
    @testset "Test 44: Spline, TFC, Panel" begin
        println("Running Test 44: Spline, TFC, Panel...")
        
        # 1. Setup data - simulate from Weibull PH
        Random.seed!(44)
        nsubj = 500
        
        shape = 1.5
        scale = 0.8
        beta = 0.5
        
        dat = DataFrame()
        
        for i in 1:nsubj
            x = randn()
            eff_scale = scale * exp(beta * x)
            dist = Weibull(shape, 1/eff_scale^(1/shape))
            
            t_event = rand(dist)
            
            # Observation times
            times = [0.0, 1.0, 2.0]
            
            # Determine state at each time
            states = Int[]
            push!(states, 1) # Start at 1
            
            for t in times[2:end]
                if t_event <= t
                    push!(states, 2)
                else
                    push!(states, 1)
                end
            end
            
            # Create rows
            for j in 1:(length(times)-1)
                push!(dat, (id=i, tstart=times[j], tstop=times[j+1], statefrom=states[j], stateto=states[j+1], obstype=2, x=x))
            end
        end
        
        # 2. Setup Spline Model
        h12 = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; 
                     degree=3, 
                     knots=[0.5, 1.0, 1.5], 
                     boundaryknots=[0.0, 2.5]) 
        
        model = multistatemodel(h12; data=dat)
        
        # 3. Fit model
        fitted = fit(model; verbose=true)
        
        # 4. Check convergence
        @test fitted.optim_ret.converged
        
        println("Test 44 passed!")
    end
end

# Test 45: Spline, TVC, Exact
function run_spline_tvc_exact()
    @testset "Test 45: Spline, TVC, Exact" begin
        println("Running Test 45: Spline, TVC, Exact...")
        
        # 1. Setup data - simulate from Weibull PH with TVC
        # For simplicity, we'll use a step function TVC.
        # x(t) = 0 if t < 1, 1 if t >= 1
        # This splits the interval into two parts.
        
        Random.seed!(45)
        nsubj = 1000
        
        shape = 1.5
        scale = 0.8
        beta = 0.5
        
        dat = DataFrame()
        
        for i in 1:nsubj
            # Simulate event time.
            # Cumulative hazard H(t) = \int_0^t h0(u) exp(beta * x(u)) du
            # x(u) = 0 for u < 1, 1 for u >= 1
            # H(t) = scale * t^shape  if t < 1
            # H(t) = scale * 1^shape + scale * exp(beta) * (t^shape - 1^shape) if t >= 1
            
            # Inverse transform sampling
            u = rand()
            target_H = -log(u)
            
            H_1 = scale * 1.0^shape
            
            if target_H < H_1
                # t < 1
                # target_H = scale * t^shape => t = (target_H / scale)^(1/shape)
                t = (target_H / scale)^(1/shape)
            else
                # t >= 1
                # target_H = H_1 + scale * exp(beta) * (t^shape - 1)
                # (target_H - H_1) / (scale * exp(beta)) = t^shape - 1
                # t^shape = 1 + (target_H - H_1) / (scale * exp(beta))
                t = (1 + (target_H - H_1) / (scale * exp(beta)))^(1/shape)
            end
            
            c = 2.0
            if t < c
                # Event observed
                # We need to split the record at t=1 if t > 1
                if t > 1.0
                    push!(dat, (id=i, tstart=0.0, tstop=1.0, statefrom=1, stateto=1, obstype=1, x=0.0))
                    push!(dat, (id=i, tstart=1.0, tstop=t, statefrom=1, stateto=2, obstype=1, x=1.0))
                else
                    push!(dat, (id=i, tstart=0.0, tstop=t, statefrom=1, stateto=2, obstype=1, x=0.0))
                end
            else
                # Censored at 2.0
                # Split at 1.0
                push!(dat, (id=i, tstart=0.0, tstop=1.0, statefrom=1, stateto=1, obstype=1, x=0.0))
                push!(dat, (id=i, tstart=1.0, tstop=2.0, statefrom=1, stateto=1, obstype=1, x=1.0))
            end
        end
        
        # 2. Setup Spline Model
        h12 = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; 
                     degree=3, 
                     knots=[0.5, 1.0, 1.5], 
                     boundaryknots=[0.0, 2.5]) 
        
        model = multistatemodel(h12; data=dat)
        
        # 3. Fit model
        fitted = fit(model; verbose=true)
        
        # 4. Check convergence
        @test fitted.optim_ret.converged
        
        println("Test 45 passed!")
    run_spline_tvc_panel()
        run_spline_monotonic_inc()
    end
end

# Test 46: Spline, TVC, Panel
function run_spline_tvc_panel()
    @testset "Test 46: Spline, TVC, Panel" begin
        println("Running Test 46: Spline, TVC, Panel...")
        
        # 1. Setup data - simulate from Weibull PH with TVC
        Random.seed!(46)
        nsubj = 500
        
        shape = 1.5
        scale = 0.8
        beta = 0.5
        
        dat = DataFrame()
        
        for i in 1:nsubj
            # Simulate event time (same as Test 45)
            u = rand()
            target_H = -log(u)
            
            H_1 = scale * 1.0^shape
            
            if target_H < H_1
                t_event = (target_H / scale)^(1/shape)
            else
                t_event = (1 + (target_H - H_1) / (scale * exp(beta)))^(1/shape)
            end
            
            # Observation times
            times = [0.0, 1.0, 2.0]
            
            # Determine state at each time
            states = Int[]
            push!(states, 1) # Start at 1
            
            for t in times[2:end]
                if t_event <= t
                    push!(states, 2)
                else
                    push!(states, 1)
                end
            end
            
            # Create rows
            for j in 1:(length(times)-1)
                t_start = times[j]
                t_stop = times[j+1]
                
                # Determine x value for this interval
                val_x = (t_start >= 1.0) ? 1.0 : 0.0
                
                push!(dat, (id=i, tstart=t_start, tstop=t_stop, statefrom=states[j], stateto=states[j+1], obstype=2, x=val_x))
            end
        end
        
        # 2. Setup Spline Model
        h12 = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; 
                     degree=3, 
                     knots=[0.5, 1.0, 1.5], 
                     boundaryknots=[0.0, 2.5]) 
        
        model = multistatemodel(h12; data=dat)
        
        # 3. Fit model
        fitted = fit(model; verbose=true)
        
        # 4. Check convergence
        @test fitted.optim_ret.converged
        
        println("Test 46 passed!")
    end
end

# Test 47: Spline, Monotonic Increasing
function run_spline_monotonic_inc()
    @testset "Test 47: Spline, Monotonic Increasing" begin
        println("Running Test 47: Spline, Monotonic Increasing...")
        
        # 1. Setup data - simulate from Weibull(1.5, 0.8) -> Increasing Hazard
        Random.seed!(47)
        nsubj = 1000
        
        shape = 1.5
        scale = 0.8
        dist = Weibull(shape, 1/scale^(1/shape)) 
        
        dat = DataFrame(id = 1:nsubj, tstart = 0.0, tstop = 0.0, statefrom = 1, stateto = 2, obstype = 1)
        
        for i in 1:nsubj
            t = rand(dist)
            c = 2.0
            if t < c
                dat.tstop[i] = t
                dat.stateto[i] = 2
            else
                dat.tstop[i] = c
                dat.stateto[i] = 1 
            end
        end
        
        # 2. Setup Spline Model with monotone=1
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, 
                     knots=[0.5, 1.0, 1.5], 
                     boundaryknots=[0.0, 2.5],
                     monotone=1) 
        
        model = multistatemodel(h12; data=dat)
        
        # 3. Fit model
        fitted = fit(model; verbose=true)
        
        # 4. Check convergence
        @test fitted.optim_ret.converged
        
        println("Test 47 passed!")
    end
end

# Test 48: Spline, Monotonic Decreasing
function run_spline_monotonic_dec()
    @testset "Test 48: Spline, Monotonic Decreasing" begin
        println("Running Test 48: Spline, Monotonic Decreasing...")
        
        # 1. Setup data - simulate from Weibull(0.5, 0.8) -> Decreasing Hazard
        Random.seed!(48)
        nsubj = 1000
        
        shape = 0.5
        scale = 0.8
        dist = Weibull(shape, 1/scale^(1/shape)) 
        
        dat = DataFrame(id = 1:nsubj, tstart = 0.0, tstop = 0.0, statefrom = 1, stateto = 2, obstype = 1)
        
        for i in 1:nsubj
            t = rand(dist)
            c = 2.0
            if t < c
                dat.tstop[i] = t
                dat.stateto[i] = 2
            else
                dat.tstop[i] = c
                dat.stateto[i] = 1 
            end
        end
        
        # 2. Setup Spline Model with monotone=-1
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, 
                     knots=[0.5, 1.0, 1.5], 
                     boundaryknots=[0.0, 2.5],
                     monotone=-1) 
        
        model = multistatemodel(h12; data=dat)
        
        # 3. Fit model
        fitted = fit(model; verbose=true)
        
        # 4. Check convergence
        @test fitted.optim_ret.converged
        
        println("Test 48 passed!")
    end
end

# Test 49: Spline, AFT, Exact
function run_spline_aft_exact()
    @testset "Test 49: Spline, AFT, Exact" begin
        println("Running Test 49: Spline, AFT, Exact...")
        
        # 1. Setup data - simulate from Weibull AFT
        # T = exp(mu + beta*x + sigma*W)
        # h(t|x) = h0(t * exp(-beta*x)) * exp(-beta*x)
        # This is equivalent to Weibull PH with different parameterization.
        # But we want to test the AFT implementation in splines.
        
        Random.seed!(49)
        nsubj = 1000
        
        # Parameters
        mu = 0.5
        sigma = 0.8 # scale parameter in AFT (1/shape in PH)
        beta = 0.5
        
        dat = DataFrame(id = 1:nsubj, tstart = 0.0, tstop = 0.0, statefrom = 1, stateto = 2, obstype = 1, x = randn(nsubj))
        
        for i in 1:nsubj
            # Simulate T
            # W ~ Gumbel? No, ExtremeValue(0,1) (Gumbel min)
            # Julia's Gumbel is Max.
            # Log-Weibull is Gumbel (min).
            # Let's use the PH relation.
            # Weibull PH: h(t) = lambda * k * t^(k-1)
            # AFT: T ~ Weibull(shape=1/sigma, scale=exp(mu + beta*x))
            
            shape_ph = 1/sigma
            scale_ph = exp(mu + beta * dat.x[i])
            
            # Julia Weibull(shape, scale)
            dist = Weibull(shape_ph, scale_ph)
            
            t = rand(dist)
            c = 2.0
            if t < c
                dat.tstop[i] = t
                dat.stateto[i] = 2
            else
                dat.tstop[i] = c
                dat.stateto[i] = 1 
            end
        end
        
        # 2. Setup Spline Model with AFT
        h12 = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; 
                     degree=3, 
                     knots=[0.5, 1.0, 1.5], 
                     boundaryknots=[0.0, 2.5],
                     linpred_effect=:aft) 
        
        model = multistatemodel(h12; data=dat)
        
        # 3. Fit model
        fitted = fit(model; verbose=true)
        
        # 4. Check convergence
        @test fitted.optim_ret.converged
        
        println("Test 49 passed!")
    end
end

# Test 50: Spline, AFT, Panel
function run_spline_aft_panel()
    @testset "Test 50: Spline, AFT, Panel" begin
        println("Running Test 50: Spline, AFT, Panel...")
        
        # 1. Setup data - simulate from Weibull AFT
        Random.seed!(50)
        nsubj = 500
        
        mu = 0.5
        sigma = 0.8
        beta = 0.5
        
        dat = DataFrame()
        
        for i in 1:nsubj
            shape_ph = 1/sigma
            scale_ph = exp(mu + beta * randn()) # Need to store x
            x_val = randn()
            scale_ph = exp(mu + beta * x_val)
            
            dist = Weibull(shape_ph, scale_ph)
            t_event = rand(dist)
            
            # Observation times
            times = [0.0, 1.0, 2.0]
            
            # Determine state at each time
            states = Int[]
            push!(states, 1) # Start at 1
            
            for t in times[2:end]
                if t_event <= t
                    push!(states, 2)
                else
                    push!(states, 1)
                end
            end
            
            # Create rows
            for j in 1:(length(times)-1)
                push!(dat, (id=i, tstart=times[j], tstop=times[j+1], statefrom=states[j], stateto=states[j+1], obstype=2, x=x_val))
            end
        end
        
        # 2. Setup Spline Model with AFT
        h12 = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; 
                     degree=3, 
                     knots=[0.5, 1.0, 1.5], 
                     boundaryknots=[0.0, 2.5],
                     linpred_effect=:aft) 
        
        model = multistatemodel(h12; data=dat)
        
        # 3. Fit model
        fitted = fit(model; verbose=true)
        
        # 4. Check convergence
        @test fitted.optim_ret.converged
        
        println("Test 50 passed!")
    end
end
