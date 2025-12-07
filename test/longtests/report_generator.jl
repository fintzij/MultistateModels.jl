
using Dates
using Printf

"""
    generate_longtest_report(results::Vector{TestResult})

Generate a Markdown report summarizing the results of the long tests.
"""
function generate_longtest_report(results::Vector{TestResult})
    # Ensure output directory exists
    mkpath(OUTPUT_DIR)
    report_path = joinpath(OUTPUT_DIR, "inference_longtests.md")
    
    open(report_path, "w") do io
        println(io, "# Inference Long Tests Report")
        println(io, "Generated on: $(Dates.now())")
        println(io, "")
        
        # Summary Statistics
        n_total = length(results)
        n_passed = count(r -> r.passed, results)
        pass_rate = n_total > 0 ? (n_passed / n_total) * 100 : 0.0
        
        println(io, "## Summary")
        println(io, "- **Total Tests**: $n_total")
        println(io, "- **Passed**: $n_passed")
        println(io, "- **Failed**: $(n_total - n_passed)")
        println(io, "- **Pass Rate**: $(@sprintf("%.1f", pass_rate))%")
        println(io, "")
        
        # Results Table
        println(io, "## Detailed Results")
        println(io, "| Test Name | Family | Covariates | Data Type | Max Rel Error (%) | Status |")
        println(io, "|---|---|---|---|---|---|")
        
        for res in results
            status = res.passed ? "✅ PASS" : "❌ FAIL"
            max_err = isnan(res.max_rel_error) ? "NaN" : @sprintf("%.2f", res.max_rel_error)
            
            println(io, "| $(res.name) | $(res.family) | $(res.covariates) | $(res.data_type) | $max_err | $status |")
        end
        println(io, "")
        
        # Failures Section
        failures = filter(r -> !r.passed, results)
        if !isempty(failures)
            println(io, "## Failures Analysis")
            for fail in failures
                println(io, "### $(fail.name)")
                println(io, "- **Family**: $(fail.family)")
                println(io, "- **Max Relative Error**: $(@sprintf("%.2f", fail.max_rel_error))%")
                println(io, "- **Parameter Errors**:")
                for (param, err) in fail.rel_errors
                    println(io, "  - `$param`: $(@sprintf("%.2f", err))%")
                end
                println(io, "")
            end
        end
    end
    
    @info "Report generated at $report_path"
end
