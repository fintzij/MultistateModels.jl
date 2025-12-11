# Part 8: Summary and Prioritization

## Executive Summary

This optimization plan identifies opportunities across 7 categories:

1. **Architecture** (01): 5 overlapping parameter systems → unify to 1
2. **Hot Paths** (02): MCEM E-step and M-step, simulation, likelihood
3. **AD Compatibility** (03): ForwardDiff + Mooncake unified backend
4. **Deduplication** (04): 8+ redundant function pairs → consolidate
5. **Profiling** (05): Scripts ready to establish baseline
6. **Implementation** (06): 7 phases with clear dependencies
7. **Allocation** (07): Pre-allocation, views, type stability

---

## Priority Ranking

### Tier 1: High Impact, Low Risk (Do First)

| Item | File(s) | Expected Impact | Effort |
|------|---------|-----------------|--------|
| Remove thin wrappers | simulation.jl | 5% fewer calls | 1 hr |
| Add @views consistently | likelihoods.jl, sampling.jl | 10% fewer allocs | 2 hr |
| Hoist Dict lookups | hazards.jl, likelihoods.jl | 5-10% speedup | 2 hr |
| Pre-allocate path vectors | simulation.jl | 15% fewer allocs | 3 hr |

**Total: ~1 day, ~20-30% allocation reduction expected**

### Tier 2: High Impact, Medium Risk (Do Second)

| Item | File(s) | Expected Impact | Effort |
|------|---------|-----------------|--------|
| Unify parameter system | helpers.jl, new file | 20% simpler code | 3 days |
| Consolidate likelihood functions | likelihoods.jl | 50% less duplication | 2 days |
| AD backend abstraction | new file | Future-proof | 2 days |

**Total: ~1 week, major code simplification**

### Tier 3: Medium Impact, Medium Risk (Do Third)

| Item | File(s) | Expected Impact | Effort |
|------|---------|-----------------|--------|
| Split phasetype.jl | phasetype.jl → 4 files | Maintainability | 1 day |
| Optimize caching | helpers.jl | Variable | 2 days |
| Batch matrix exponentials | likelihoods.jl | 10-20% speedup | 2 days |

**Total: ~1 week, conditional on profiling results**

### Tier 4: Experimental (Profile First)

| Item | File(s) | Expected Impact | Effort |
|------|---------|-----------------|--------|
| Root-finding problem reuse | sampling.jl | Unknown | 2 days |
| Memory layout changes | common.jl | Unknown | 3 days |
| SIMD hazard evaluation | hazards.jl | Unknown | 3 days |

**Total: Unknown - requires profiling data**

---

## Risk Assessment

### Breaking Changes Required

| Change | Impact | Migration Path |
|--------|--------|----------------|
| Remove `simulate_data`, `simulate_paths` | Low | Deprecation warning → removal |
| Unified parameter system | Medium | Keep old API as wrapper |
| Single likelihood function | Low | Internal change only |
| AD backend selection | Low | Opt-in configuration |

### Backwards Compatibility Strategy

```julia
# Deprecation pattern
function simulate_data(model; kwargs...)
    Base.depwarn("`simulate_data` is deprecated, use `simulate(...; data=true, paths=false)`", :simulate_data)
    return simulate(model; data=true, paths=false, kwargs...)
end
```

### Testing Requirements

- [ ] All existing tests pass
- [ ] New tests for unified parameter system
- [ ] AD gradient correctness tests (ForwardDiff vs Mooncake)
- [ ] Performance regression tests

---

## Metrics for Success

### Performance Targets

| Metric | Current (estimate) | Target |
|--------|-------------------|--------|
| `simulate_path` time | ~500 μs | < 300 μs |
| `simulate_path` allocs | ~200 | < 100 |
| `loglik_markov` (n=100) | ~50 μs | < 30 μs |
| MCEM iteration (n=50) | ~500 ms | < 300 ms |
| Memory per MCEM iter | ~50 MB | < 30 MB |

### Code Quality Targets

| Metric | Current | Target |
|--------|---------|--------|
| Lines in phasetype.jl | 4,598 | < 1,500 |
| Duplicate function pairs | ~8 | 0 |
| Parameter access methods | 5 | 1 |
| AD backends supported | 1 (ForwardDiff) | 2+ |

---

## Immediate Next Steps

1. **Run profiling scripts** (Phase 0)
   - Execute `profile_simulation.jl`
   - Execute `profile_likelihood.jl`
   - Execute `profile_mcem.jl`
   - Record baseline metrics

2. **Review results together**
   - Identify actual (not assumed) hotspots
   - Prioritize based on data

3. **Implement Tier 1 optimizations**
   - Quick wins with immediate benefit
   - Low risk of regression

4. **Re-profile and iterate**
   - Measure improvement
   - Adjust plan based on results

---

## File Index

| Document | Contents |
|----------|----------|
| `00_overview.md` | File inventory, constraints, success metrics |
| `01_architecture.md` | Parameter systems, caching, data flow |
| `02_hot_paths.md` | MCEM, simulation, likelihood analysis |
| `03_ad_compatibility.md` | ForwardDiff/Mooncake strategy |
| `04_deduplication.md` | Redundant code, API cleanup |
| `05_profiling_scripts.md` | Ready-to-run benchmark scripts |
| `06_implementation_phases.md` | Phased rollout plan |
| `07_allocation_reduction.md` | Memory optimization strategies |
| `08_summary.md` | This document |

---

## Questions for Review

Before proceeding, please clarify:

1. **Performance targets**: Are the targets above reasonable? What's acceptable slowdown during refactoring?

2. **Breaking changes timeline**: When is the next major version planned? Can we batch breaking changes?

3. **AD priority**: Is Mooncake support urgent, or can it wait until after code cleanup?

4. **Test coverage**: Are there critical paths not covered by existing tests that need tests before refactoring?

5. **Profiling environment**: Should profiling use specific hardware/configuration for reproducibility?
