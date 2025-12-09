# Deprecated Tests

This folder holds test files that no longer run under `Pkg.test()` but are retained for reference.

- `longtest_*.jl` were exploratory Monte Carlo scripts predating the current infrastructure. They
	require manual invocation and can take a long time to finish.
- `test_loglik.jl` exercises the legacy likelihood code path and depends on helpers removed from
	the active test suite.
- `test_pathfunctions.jl` targets an old `observe_path` API that relies on setup files we no longer
	ship with the default test harness.

Run any of these scripts directly from this directory if you need to reproduce historical behavior,
but expect unmaintained APIs and missing fixtures.
