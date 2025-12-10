"""
Unified long test entry point for phase-type hazard model validation.

This file groups the existing long tests for:

  * Phase-type hazard models with exact data (`longtest_phasetype_exact.jl`)
  * Phase-type hazard models with panel/mixed data (`longtest_phasetype_panel.jl`)

The underlying test logic remains in the original files; this wrapper
simply includes them so that the phase-type long tests can be treated as
one logical suite from the test harness.
"""

include(joinpath(@__DIR__, "longtest_phasetype_exact.jl"))
include(joinpath(@__DIR__, "longtest_phasetype_panel.jl"))
