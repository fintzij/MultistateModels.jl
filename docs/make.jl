using Documenter, MultistateModels

makedocs(
    sitename="MultistateModels.jl",
    pages=[
        "Home" => "index.md",
        "Optimization and Variance" => "optimization.md",
        "Phase-Type FFBS Algorithm" => "phasetype_ffbs.md",
    ]
)