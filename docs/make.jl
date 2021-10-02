using SIMDDualNumbers
using Documenter

DocMeta.setdocmeta!(SIMDDualNumbers, :DocTestSetup, :(using SIMDDualNumbers); recursive=true)

makedocs(;
    modules=[SIMDDualNumbers],
    authors="Chris Elrod <elrodc@gmail.com> and contributors",
    repo="https://github.com/chriselrod/SIMDDualNumbers.jl/blob/{commit}{path}#{line}",
    sitename="SIMDDualNumbers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chriselrod.github.io/SIMDDualNumbers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chriselrod/SIMDDualNumbers.jl",
    devbranch="main",
)
