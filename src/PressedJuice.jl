module PressedJuice

    using LogicCircuits
    using ProbabilisticCircuits
    using CUDA


    include("transformations.jl")

    include("bitcircuit.jl")
    include("likelihood.jl")

    include("compress.jl")
    include("decompress.jl")

end