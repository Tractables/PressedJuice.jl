using Test
using PressedJuice
using CUDA

include("helper/plain_dummy_circuits.jl")


@testset "compress" begin

    pc = medium_3var_categorical()

    data = cu(UInt8.([0 0 1; 2 2 1]))

    comp_bpc = CuCompressBitsProbCircuit(pc)
    codes = compress(comp_bpc, data; batch_size = 2048, precision = 24, Float = Float64)

    @test all(codes[1] .== Bool[1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 
                                0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1])
    @test all(codes[2] .== Bool[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 
                                1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1])

end