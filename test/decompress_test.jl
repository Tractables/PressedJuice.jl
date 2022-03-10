using Test
using PressedJuice
using CUDA

include("helper/plain_dummy_circuits.jl")


@testset "compress" begin

    pcs = [little_3var_categorical(), little2_3var_categorical(), medium_3var_categorical()]

    data = cu(UInt8.([0 0 1; 2 2 1]))

    for pc in pcs

        comp_bpc = CuCompressBitsProbCircuit(pc)
        codes = compress(comp_bpc, data; batch_size = 2, precision = 24, Float = Float64)

        dec_data = decompress(comp_bpc, codes; batch_size = 2, precision = 24, Float = Float64)

        @test all(dec_data .== Array(data))

    end

end