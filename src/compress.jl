export compress


function compress(comp_bpc::CuCompressBitsProbCircuit, data::CuMatrix; batch_size, precision = 28, Float = Float32,
                  mars_mem = nothing, tars_mem = nothing, tar_lls_mem = nothing, tar_cum_lls_mem = nothing)
    # get target marginals (Pr(X_1 = x_1, ..., X_k < x_k) and Pr(X_1 = x_1, ..., X_k = x_k))
    target_lls, target_cum_lls = get_compress_marginals(comp_bpc, data; batch_size, Float)
    println("done")
    # useful constants for rANS
    rans_l::UInt64 = one(UInt64) << 31
    tail_bits::UInt64 = (one(UInt64) << 32) - 1
    rans_prec_l::UInt64 = ((UInt64(rans_l) >> precision) << 32)
    log_prec::Float = precision * log(Float(2.0))

    # encode
    num_examples = size(data, 1)
    num_vars = size(data, 2)
    states_head = UInt64[rans_l for _ = 1 : num_examples] # the numbers at the head of the state stacks
    states_remain = Vector{UInt32}[UInt32[] for _ = 1 : num_examples]
    Threads.@threads for ex_id = 1 : num_examples
        # encode the variables in reverse order so that the decoding can be done in the right order
        for idx = num_vars : -1 : 1
            # get (ref_low, ref_ll), which correspond to an interval [ref_low, ref_ll)
            if idx > 1
                @inbounds ll_div = target_lls[ex_id, idx - 1]
                @inbounds ref_low = target_cum_lls[ex_id, idx] - ll_div
                @inbounds ref_ll = target_lls[ex_id, idx] - ll_div
            else
                @inbounds ref_low = target_cum_lls[ex_id, idx]
                @inbounds ref_ll = target_lls[ex_id, idx]
            end

            # convert (ref_low, ref_ll) to integers with precision `precision`
            ref_low_int = UInt64(ceil(exp(ref_low + log_prec)))
            ref_ll_int = UInt64(ceil(exp(ref_ll + log_prec)))
            if ref_low_int % ref_ll_int == 0 # handle this special case to avoid numerical errors
                ref_low_int -= one(UInt64)
            end

            @assert ref_ll_int > 0 "Please increase encoding precision (i.e., `precision`)."

            # rANS encoding
            @inbounds if states_head[ex_id] >= rans_prec_l * ref_ll_int
                push!(states_remain[ex_id], UInt32(states_head[ex_id] & tail_bits))
                states_head[ex_id] = (states_head[ex_id] >> 32)
            end
            @inbounds states_head[ex_id] = ((states_head[ex_id] ÷ ref_ll_int) << precision) + (states_head[ex_id] % ref_ll_int) + ref_low_int
        end
    end

    # merge `states_head` and `states_remain` into a BitVector
    @inline uint64tobitvector(v::UInt64) = begin
        bv = falses(64)
        idx = 1
        while v > UInt64(0)
            bv[idx] = Bool(v & UInt64(1))
            v = (v >> 1)
            idx += 1
        end
        bv[1:idx-1]
    end
    codes = BitVector[BitVector() for _ = 1 : num_examples]
    Threads.@threads for ex_id = 1 : num_examples
        head_bv = uint64tobitvector(states_head[ex_id])
        v = @inbounds states_remain[ex_id]
        siz = size(v, 1)
        bv = falses(siz << 5 + size(head_bv, 1) + 1)
        unsafe_copyto!(reinterpret(Ptr{UInt32}, pointer(bv.chunks)), pointer(v), siz)
        @inbounds @views bv[(siz << 5)+1:end-1] = head_bv[:]
        @inbounds bv[end] = (states_head[ex_id] > tail_bits)
        @inbounds codes[ex_id] = bv
    end

    codes
end