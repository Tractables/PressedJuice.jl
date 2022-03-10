export decompress


function decompress(comp_bpc::CuCompressBitsProbCircuit, codes::Vector{BitVector}; 
                    batch_size, precision = 28, Float = Float32,
                    mars_mem = nothing, tars_mem = nothing, data_mem = nothing, reuse = (nothing, nothing, nothing, nothing,
                    nothing, nothing, nothing, nothing, nothing, nothing, nothing))

    num_examples = length(codes)
    num_vars = num_randvars(comp_bpc)
    num_cats = num_categories(comp_bpc)
    num_nodes = length(comp_bpc.bpc.nodes)
    max_num_binary_search = Int(ceil(log2(num_cats)))

    if batch_size > num_examples
        batch_size = num_examples
    end

    # batch reuse memory
    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true); DType = Float)
    tars = prep_memory(tars_mem, (batch_size,); DType = Float)
    
    target_data = prep_memory(data_mem, (num_examples, num_vars), (true, true); DType = UInt32)

    lls_buffer = prep_memory(reuse[1], (batch_size,); DType = Float)
    cat_idxs_low = prep_memory(reuse[2], (batch_size,); DType = UInt32)
    cat_idxs_mid = prep_memory(reuse[3], (batch_size,); DType = UInt32)
    cat_idxs_high = prep_memory(reuse[4], (batch_size,); DType = UInt32)
    lls_div = prep_memory(reuse[5], (batch_size,); DType = Float)
    lls_ref = prep_memory(reuse[6], (batch_size,); DType = Float)
    lls_ref_cpu = prep_memory(reuse[6], (batch_size,); DType = Float, ArrType = Array)
    lls_cdf = prep_memory(reuse[7], (batch_size,); DType = Float)
    lls_new = prep_memory(reuse[8], (batch_size,); DType = Float)
    ref_low_int  = prep_memory(reuse[9], (batch_size,); DType = UInt64)
    ref_ll_int  = prep_memory(reuse[10], (batch_size,); DType = UInt64)
    dummy_data = prep_memory(reuse[11], (batch_size, num_vars), (true, true); DType = UInt32)

    # useful constants for rANS
    rans_l::UInt64 = one(UInt64) << 31
    tail_bits::UInt64 = (one(UInt64) << 32) - 1
    rans_prec_l::UInt64 = ((UInt64(rans_l) >> precision) << 32)
    prec_tail_bits::UInt64 = (one(UInt64) << precision) - 1
    prec_val::UInt64 = (one(UInt64) << precision)
    log_prec::Float = precision * log(Float(2.0))

    # bpc-related variables
    var_order = comp_bpc.var_order
    sub_bpcs = comp_bpc.sub_bpcs
    td_probs = comp_bpc.top_down_probs
    root_ids = comp_bpc.root_node_ids
    active_ids = comp_bpc.active_node_ids

    # work per batch
    for batch_id = 1 : Int(ceil(num_examples / batch_size))
        batch_start = (batch_id - 1) * batch_size + 1
        batch_end = min(batch_id * batch_size, num_examples)
        curr_batch_size = batch_end - batch_start + 1

        clear_mar(mars, comp_bpc.bpc)
        @inbounds @views lls_div .= zero(Float)

        # convert code to its unflattened form
        @inline bitvectortouint64(c::BitVector) = begin
            if length(c) == 0
                zero(UInt64)
            else
                v::UInt64 = zero(UInt64)
                for idx = length(c) : -1 : 1
                    v = (v << 1)
                    if c[idx]
                        v |= UInt64(1)
                    end
                end
                v
            end
        end
        states_head = UInt64[rans_l for _ = 1 : curr_batch_size] # The numbers at the head of the state stacks
        states_remain = Vector{UInt32}[UInt32[] for _ = 1 : curr_batch_size]
        Threads.@threads for i = 1 : curr_batch_size
            ex_id = batch_start + i - 1
            siz = @inbounds length(codes[ex_id]) - 1
            h_len = codes[ex_id][end] ? 2 : 1
            s_len = Int(ceil(siz / 32)) - h_len
            states_remain[i] = zeros(UInt32, s_len)
            unsafe_copyto!(pointer(states_remain[i]), reinterpret(Ptr{UInt32}, pointer(codes[ex_id].chunks)), s_len)
            states_head[i] = @inbounds bitvectortouint64(codes[ex_id][(s_len << 5)+1:end-1])
        end

        # decode the variables one by one
        for (idx, var_idx) in enumerate(var_order)
            # compute the reference LLs for variable #`var_idx`
            Threads.@threads for j = 1 : curr_batch_size
                @inbounds lls_ref_cpu[j] = log((states_head[j] & prec_tail_bits)) - log_prec
            end
            lls_ref .= cu(lls_ref_cpu) #### to be optimized

            # use the dichotomy search method to locate the target categories
            @inbounds @views cat_idxs_low .= UInt32(0)
            @inbounds @views cat_idxs_high .= UInt32(num_cats)
            @inbounds @views lls_cdf .= typemin(Float)
            for _ = 1 : max_num_binary_search # always take the maximum possible number of iters
                # compute `cat_idxs_mid` as the mean of `cat_idxs_low` and `cat_idxs_high`
                compute_mean(dummy_data, var_idx, cat_idxs_low, cat_idxs_mid, cat_idxs_high)

                ll_partial_propagation(sub_bpcs[var_idx], active_ids[var_idx], td_probs[var_idx], root_ids[var_idx], 
                                       mars, dummy_data, 1:curr_batch_size, lls_buffer; cum_input = true)
                
                # update `cat_idxs_low`, `cat_idxs_high`, and the corresponding LLs
                update_lls_and_val_bounds(cat_idxs_low, cat_idxs_mid, cat_idxs_high, lls_cdf, lls_buffer, lls_div, lls_ref)
            end
            
            # Assign decoded categories of variable #`var_idx`
            @inbounds @views target_data[batch_start:batch_end, var_idx] .= cat_idxs_low[1:curr_batch_size]

            # P(X_1 = x_1, ..., X_k = x_k)
            @inbounds @views dummy_data[:,var_idx] .= cat_idxs_low
            ll_partial_propagation(sub_bpcs[var_idx], active_ids[var_idx], td_probs[var_idx], root_ids[var_idx], 
                                   mars, dummy_data, 1:curr_batch_size, lls_new; cum_input = false)
            
            # Update the rANS states
            update_rans_codes(lls_cdf, lls_new, lls_div, log_prec, ref_low_int, ref_ll_int)
            ref_ll_int_cpu = Array(ref_ll_int) ###### to be optimized
            ref_low_int_cpu = Array(ref_low_int) ###### to be optimized
            
            cf = (states_head .& prec_tail_bits)
            states_head .= (states_head .>> precision) .* ref_ll_int_cpu .+ cf .- ref_low_int_cpu
            Threads.@threads for j = 1 : curr_batch_size
                if states_head[j] < rans_l
                    if length(states_remain[j]) != 0
                        states_head[j] = (states_head[j] << 32) | UInt64(pop!(states_remain[j]))
                    else
                        states_head[j] = (states_head[j] << 32)
                    end
                end
            end
            
            @inbounds @views lls_div[:] .= lls_new[:]
        end
    end
    
    Array(target_data)
end

#####################
## Helper functions
#####################

num_categories(comp_bpc::Union{CompressBitsProbCircuit, CuCompressBitsProbCircuit}) = begin
    num_cats = 0
    bpc = comp_bpc.bpc
    nodes = Array(bpc.nodes)
    input_node_ids = Array(bpc.input_node_ids)
    for node_id in input_node_ids
        node = nodes[node_id]
        @assert node isa BitsInput
        d = dist(node)
        num_cats = max(num_cats, d.num_cats)
    end
    num_cats
end

function compute_mean_kernel(dummy_data, var_idx, cat_idxs_low, cat_idxs_mid, cat_idxs_high)
    idx = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if idx <= length(cat_idxs_low)
        val = (cat_idxs_low[idx] + cat_idxs_high[idx]) ÷ UInt32(2)
        cat_idxs_mid[idx] = val
        dummy_data[idx, var_idx] = val
    end
    nothing
end

function compute_mean(dummy_data, var_idx, cat_idxs_low, cat_idxs_mid, cat_idxs_high)
    args = (dummy_data, var_idx, cat_idxs_low, cat_idxs_mid, cat_idxs_high)
    kernel = @cuda name="compute_mean" launch=false compute_mean_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(cat_idxs_low), threads)

    kernel(args...; threads, blocks)
    nothing
end

function update_lls_and_val_bounds_kernel(cat_idxs_low, cat_idxs_mid, cat_idxs_high, lls_cdf, lls_buffer, lls_div, lls_ref)
    idx = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if idx <= length(cat_idxs_low)
        cond = (lls_buffer[idx] - lls_div[idx] < lls_ref[idx])
        if cond
            cat_idxs_low[idx] = cat_idxs_mid[idx]
            lls_cdf[idx] = lls_buffer[idx]
        else
            cat_idxs_high[idx] = cat_idxs_mid[idx]
        end
    end
    nothing
end

function update_lls_and_val_bounds(cat_idxs_low, cat_idxs_mid, cat_idxs_high, lls_cdf, lls_buffer, lls_div, lls_ref)
    args = (cat_idxs_low, cat_idxs_mid, cat_idxs_high, lls_cdf, lls_buffer, lls_div, lls_ref)
    kernel = @cuda name="update_lls_and_val_bounds" launch=false update_lls_and_val_bounds_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(cat_idxs_low), threads)

    kernel(args...; threads, blocks)
    nothing
end

function update_rans_codes_kernel(lls_cdf, lls_new, lls_div, log_prec, ref_low_int, ref_ll_int)
    idx = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if idx <= length(lls_cdf)
        val1 = UInt64(ceil(exp(lls_cdf[idx] - lls_div[idx] + log_prec)))
        val2 = UInt64(ceil(exp(lls_new[idx] - lls_div[idx] + log_prec)))
        if val1 % val2 == zero(UInt64)
            val1 = val1 - one(UInt64)
        end
        ref_low_int[idx] = val1
        ref_ll_int[idx] = val2
    end
    nothing
end

function update_rans_codes(lls_cdf, lls_new, lls_div, log_prec, ref_low_int, ref_ll_int)
    args = (lls_cdf, lls_new, lls_div, log_prec, ref_low_int, ref_ll_int)
    kernel = @cuda name="update_rans_codes" launch=false update_rans_codes_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(lls_cdf), threads)

    kernel(args...; threads, blocks)
    nothing
end