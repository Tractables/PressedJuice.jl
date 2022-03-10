using ProbabilisticCircuits: cleanup_memory, balance_threads, BitsSum, BitsMul, BitsInput, isfirst, islast, SumEdge


function get_compress_marginals(comp_bpc::CuCompressBitsProbCircuit, data::CuMatrix; batch_size, Float = Float32,
                                mars_mem = nothing, tars_mem = nothing, tar_lls_mem = nothing, tar_cum_lls_mem = nothing)
    num_examples = size(data, 1)
    num_vars = size(data, 2)
    num_nodes = length(comp_bpc.bpc.nodes)

    # batch reuse memory
    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true); DType = Float)
    tars = prep_memory(tars_mem, (batch_size,); DType = Float)

    # targets
    target_lls = prep_memory(tar_lls_mem, (num_examples, num_vars), (true, true); DType = Float)
    target_cum_lls = prep_memory(tar_cum_lls_mem, (num_examples, num_vars), (true, true); DType = Float)

    var_order = comp_bpc.var_order
    sub_bpcs = comp_bpc.sub_bpcs
    td_probs = comp_bpc.top_down_probs
    root_ids = comp_bpc.root_node_ids
    active_ids = comp_bpc.active_node_ids

    for batch_id = 1 : Int(ceil(num_examples / batch_size))
        batch_start = (batch_id - 1) * batch_size + 1
        batch_end = min(batch_id * batch_size, num_examples)
        batch = batch_start : batch_end
        curr_batch_size = batch_end - batch_start + 1

        clear_mar(mars, comp_bpc.bpc)

        for (idx, var_idx) in enumerate(var_order)
            # Pr(X_1 = x_1, ..., X_k < x_k)
            ll_partial_propagation(sub_bpcs[var_idx], active_ids[var_idx], td_probs[var_idx], root_ids[var_idx], 
                                   mars, data, batch, tars; cum_input = true)
            target_cum_lls[batch_start:batch_end, idx] .= tars[1:curr_batch_size]

            # Pr(X_1 = x_1, ..., X_k = x_k)
            ll_partial_propagation(sub_bpcs[var_idx], active_ids[var_idx], td_probs[var_idx], root_ids[var_idx], 
                                   mars, data, batch, tars; cum_input = false)
            target_lls[batch_start:batch_end, idx] .= tars[1:curr_batch_size]
        end
    end

    target_lls_cpu = Array(target_lls)
    target_cum_lls_cpu = Array(target_cum_lls)

    cleanup_memory((mars, mars_mem), (tars, tars_mem), (target_lls, tar_lls_mem), (target_cum_lls, tar_cum_lls_mem))

    target_lls_cpu, target_cum_lls_cpu
end

###################################
## Likelihood partial propagation
###################################

function ll_partial_propagation(bpc::CuBitsProbCircuitUp, active_node_ids, td_probs, root_ids, mars, data, 
                                example_ids, tars; cum_input, mine = 2, maxe = 32)
    # input layer
    if cum_input
        value_fn = cum_loglikelihood
    else
        value_fn = loglikelihood
    end
    init_mar!(mars, bpc, active_node_ids, data, example_ids, value_fn; mine, maxe)

    # inner layers (if any)
    if length(bpc.edge_layers.vectors) > 0
        layer_start = 1
        for layer_end in bpc.edge_layers.ends
            layer_up(mars, bpc, layer_start, layer_end, length(example_ids); mine, maxe)
            layer_start = layer_end + 1
        end
    end

    # collect lls
    tars .= typemin(eltype(tars))
    record_lls(mars, td_probs, root_ids, tars, length(example_ids); mine, maxe)

    nothing
end

function clear_mar_kernel(mars::CuDeviceMatrix{Float,1}, nodes, num_ex_threads::Int32, node_work::Int32) where Float <: AbstractFloat
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work 
    node_end = min(node_start + node_work - one(Int32), length(nodes))

    @inbounds if ex_id <= size(mars, 1)
        for node_id = node_start:node_end
            node = nodes[node_id]
            mars[ex_id, node_id] = 
                if node isa BitsSum || node isa BitsInput
                    zero(Float)
                else # node isa BitsMul
                    typemin(Float)
                end
        end
    end
    nothing
end

function clear_mar(mars, bpc; mine = 2, maxe = 32)
    num_examples = size(mars, 1)
    num_nodes = length(bpc.nodes)
    
    dummy_args = (mars, bpc.nodes, Int32(1), Int32(1))
    kernel = @cuda name="clear_mar" launch=false clear_mar_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_nodes, num_examples, config; mine, maxe)
    
    args = (mars, bpc.nodes, Int32(num_example_threads), Int32(node_work))
    kernel(args...; threads, blocks)
    nothing
end

cum_loglikelihood(d::BitsCategorical, value, heap) = 
    heap[d.heap_start + d.num_cats + UInt32(value)]

"Input node left-side cumulative probability"
function init_mar!_kernel(mars::CuDeviceMatrix{Float,1}, nodes, active_node_ids, data, example_ids, heap, 
                              num_ex_threads::Int32, node_work::Int32, value_fn) where Float <: AbstractFloat
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work 
    node_end = min(node_start + node_work - one(Int32), length(active_node_ids))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start:node_end
            orig_node_id = active_node_ids[node_id]
            node = nodes[orig_node_id]
            
            mars[ex_id, orig_node_id] = 
                if node isa BitsSum
                    typemin(Float)
                elseif node isa BitsMul
                    zero(Float)
                else # node isa BitsInput
                    orig_ex_id::Int32 = example_ids[ex_id]
                    inputnode = node::BitsInput
                    variable = inputnode.variable
                    value = data[orig_ex_id, variable]
                    if ismissing(value)
                        zero(Float)
                    else
                        value_fn(dist(inputnode), value, heap)
                    end
                end
        end
    end
    nothing
end

function init_mar!(mars, bpc, active_node_ids, data, example_ids, value_fn; mine, maxe)
    num_examples = length(example_ids)
    num_nodes = length(bpc.nodes)
    
    dummy_args = (mars, bpc.nodes, active_node_ids, data, example_ids, bpc.heap, Int32(1), Int32(1), value_fn)
    kernel = @cuda name="init_mar!" launch=false init_mar!_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(active_node_ids), num_examples, config; mine, maxe)
    
    args = (mars, bpc.nodes, active_node_ids, data, example_ids, bpc.heap, 
            Int32(num_example_threads), Int32(node_work))
    kernel(args...; threads, blocks)
    nothing
end

function layer_up_kernel(mars::CuDeviceMatrix{Float,1}, edges, num_ex_threads::Int32, num_examples::Int32, 
                         layer_start::Int32, edge_work::Int32, layer_end::Int32) where Float <: AbstractFloat

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    @inbounds if ex_id <= num_examples

        local acc::Float   
        owned_node::Bool = false
        
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            tag = edge.tag
            isfirstedge = isfirst(tag)
            islastedge = islast(tag)
            issum = edge isa SumEdge
            owned_node |= isfirstedge

            # compute probability coming from child
            child_prob = mars[ex_id, edge.prime_id]
            if edge.sub_id != 0
                child_prob += mars[ex_id, edge.sub_id]
            end
            if issum
                child_prob += edge.logp
            end

            # accumulate probability from child
            if isfirstedge || (edge_id == edge_start)  
                acc = child_prob
            elseif issum
                acc = logsumexp(acc, child_prob)
            else
                acc += child_prob
            end

            # write to global memory
            if islastedge || (edge_id == edge_end)   
                pid = edge.parent_id

                if islastedge && owned_node
                    # no one else is writing to this global memory
                    mars[ex_id, pid] = acc
                else
                    if issum
                        CUDA.@atomic mars[ex_id, pid] = logsumexp(mars[ex_id, pid], acc)
                    else
                        CUDA.@atomic mars[ex_id, pid] += acc
                    end 
                end    
            end
        end
    end
    nothing
end

function layer_up(mars, bpc, layer_start, layer_end, num_examples; mine, maxe)
    edges = bpc.edge_layers.vectors
    num_edges = layer_end - layer_start + 1
    dummy_args = (mars, edges, 
                  Int32(32), Int32(num_examples), 
                  Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="layer_up" launch=false layer_up_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe)
    
    args = (mars, edges, 
            Int32(num_example_threads), Int32(num_examples), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end))
    kernel(args...; threads, blocks)
    nothing
end

function record_lls_kernel(mars::CuDeviceMatrix{Float,1}, td_probs, root_ids, tars, num_ex_threads::Int32, 
                           num_examples::Int32, node_work::Int32) where Float <: AbstractFloat
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = 1 + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(root_ids))

    @inbounds if ex_id <= num_examples
        local acc::Float = typemin(Float)

        for node_id = node_start : node_end
            ori_node_id = root_ids[node_id]
            acc = logsumexp(acc, mars[ex_id, ori_node_id] + td_probs[node_id])
        end

        CUDA.@atomic tars[ex_id] = logsumexp(tars[ex_id], acc)
    end
    nothing
end

function record_lls(mars, td_probs, root_ids, tars, num_examples; mine = 2, maxe = 32)
    dummy_args = (mars, td_probs, root_ids, tars, Int32(32), Int32(num_examples), Int32(1))
    kernel = @cuda name="record_lls" launch=false record_lls_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(root_ids), num_examples, config; mine, maxe)

    args = (mars, td_probs, root_ids, tars, Int32(num_example_threads), Int32(num_examples), Int32(node_work))
    kernel(args...; threads, blocks)
    nothing
end

#####################
## Helper functions
#####################

function prep_memory(reuse, sizes, exact = map(x -> true, sizes); DType = Float32, ArrType = CuArray)
    if isnothing(reuse)
        return ArrType{DType}(undef, sizes...)
    else
        @assert ndims(reuse) == length(sizes)
        for d = 1:length(sizes)
            if exact[d]
                @assert size(reuse, d) == sizes[d] 
            else
                @assert size(reuse, d) >= sizes[d] 
            end
        end
        return reuse
    end
end

import StatsFuns: logsumexp

function logsumexp(x::Float32, y::Float32)
    if isfinite(x) && isfinite(y)
        @fastmath max(x,y) + log1p(exp(-abs(x-y))) 
    else
        max(x,y)
    end
end

function logsumexp(x::Float64, y::Float64)
    if isfinite(x) && isfinite(y)
        @fastmath max(x,y) + log1p(exp(-abs(x-y))) 
    else
        max(x,y)
    end
end