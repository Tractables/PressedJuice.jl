export CuCompressBitsProbCircuit

using ProbabilisticCircuits: BitsNode, BitsEdge, FlatVectors, BitsCategorical, BitsProbCircuit
using StatsFuns: logaddexp, logsumexp


function unflatten(v::FlatVectors{Vector{F}}) where F
    vecs = Vector{F}[]
    vec_start = 1
    for vec_end in v.ends
        push!(vecs, v.vectors[vec_start:vec_end])
        vec_start = vec_end + 1
    end
    vecs
end

import Base: strip # extend

function strip(v::Vector{Vector{F}}) where F
    vecs = Vector{F}[]
    for vec in v
        if length(vec) > 0
            push!(vecs, vec)
        end
    end
    vecs
end

"A BitPC for the upward pass"
struct BitsProbCircuitUp

    # all the nodes in the circuit
    nodes::Vector{BitsNode}

    # mapping from BitPC to PC nodes
    nodes_map::Union{Vector{ProbCircuit},Nothing}

    # the ids of the subset of nodes that are inputs
    input_node_ids::Vector{UInt32}

    # layers of edges for upward pass
    edge_layers::FlatVectors{Vector{BitsEdge}}

    # memory used by input nodes for their parameters
    heap::Vector{Float32}

    BitsProbCircuitUp(n, nm, in, e, h) = begin
        if nm === nothing
            @assert length(n) >= length(in) > 0
        else
            @assert length(n) == length(nm) >= length(in) > 0
        end
        @assert allunique(e.ends) "No empty layers allowed"
        if length(e.vectors) == 0
            e = FlatVectors(BitsEdge[], e.ends)
        end
        new(n, nm, in, e, h)
    end
end

struct CuBitsProbCircuitUp{BitsNodes <: BitsNode}

    # all the nodes in the circuit
    nodes::CuVector{BitsNodes}

    # mapping from BitPC to PC nodes
    nodes_map::Union{Vector{ProbCircuit},Nothing}

    # the ids of the subset of nodes that are inputs
    input_node_ids::CuVector{UInt32}

    # layers of edges for upward pass
    edge_layers::FlatVectors{<:CuVector{BitsEdge}}

    # memory used by input nodes for their parameters
    heap::CuVector{Float32}

    CuBitsProbCircuitUp(bpc; heap = nothing) = begin
        # find union of bits node types actually used in the circuit
        BitsNodes = mapreduce(typeof, (x,y) -> Union{x,y}, bpc.nodes)
        @assert Base.isbitsunion(BitsNodes) || Base.isbitstype(BitsNodes)
        nodes = CuVector{BitsNodes}(bpc.nodes)
        input_node_ids = cu(bpc.input_node_ids)
        edge_layers = cu(bpc.edge_layers)
        if heap === nothing
            heap = cu(bpc.heap)
        end
        new{BitsNodes}(nodes, bpc.nodes_map, input_node_ids, edge_layers, heap)
    end
end

function BitsProbCircuitUp(pc::ProbCircuit)
    bpc = BitsProbCircuit(pc)
    BitsProbCircuitUp(bpc.nodes, bpc.nodes_map, bpc.input_node_ids, bpc.edge_layers_up, bpc.heap)
end

CuBitsProbCircuitUp(pc::ProbCircuit) = 
    CuBitsProbCircuitUp(BitsProbCircuitUp(pc))

struct CompressBitsProbCircuit{F <: AbstractFloat}

    # the original BitPC
    bpc::BitsProbCircuitUp

    # the variable order
    var_order::Vector{Var}

    # sub-PCs for all variables (indexed by variable id)
    sub_bpcs::Vector{BitsProbCircuitUp}

    # top-down probabilities for all variables (indexed by variable id)
    top_down_probs::Vector{Vector{F}}

    # root node indices for all variables (indexed by variable id)
    root_node_ids::Vector{Vector{UInt32}}

    # indices of all nodes in this sub BitPC
    active_node_ids::Vector{Vector{UInt32}}

    CompressBitsProbCircuit(bpc, var_order, sub_bpcs, td_probs, root_ids, active_ids) = begin
        @assert length(var_order) == length(sub_bpcs)
        for i = 1 : length(var_order)
            @assert length(td_probs[i]) == length(root_ids[i])
            @assert sub_bpcs[i].heap === bpc.heap
        end
        F = eltype(td_probs[1])
        new{F}(bpc, var_order, sub_bpcs, td_probs, root_ids, active_ids)
    end
end

struct CuCompressBitsProbCircuit{F <: AbstractFloat}

    # the original BitPC
    bpc::CuBitsProbCircuitUp

    # the variable order
    var_order::Vector{Var}

    # sub-PCs for all variables (indexed by variable id)
    sub_bpcs::Vector{<:CuBitsProbCircuitUp}

    # top-down probabilities for all variables (indexed by variable id)
    top_down_probs::Vector{<:CuVector{F}}

    # root node indices for all variables (indexed by variable id)
    root_node_ids::Vector{<:CuVector{UInt32}}

    # indices of all nodes in this sub BitPC
    active_node_ids::Vector{<:CuVector{UInt32}}

end

CuCompressBitsProbCircuit(comp_bpc::CompressBitsProbCircuit) = begin
    bpc = CuBitsProbCircuitUp(comp_bpc.bpc)
    sub_bpcs = map(x -> CuBitsProbCircuitUp(x; heap = bpc.heap), comp_bpc.sub_bpcs)
    top_down_probs = map(cu, comp_bpc.top_down_probs)
    root_node_ids = map(cu, comp_bpc.root_node_ids)
    active_node_ids = map(cu, comp_bpc.active_node_ids)

    CuCompressBitsProbCircuit(bpc, comp_bpc.var_order, sub_bpcs, top_down_probs, root_node_ids, active_node_ids)
end

function CompressBitsProbCircuit(pc::ProbCircuit; Float = Float64)
    pc, vtree, pc2vtree, vtree2pc = convert_to_sd_pc(pc)
    var_order = get_flattened_vtree_var_order(vtree)
    vtree_order = get_vtree_computation_order(vtree, var_order)
    td_probs = get_top_down_probs(pc; Float)

    # construct BitPC for the full PC
    bpc = BitsProbCircuitUp(pc)

    # record (left-side) cumulative probabilities for categorical input nodes
    heap = bpc.heap
    foreach(bpc.nodes) do bitn
        if bitn isa BitsInput && bitn.dist isa BitsCategorical
            d = bitn.dist
            heap_start = d.heap_start
            num_cats = d.num_cats

            heap[heap_start + num_cats] = typemin(eltype(heap))
            for i = 1 : num_cats - 1
                heap[heap_start+num_cats+i] = logaddexp(heap[heap_start+num_cats+i-1], heap[heap_start+i-1])
            end
        end
    end

    num_vars = length(var_order)
    num_nodes = length(bpc.nodes)

    unflat_edge_layers = unflatten(bpc.edge_layers)
    num_layers = length(unflat_edge_layers)

    # construct sub-BitPCs
    sub_bpcs = Vector{BitsProbCircuitUp}()
    top_down_probs = Vector{Float}[]
    root_node_ids = Vector{UInt32}[]
    active_node_ids = Vector{UInt32}[]
    for var_idx = 1 : num_vars
        target_vtrees = vtree_order[var_idx]

        # select relevant nodes
        nodes = Vector{BitsNode}()
        node_ids = Set{UInt32}()
        input_node_ids = Vector{UInt32}()
        for node_idx = 1 : num_nodes
            n = bpc.nodes_map[node_idx]
            vtree = pc2vtree[n]
            if vtree in target_vtrees
                push!(nodes, deepcopy(bpc.nodes[node_idx]))
                push!(node_ids, UInt32(node_idx))
                if isinput(n)
                    push!(input_node_ids, node_idx)
                end
            end
        end

        # select relevant edges
        edge_layers = Vector{BitsEdge}[]
        for layer_id = 1 : num_layers
            curr_layer = unflat_edge_layers[layer_id]
            for edge_id = 1 : length(curr_layer)
                edge = curr_layer[edge_id]
                if edge.parent_id in node_ids
                    while length(edge_layers) < layer_id
                        push!(edge_layers, BitsEdge[])
                    end
                    push!(edge_layers[layer_id], deepcopy(edge))
                end
            end
        end

        # select root nodes
        if length(edge_layers) > 0
            @assert length(edge_layers[end]) > 0
            curr_root_ids = Vector{UInt32}()
            for edge in edge_layers[end]
                node_id = edge.parent_id
                if !(node_id in curr_root_ids)
                    push!(curr_root_ids, node_id)
                end
            end
        else
            # in this case, no need to compute inner node probabilities
            # all nodes must be input nodes

            curr_root_ids = deepcopy(input_node_ids)
        end

        curr_td_probs = Vector{Float}()
        for node_id in curr_root_ids
            n = bpc.nodes_map[node_id]
            push!(curr_td_probs, td_probs[n])
        end
        @assert isapprox(sum(exp.(curr_td_probs)), 1.0, atol = 1e-3)
        curr_td_probs .-= logsumexp(curr_td_probs) # normalize out left-over errors

        edge_layers = FlatVectors(strip(edge_layers))
        sub_bpc = BitsProbCircuitUp(bpc.nodes, nothing, input_node_ids, edge_layers, bpc.heap)

        # add to the vectors
        push!(sub_bpcs, sub_bpc)
        push!(top_down_probs, curr_td_probs)
        push!(root_node_ids, curr_root_ids)
        push!(active_node_ids, collect(node_ids))
    end

    CompressBitsProbCircuit(bpc, var_order, sub_bpcs, top_down_probs, root_node_ids, active_node_ids)
end

CuCompressBitsProbCircuit(pc::ProbCircuit; Float = Float64) = 
    CuCompressBitsProbCircuit(CompressBitsProbCircuit(pc; Float))

import ProbabilisticCircuits: randvars, num_randvars # extend

num_randvars(comp_bpc::Union{CompressBitsProbCircuit, CuCompressBitsProbCircuit}) = length(randvars(comp_bpc))

randvars(comp_bpc::Union{CompressBitsProbCircuit, CuCompressBitsProbCircuit}) = begin
    vars = BitSet()
    bpc = comp_bpc.bpc
    nodes = Array(bpc.nodes)
    input_node_ids = Array(bpc.input_node_ids)
    for node_id in input_node_ids
        node = nodes[node_id]
        @assert node isa BitsInput
        push!(vars, node.variable)
    end
    vars
end

#####################
## Helper functions 
#####################

function get_flattened_vtree_var_order(vtree::Vtree; cache::Set{Vtree} = Set{Vtree}())
    var_order = Vector{Var}()
    
    dfs(v::Vtree) = begin
        if v in cache
            return
        end
        push!(cache, v)
        
        if v isa PlainVtreeLeafNode
            push!(var_order, v.var)
        else
            dfs(v.left)
            dfs(v.right)
        end
        
        nothing
    end
    
    dfs(vtree)
    
    var_order
end


function get_vtree_computation_order(vtree::Vtree, var_order::Vector{Var})
    num_vars = length(var_order)
    vtree_order::Vector{Set{Vtree}} = Set{Vtree}[Set{Vtree}() for i = 1 : num_vars]
    leaf_nodes::Dict{Var,Vtree} = Dict{Var,Vtree}()
    
    f(v) = begin
        if v isa PlainVtreeLeafNode
            leaf_nodes[v.var] = v
        end
    end
    foreach(f, vtree)
    
    for i = 1 : num_vars
        var_idx = var_order[i]
        v = leaf_nodes[var_idx]
        s = BitSet(var_order[1:i])
        push!(vtree_order[var_idx], v)
        while length(setdiff(s, (v isa PlainVtreeLeafNode ? [v.var] : v.variables))) > 0
            v = v.parent
            push!(vtree_order[var_idx], v)
        end
    end
    
    vtree_order # note that `vtree_order` is indexed by the actual variable idx
end


function get_top_down_probs(pc::ProbCircuit; Float = Float64)
    top_down_prob = Dict{ProbCircuit, Float}()
    
    process(n::ProbCircuit) = begin
        n_td_prob = top_down_prob[n]
        if issum(n)
            for (c, p) in zip(n.inputs, n.params)
                logprob = n_td_prob + p
                c_td_prob = get(top_down_prob, c, -Inf)
                top_down_prob[c] = logaddexp(c_td_prob, logprob)
            end
        elseif ismul(n)
            for c in n.inputs
                c_td_prob = get(top_down_prob, c, -Inf)
                top_down_prob[c] = logaddexp(c_td_prob, n_td_prob)
            end
        end
    end
    
    top_down_prob[pc] = zero(Float)
    foreach_down(process, pc)
    
    top_down_prob
end