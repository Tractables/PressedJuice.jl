using ProbabilisticCircuits: foldup_aggregate


function convert_to_sd_pc(pc::ProbCircuit)
    vtree_cache = Dict{String,Vtree}()
    vtree_str_cache = Dict{Vtree,String}()
    pc2vtree = Dict{ProbCircuit,Vtree}()
    vtree2pc = Dict{Vtree,Vector{ProbCircuit}}()
    
    f_i(n)::Tuple{ProbCircuit,Vtree} = begin
        @assert length(randvars(n)) == 1
        vtree_str = "($(first(randvar(n))))"
        v = get!(vtree_cache, vtree_str) do 
            vtree = PlainVtreeLeafNode(Var(randvar(n)))
            vtree_str_cache[vtree] = vtree_str
            vtree
        end
        pc2vtree[n] = v
        if v in keys(vtree2pc)
            push!(vtree2pc[v], n)
        else
            vtree2pc[v] = [n]
        end
        n, v
    end
    f_a(n, cns)::Tuple{ProbCircuit,Vtree} = begin
        convert_to_sum(n, v) = begin
            if ismul(n)
                new_n = summate(n)
                pc2vtree[new_n] = v
                vtree2pc[v] = new_n
                new_n, v
            else
                n, v
            end
        end
        
        m, vm = cns[1]
        m, vm = convert_to_sum(m, vm)
        for (idx, (cm, vcm)) in enumerate(cns[2:end])
            cm, vcm = convert_to_sum(cm, vcm)
            vm_len = (vm isa PlainVtreeLeafNode) ? 1 : length(vm.variables)
            vcm_len = (vcm isa PlainVtreeLeafNode) ? 1 : length(vcm.variables)
            if vcm_len > vm_len
                m, cm = cm, m
                vm, vcm = vcm, vm
            end
            
            if idx < length(cns) - 1
                m = summate(multiply(m, cm))
                vtree_str = "($(vtree_str_cache[vm])|$(vtree_str_cache[vcm]))"
                vm = get!(vtree_cache, vtree_str) do
                    vtree = PlainVtreeInnerNode(vm, vcm)
                    vtree_str_cache[vtree] = vtree_str
                    vtree
                end
                pc2vtree[m] = vm
                pc2vtree[m.inputs[1]] = vm
                if vm in keys(vtree2pc)
                    push!(vtree2pc[vm], m)
                    push!(vtree2pc[vm], m.inputs[1])
                else
                    vtree2pc[vm] = [m, m.inputs[1]]
                end
            else
                m = multiply(m, cm)
                vtree_str = "($(vtree_str_cache[vm])|$(vtree_str_cache[vcm]))"
                vm = get!(vtree_cache, vtree_str) do
                    vtree = PlainVtreeInnerNode(vm, vcm)
                    vtree_str_cache[vtree] = vtree_str
                    vtree
                end
                pc2vtree[m] = vm
                if vm in keys(vtree2pc)
                    push!(vtree2pc[vm], m)
                else
                    vtree2pc[vm] = [m]
                end
            end
        end
        m, vm
    end
    f_o(n, cns)::Tuple{ProbCircuit,Vtree} = begin
        new_n = summate([item[1] for item in cns]...)
        new_n.params = n.params
        v = pc2vtree[cns[1][1]]
        pc2vtree[new_n] = v
        push!(vtree2pc[v], new_n)
        new_n, v
    end
    pc, vtree = foldup_aggregate(pc, f_i, f_a, f_o, Tuple{ProbCircuit,Vtree})
    
    pc, vtree, pc2vtree, vtree2pc
end