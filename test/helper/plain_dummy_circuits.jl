using ProbabilisticCircuits
using ProbabilisticCircuits: PlainInputNode


function little_3var_categorical(firstvar = 1; num_cats = 3)
    n1 = PlainInputNode(firstvar, Categorical(num_cats))
    n2 = PlainInputNode(firstvar + 1, Categorical(num_cats))
    n3 = PlainInputNode(firstvar + 2, Categorical(num_cats))
    pc = summate(multiply(n1, n2, n3))
    init_parameters(pc; perturbation = 0.5)
    pc
end

function little2_3var_categorical(firstvar = 1; num_cats = 3)
    n1_1 = PlainInputNode(firstvar, Categorical(num_cats))
    n1_2 = PlainInputNode(firstvar, Categorical(num_cats))
    n2 = PlainInputNode(firstvar + 1, Categorical(num_cats))
    n3 = PlainInputNode(firstvar + 2, Categorical(num_cats))
    pc = summate(multiply(summate(n1_1, n1_2), n2, n3))
    init_parameters(pc; perturbation = 0.5)
    pc
end

function medium_3var_categorical(firstvar = 1; num_cats = 3)
    n1_1 = PlainInputNode(firstvar, Categorical(num_cats))
    n1_2 = PlainInputNode(firstvar, Categorical(num_cats))
    n2_1 = PlainInputNode(firstvar + 1, Categorical(num_cats))
    n2_2 = PlainInputNode(firstvar + 1, Categorical(num_cats))
    n3_1 = PlainInputNode(firstvar + 2, Categorical(num_cats))
    n3_2 = PlainInputNode(firstvar + 2, Categorical(num_cats))
    pc = summate(multiply(n1_1, n2_1, n3_1), multiply(n1_2, n2_2, n3_2))
    init_parameters(pc; perturbation = 0.5)
    pc
end