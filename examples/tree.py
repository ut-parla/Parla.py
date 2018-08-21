# George's tree.py (in mock_ups) rewritten using the concepts and
# syntax of Arthur's Parla strawman.

# "???" marks things that were not specified in the original, but are
# needed in this version and Arthur couldn't figure out. Pipeline
# parallelism as mentioned in the orginal is NOT possible with Parla
# as it currently exists, but there are hypothetical "with reads"
# blocks which if supported should do what is needed. However,
# supporting them would require the compiler to track indicies through
# lookup tables (unless I misunderstand), which would be very
# difficult and might require a complex runtime (like phase completion
# information stored in data structures).

def create_tree(points, maxDepth=30):
    '''
    Sketch of algorithm that creates a tree; not the usual top-down construction 
    but rather a natively parallel one that ensures good load balancing during
    construction.
    '''

    # Create a 1d array of 32-bit ints with length len(points)
    leaves = Array(int(32), 1)(len(points))
    # Fill leaves from points in parallel
    for i in parrange(len(points)):
        leaves[i] = point_to_morton_index(point[i])

    # Sort the array in place (hypothetical library function)
    sort_array(leaves)

    # Create a 2d array of 32-bit ints with dims (len(leaves), maxDepth)
    ancestors = Array(int(32), 2)(len(leaves), maxDepth)
    # Fill the array with 0s.
    ancestors.fill(0)

    # Compute ancesters for each point at least depth
    for i in parrange(len(leaves)):
        for j in parrange(maxDepth):
            ancestors[i, j] = find_ancestors(leaves[i], j)

    # ???
    allnodes = sort_unique([leaves, ancestors])

    # ???
    tree = Tree(allnodes, points)

    return tree

# ====================================================

def evaluate(tree, density):
    '''
    N-body evaluation, the potential is array with same length as points
    '''

    def near_field():
        '''
        Compute near-field potentials
        '''
        # Create a 1d array of potentials, same size as leaves
        potential_nf = Array(???, 1)(len(tree.leaves))
        # Process all leaves in parallel
        for i in parrange(len(tree.leaves)):
            leaf = tree.leaves(i)

            # Reduce the interactions with each ??? leaf into a single potential
            potential_nf[leaf.index] = reduce(
                pariter(leaf.ulist),
                lambda other: direct_interaction(leaf, other),
                lambda a, b: a + b)
        # Return the resulting per leaf near-field protentials
        return potential_nf

    # perhaps with inline functions no need for stages
    @variant(GPU, FPGA)  
    def upward():
        for node in pariter(tree.nodes):
            node.d = reduce(
                pariter(node.children),
                lambda child: U[node.level] * child.d,
                sum)

    @variant(CPU)
    def upward():
        # Group the nodes by their level but return list(Array(Index)) of the indicies. (hypothetical library function)
        nodes_at_level = group_indicies_by(tree.nodes, lambda e: e.level, groups = range(tree.depth))

        # TODO: (amp) George implied that the outer loop could be parallel, but I don't understand how.
        # By-level traversal of the node tree
        for level in reversed(range(tree.depth)):
            for i in parrange(len(nodes_at_level[level])):
                currentnode = tree.node[i]
                def compute(child):
                    # This reads child.d, but child cannot be represented in terms of the the outer loop indicies without pointers or data dependent indexing.
                    # So Parla has no way to express it. This means the outer loop has to be sequential.
                    # with reads(child.d): # PIPELINING
                    return U[level] * child.d
                # Compute d for each node based on it's children
                currentnode.d = reduce(
                    pariter(currentnode.children),
                    compute,
                    sum)

    @variant(GPU)
    def far_field():
        potential_ff = Array(???, 1)(len(tree.leaves))
        for i in parrange(len(tree.leaves)):
            leaf = tree.leaves(i)
            def compute(other):
                # TODO: This reads other.d, and could be be pipelines with other functions if that dependancy could be directly expressed.
                # However, the previous write to other.d is in upward and is not performed in a loop with the same structure as this one.
                # with reads(other.d): # PIPELINING
                return direct_interaction(leaf, other)
            potential_ff[n] = reduce(
                pariter(n.vlist),
                compute,
                sum)
        
        
    @variant(CPU)
    def far_field():
        potential_ff = Array(???, 1)(len(tree.leaves))
        Aind = Array(Size, 1)(???)
        A = Array(???, 3)(NUM_DIRECTIONS, ???, ???).memory(FAST)

        k = 0

        for leaf in pariter(tree.leaves):
            for j in parrange(len(leaf.vlist)):
                other = leaf.vlist[j]
                d = direction(other)
                # with reads(other.d): # PIPELINING
                A[d,:,k] = permute(j.d)
                k += 1
                Aind[d] = update(Aind[d], k, leaf, other)
        # Loops end with an implicit barrier for their iterations
        
        for d in parrange(NUM_DIRECTIONS):
            C = G[d]*A[d]
            
            def compute(i):
                cn = Aind[d].n(i)
                return unpermute(C[:,cn],Aind[d],i)

            potential_ff[Aind[d].n] = reduce(
                parrange(len(Aind[d])),
                compute,
                sum)
        return potential_ff

    # For pipelined parallelism:
    # (To use this all the lines with "PIPELINING" would need to be uncommented and supported)
    # with sync():
    #     with task():
    #         potential_nf = near_field()
    #     with task():
    #         upward()
    #     with task():
    #         potential_ff = far_field()

    # Parallel phases without pipelining
    potential_nf = near_field()
    upward()
    potential_ff = far_field()

    return potential_nf + potential_ff


                
                
                            

            
            
        
                

                    

                    
                
        
        
        

        


                

        
            
            
        
        
        
        
    
