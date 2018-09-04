# George's tree.py (in mock_ups) rewritten using the concepts and
# syntax of Arthur's Parla strawman.

from parla.primitives import *
from parla.array import *
from parla.function_decorators import *
from parla.loops import *

FIX = "???"

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
    leaves = zeros[I[32]](len(points))
    # Fill leaves from points in parallel
    for i in range(len(points)):
        leaves[i] = point_to_morton_index(point[i])

    # Sort the array in place (hypothetical library function)
    sort_array(leaves)

    # Create a 2d array of 32-bit ints with dims (len(leaves), maxDepth)
    ancestors = zeros[I[32]](len(leaves), maxDepth)

    # Compute ancesters for each point at least depth
    for i in range(len(leaves)):
        for j in range(maxDepth):
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
        potential_nf = zeros[FIX](len(tree.leaves))
        # Process all leaves in parallel
        for i in range(len(tree.leaves)):
            leaf = tree.leaves(i)

            # Reduce the interactions with each ??? leaf into a single potential
            potential_nf[leaf.index] = reduce(
                (direct_interaction(leaf, other) for other in iter(leaf.ulist)),
                lambda a, b: a + b)
        # Return the resulting per leaf near-field protentials
        return potential_nf

    # perhaps with inline functions no need for stages
    @device_specialized
    def upward():
        # Group the nodes by their level but return list(Array(Index)) of the indicies. (hypothetical library function)
        nodes_at_level = group_indicies_by(tree.nodes, lambda e: e.level, groups = range(tree.depth))

        # TODO: (amp) George implied that the outer loop could be parallel, but I don't understand how.
        # By-level traversal of the node tree
        for level in reversed(range(tree.depth)):
            for i in range(len(nodes_at_level[level])):
                currentnode = tree.node[i]
                def compute(child):
                    # This reads child.d, but child cannot be represented in terms of the the outer loop indicies without pointers or data dependent indexing.
                    # So Parla has no way to express it. This means the outer loop has to be sequential.
                    # with reads(child.d): # PIPELINING
                    return U[level] * child.d
                # Compute d for each node based on it's children
                currentnode.d = reduce(
                    (compute(child) for child in iter(currentnode.children)),
                    sum)
                
    @upward.variant("GPU", "FPGA")
    def upward():
        for node in iter(tree.nodes):
            node.d = reduce(
                (U[node.level] * child.d for child in iter(node.children)),
                sum)


    def far_field():
        potential_ff = zeros[FIX](len(tree.leaves))
        for i in range(len(tree.leaves)):
            leaf = tree.leaves(i)
            def compute(other):
                # TODO: This reads other.d, and could be be pipelines with other functions if that dependancy could be directly expressed.
                # However, the previous write to other.d is in upward and is not performed in a loop with the same structure as this one.
                # with reads(other.d): # PIPELINING
                return direct_interaction(leaf, other)
            potential_ff[n] = reduce(
                (compute(child) for child in iter(n.vlist)),
                sum)
        
        
    def far_field():
        potential_ff = zeros[FIX](len(tree.leaves))
        Aind = zeros[I[32]](FIX)
        A = zeros[FIX](NUM_DIRECTIONS, FIX, FIX).memory(FAST)

        k = 0

        for leaf in iter(tree.leaves):
            for j in range(len(leaf.vlist)):
                other = leaf.vlist[j]
                d = direction(other)
                # with reads(other.d): # PIPELINING
                A[d,:,k] = permute(j.d)
                k += 1
                Aind[d] = update(Aind[d], k, leaf, other)
        # Loops end with an implicit barrier for their iterations
        
        for d in range(NUM_DIRECTIONS):
            C = G[d]*A[d]
            
            def compute(i):
                cn = Aind[d].n(i)
                return unpermute(C[:,cn],Aind[d],i)

            potential_ff[Aind[d].n] = reduce(
                (compute(child) for child in range(len(Aind[d]))),
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


                
                
                            

            
            
        
                

                    

                    
                
        
        
        

        


                

        
            
            
        
        
        
        
    
