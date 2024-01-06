using Graphs
using Printf
using CSV
using DataFrames
using SpecialFunctions
using GraphPlot, Graphs
using Compose, Cairo
using Random

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end


# Ref Mykel's Textbook Pg. 124
function generate_random_neighbor(G,n,max_parents = 2)
 while true
    i = rand(1:n)
    j = rand(1:n)
    while(i==j)
        j = rand(1:n)
    end
 
    G_neighbor = copy(G)

    if(has_edge(G,i,j))
        rem_edge!(G_neighbor,i,j)        
    else
        add_edge!(G_neighbor,i,j)
    end   

    if(length(inneighbors(G_neighbor,j))<=max_parents)
        return G_neighbor
    end
    
end
    return G_neighbor

end

function Find_Optimal_Graph(max_iters, df,annealing_thresh = 0.05)
    eps = 0.001
    Data_Matrix = Matrix(df) 

    n = length(names(df))
    G_opt = DiGraph(n)

    #M,alpha,n,q,r = compute_m_alpha(Data_Matrix,G_Opt)
    score_prev = compute_bayesian_score(Data_Matrix,G_opt)
    
    for i = 1:max_iters

        G_next = generate_random_neighbor(G_opt,n)

        if(is_cyclic(G_next))
        continue
        end
        score_next = compute_bayesian_score(Data_Matrix,G_next)
        score_now = compute_bayesian_score(Data_Matrix,G_opt)
        Energy = 1/(1+exp(-score_next/score_now))
        rn  = rand()
        if(Energy>rn)
            #println("Score after i = ", i, compute_bayesian_score(Data_Matrix,G_next))
            G_opt =  G_next
            if(abs(score_next - score_prev) < eps)
                break
            end 

        #Simulated Annealing Part    
        # We at random, choose to go with a worse score to avoid falling to local minimums.

        end

    end

    score = compute_bayesian_score(Data_Matrix,G_opt) 
    println("The Final Bayesian Score is: " , score)
    return G_opt

end

function Plot_And_Write(Graph,df,name)

n = length(names(df))
p = gplot(Graph; nodelabel=names(df), layout=circular_layout)
draw(PNG(name *".png"), p)
idx2names = Dict(i=>names(df)[i] for i=1:n)
write_gph(Graph, idx2names, name * ".gph")

end





function add_counts(M,data_inst,parents,r,i,k,q)
    parent_instantiations = ([1:Integer(r[alpha]) for alpha in parents])
    parent_instantiation_list = collect(reduce(vcat,((Iterators.product(parent_instantiations...)))))
    for (idx,e) in enumerate(parent_instantiation_list)
        if(data_inst[parents] == collect(e))
            M[i][idx,k] +=1
        end    
    end    
   return M
end


function compute_m_alpha(Data_Mat,G)
    n = Integer(size(Data_Mat,2))
    m = size(Data_Mat,1)
    i = 1
    r = zeros(n)
    q = zeros(n)
    Parent_Instantiations = Vector()

    for i = 1:n
        r[i] = Integer(maximum(unique(Data_Mat[:,i])))
    end 

    for i = 1:n
        q[i] = Integer(prod([r[j] for j in inneighbors(G,i)]))
    end 

    M = [zeros(Integer(q[i]),Integer(r[i])) for i in 1:n]
    alpha = [ones(Integer(q[i]),Integer(r[i])) for i in 1:n]
    for i=1:m
        for j = 1:n
            k = Data_Mat[i,j]
            parents = inneighbors(G,j)
            if(length(parents)==0)
            # Only one parental instantiation! 
                M[j][1,k] += 1
            else
                 M = add_counts(M,Data_Mat[i,:],parents,r,j,k,q[j])    
            end    
        end
    end
    return M,alpha,n,q,r
end    

function compute_bayesian_score(Data_Mat,G)
    M,alpha,n,q,r = compute_m_alpha(Data_Mat,G)
    score = 0
    for i = 1:n
        Mi = M[i]  
        Ai = alpha[i]
        for j = 1:q[i]
            j = Integer(j)
            score = score + loggamma(sum(Ai,dims=2)[j,1]) - loggamma(sum(Mi,dims=2)[j,1] + sum(Ai,dims=2)[j,1])

            for k = 1:r[i]
                k = Integer(k)
                score = score + loggamma(Ai[j,k]+Mi[j,k])- loggamma(Ai[j,k])
            end
        end
    end
    return score
end

function compute(infile,name)

    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    max_iters = 1000
    df = DataFrame(CSV.File(infile))
    
    G_opt = Find_Optimal_Graph(max_iters, df)
     
    Plot_And_Write(G_opt,df,name)

    
end



infile = "data/small.csv"
name = "small"
# Small
@time begin
compute(infile,name)
end


# infile = "data/medium.csv"
# name = "medium"
# # Medium
# @time begin
# compute(infile,name)
# end


# infile = "data/large.csv"
# name = "large"
# # Large
# @time begin
# compute(infile,name)
# end


