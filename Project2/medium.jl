using Printf
using CSV
using DataFrames
using Random

using StatsBase

infile = "data/medium.csv"
df = DataFrame(CSV.File(infile))

states = []

for i=1:100
    for j=1:500
        push!(states,1+j+500*i)
    end
end
actions = collect(1:7)
 
function learn_Q(df,states,actions,gamma = 1,alpha = 0.5)
    Q = Dict()
    for s in states
        for a in actions
            Q[(s,a)] = 0
        end
    end
    Data_Matrix = Matrix(df)
    n = size(Data_Matrix)[1]

    for i = 1:n

        s = Data_Matrix[i,1]
        a = Data_Matrix[i,2]
        r = Data_Matrix[i,3]
        s_new = Data_Matrix[i,4]
        Q_opt = maximum([Q[(s_new,a1)] for a1 in actions])
        Q[(s,a)] = Q[(s,a)]  + alpha*(r + gamma*Q_opt - Q[(s,a)])

    end
    
    return Q

end    

function Q_Learning(states,actions,df)
    policy = []
    Q = learn_Q(df,states,actions)
    for s in states
        
        Q_opt = maximum([Q[(s,a1)] for a1 in actions])
        a = 0
        for a1 in actions
            if(Q[(s,a1)]==Q_opt)
                a = a1
                break
            end
        end
        push!(policy,a)  
    end
    return policy
end    

function compute_optimal_policy(method,df,states,actions)
    policy = method(states,actions,df)
    #println(Reward_Model)
    open("medium.policy", "w") do io
        for p in policy
            @printf(io, "%s\n", p)
        end
    end
end

@time begin
compute_optimal_policy(Q_Learning,df,states,actions)
end