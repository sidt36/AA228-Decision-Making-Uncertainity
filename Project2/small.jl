using Printf
using CSV
using DataFrames
using Random
infile = "data/small.csv"
df = DataFrame(CSV.File(infile))
using StatsBase

states = LinearIndices((10,10))
actions = [1,2,3,4]

function Learn_Model(df,states,actions)
    
    Transition_Count = Dict()
    Transition_Count_next_state =  Dict()
    Reward_Count= Dict()
    Data_Matrix = Matrix(df)
    n = size(Data_Matrix)[1]
    for s in states
        for a in actions
            Transition_Count[(s,a)] = 0
            Reward_Count[(s,a)] = 0
            for s1 in states
                Transition_Count_next_state[(s,a,s1)] =  0
            end
        end
    end

    for i = 1:n
        s = Data_Matrix[i,1]
        a = Data_Matrix[i,2]
        r = Data_Matrix[i,3]
        s_new = Data_Matrix[i,4]
        Transition_Count[(s,a)] +=1
        Transition_Count_next_state[(s,a,s_new)] +=1
        Reward_Count[(s,a)] += r
    end

    Transition_Model = Dict()
    Reward_Model= Dict()

    for s in states
        for a in actions
            if(Transition_Count[(s,a)] == 0)
                Reward_Model[(s,a)] =  0
            else
                Reward_Model[(s,a)] =  Reward_Count[(s,a)]/Transition_Count[(s,a)]
            end

            for s1 in states
                if(Transition_Count[(s,a)] == 0)
                    Transition_Model[(s,a,s1)] =  0
                else
                    Transition_Model[(s,a,s1)] =  Transition_Count_next_state[(s,a,s1)]/Transition_Count[(s,a)]
                end
            end
        end
    end

    return Transition_Model, Reward_Model
end

function random_action(actions)
    idx = rand(1:length(actions))
    random_action = actions[idx]
end

function random_sample(s,a,states,Transition_Model)
    random_action = a
    New_State_Probab = Dict()
    for s1 in states
        New_State_Probab[s1] = Transition_Model[(s,random_action,s1)]
    end
    s1 = sample([collect(keys(New_State_Probab))...], Weights([collect(values(New_State_Probab))...]), 1)
    return s1[1]
end

function rollouts(s,Transition_Model,Reward_Model,states,actions,d = 10,m = 1000)
    gamma = 0.95
    max_utility = -1000000000
    optimal_action = 0
    optimal_utility_across_simulations = -100000
    optimal_action_across_simulations = 0

    for k = 1:m
        for a in actions
            a1 = a
            s1 = s
            est_utility = Reward_Model[(s1,a1)]

            for i = 1:d
                s1 = random_sample(s1,a1,states,Transition_Model)
                est_utility = est_utility + gamma^i*Reward_Model[(s1,a1)]
                a1 = random_action(actions)
            end

            if(est_utility>max_utility)
                optimal_action = a
                max_utility = est_utility
            end

        end 

        if(max_utility>optimal_utility_across_simulations)
            optimal_action_across_simulations = optimal_action
            optimal_utility_across_simulations = max_utility
        end       
    end

  return optimal_action_across_simulations

end    


function compute_optimal_policy(method,df,states,actions)
    Transition_Model,Reward_Model = Learn_Model(df,states,actions)
    #println(Reward_Model)
    open("small.policy", "w") do io
        for s in states
            policy = method(s,Transition_Model,Reward_Model,states,actions)
            @printf(io, "%s\n", policy)
        end
    end
end

compute_optimal_policy(rollouts,df,states,actions)
