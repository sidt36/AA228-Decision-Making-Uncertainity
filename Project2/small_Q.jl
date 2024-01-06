using Printf
using CSV
using DataFrames
using Random
using StatsBase



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

function rollouts(s,Transition_Model,Reward_Model,states,actions,d = 10,m = 10)
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


function learn_Q(df,states,actions,gamma = 0.95,alpha = 0.5)
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
    open("small.policy", "w") do io
        for p in policy
            @printf(io, "%s\n", p)
        end
    end
end

@time begin
compute_optimal_policy(Q_Learning,df,states,actions)
end