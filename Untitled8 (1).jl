
using LinearAlgebra, Plots
using Statistics
using Interpolations, NLsolve, Optim, Random
using QuantEcon
using Distributions

#Parameter Values
β = 0.9
a_bar = 1
γ = 0.2
τ = 1.1
N = 10
ρ = 0.5
σ = 1
ξ = 1.1
f_0 = 5
f_1 = 4
n_std = 1
T = 10000
ϵ = 4
v=0.2

#Discretize fixed cost probability function:
f_0_max = v*f_0
f_1_max = v*f_1
# Generate n possible values of f for the two cases
n=5
f0_grid = range(0, stop=f_0_max, step=(f_0_max/(n-1)))
f1_grid = range(0, stop=f_1_max, step=(f_1_max/(n-1)))
#Obtain probabilities:
prob_0 = zeros(n)
prob_1 = zeros(n)
for i=1:n-1
    
            prob_0[i] = (f0_grid[i+1]/f_0_max)^(1/(1 - v))-(f0_grid[i]/f_0_max)^(1/(1 - v)) #prob fction under f0
            prob_1[i] = (f1_grid[i+1]/f_1_max)^(1/(1 - v))-(f1_grid[i]/f_1_max)^(1/(1 - v)) #prob fction under f1
        
end
h0=zeros(length(f0_grid))
h1=zeros(length(f1_grid))

for i=1:length(f0_grid) h0[i]=f0_grid[i] end
for i=1:length(f1_grid) h1[i]=f1_grid[i] end

f_grid=[0;h0[1:length(prob_1)-1];h1[1:length(prob_1)-1]]
f_prob = [1;prob_0[1:length(prob_0)-1];prob_1[1:length(prob_1)-1]] #probabilities all together (i.e. dont sum up to 1)


#Discretize AR(1) process:
a = QuantEcon.tauchen(N,ρ,σ,a_bar,n_std)
#Obtain T observations of productivity
a_t = simulate(a, T)
p_z_i = stationary_distributions(a)
p_z= hcat(p_z_i...)' #probabilities
z = unique(a_t) #states



## function that does the VFI

function solve(

                # grid returned by the create_grid function

                z,

                # model parameters

                β, ϵ, ξ, τ,

                # VFI tolerance (set to be tiny when running)

                tolerance=10^(-6),

                # max VFI iterations (to prevent the loop from running forever if something is broken), set to a large integer

                iterations_max=1000

            )





    n = length(z)


    ## prepare objects to store the value and policy functions

    # initial V_0 guess

    V = zeros(Float64, (length(z), length(f_grid)))
    
    # empty vector to store the policy function in:

    policy_m = zeros(Float64, (length(z), length(f_grid)))


    # initialize variables to keep tracking of iterations

    iteration_counter = 0

    distance = 10.0^6



    # iterate!

    while distance > tolerance && iteration_counter < iterations_max

        # increase iteration counter by 1

        iteration_counter = iteration_counter + 1

        
        # initialize an empty V_prime

        
        V_prime = zeros(Float64, (length(z), length(f_grid)))


        # iterate over productivity (i.e. over rows)

        for i=1:length(z)
            
            
            # iterate over f (columns):
            
            for j=1:length(f_grid)
                
                
            # initialize a vector to hold values of the objective function at m={0,1}

            rhs_value = zeros(Float64, (length(z), 1+length(f0_grid)+length(f1_grid)))
            
                # compute price at a=a_i, {m-1,m}={m-1_j,m_k} and profits            
                
                if j<=1 #non-exporter
                    
                p = (1/z[i])*(ϵ/(ϵ-1))
                π = (p^(1-ϵ)-p^(-ϵ))/z[i]
                        
                        else #exporter
                        
                p = (1/z[i])*(ϵ/(ϵ-1))*((1+ξ^(-ϵ)*τ^(-ϵ))/(1+ξ^(1-ϵ)*τ^(-ϵ)))
                π = p^(1-ϵ)+p^(1-ϵ)*ξ^(1-ϵ)*τ^(-ϵ)-(p^(-ϵ)+((p*ξ)^(-ϵ)*τ^(-ϵ))/z[i]) 
            
                end
                    
              
                # Calculate expected value of not exporting at t+1:
                        
                E_0 = sum((p_z').*V[:,1])
                
                        
                            if j<=1 #non-exporter at t
                                
                                E_1 = sum(((p_z').*V[:,2:length(prob_0)]).*(prob_0[1:length(prob_0)-1]'))
                                
                                if E_0 > E_1
                                    
                                rhs_value[i,j] = π - f_grid[j] + β*E_0 #decides not to export at t+1
                                
                                policy_m[i,j] = 0
                    
                                
                                else
                                    
                                    rhs_value[i,j] = π - f_grid[j] + β*E_1 #decides to export at t+1
                                    
                                    policy_m[i,j] = 1
                                    
                                end
                
              
                            else #exporter at t
                                
                        
                                E_1 = sum(((p_z').*V[:,length(prob_0):length(f_grid)]).*(prob_1'))
                                
                                
                                if E_0 > E_1
                                    
                                rhs_value[i,j] = π - f_grid[j] + β*E_0 #decides not to export at t+1
                                
                                policy_m[i,j] = 0
                                    
                                else
                                    
                                rhs_value[i,j] = π - f_grid[j] + β*E_1 #decides to export at t+1
                                
                                policy_m[i,j] = 1
                                    
                                end
                                
                            
                                
                                end                
            
                            


            # update v_prime at i,j with this value

            V_prime[i,j] = rhs_value[i,j]
                    
                    
                    
                        end



        end



        # measure the distance between V_prime and V

        distance = maximum(abs.(V_prime.-V))



        # update the VF guess. If we just do V=V_prime, Julia will simply point both arrays to the same object in memory, and so when start overwriting V_prime on next iteration, V will be overwritten with it. We don't want that, so actually create a full copy of V_prime, to disconnect the two objects from each other in memory

        V = deepcopy(V_prime)



        if distance > tolerance

            printstyled("Iteration #", iteration_counter, ". Distance: ", distance, " > ", tolerance, ". Keep going!\n", color=:red)

        else

            printstyled(". Iteration #", iteration_counter, ". Distance: ", distance, " <= ", tolerance, ". Ok!\n", color=:green)

        end

    end



    # return grid, policy, and VF (this is a matrix)

    return V, policy_m, z



end





#tobe completed
# run the solve() function, pass the grid you've created and all other parameters

VFI = solve(

                # grid returned by the create_grid function

                k_grid,

                # model parameters

                0.98, 4, 1.2, 1.1,

                # VFI tolerance (set to be tiny when running)

                10^(-6),

                # max VFI iterations (to prevent the loop from running forever if something is broken), set to a large integer

                1000

            )





## plot the results

#plot(VFI[3],VFI[1])

#plot(VFI[3],VFI[2])



