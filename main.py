import numpy as np
import random 
import matplotlib.pyplot as plt

n_steps_per_run = 1000
n_runs = 2000
e1 = 0.1
e2 = 0.01
k = 10
q_star = np.random.normal(loc=0.0, scale=1,size=k)



mean_rew_greedy = [0 for _ in range(n_steps_per_run)]

mean_rew_epsilon1 = [0 for _ in range(n_steps_per_run)]

mean_rew_epsilon2 = [0 for _ in range(n_steps_per_run)]


for i in range(n_runs):
    print(f"run {i}....")
    q_star = np.random.normal(loc=0.0, scale=1,size=k)
    Q_greedy = [0 for _ in range(k)]
    choice_greedy = [0 for _ in range(k)]

    Q_e1 = [0 for _ in range(k)]
    choice_e1 = [0 for _ in range(k)]

    Q_e2 = [0 for _ in range(k)]
    choice_e2 = [0 for _ in range(k)]

    for j in range(n_steps_per_run):
        #greedy
        a = np.argmax(Q_greedy)
        r = np.random.normal(loc=q_star[a],scale=1)
        choice_greedy[a] +=1
        Q_greedy[a] = Q_greedy[a] + 1/(choice_greedy[a]) * (r - Q_greedy[a])
        mean_rew_greedy[j] = mean_rew_greedy[j] + 1/(i+1) * (r - mean_rew_greedy[j])

        if random.random() < e1:
            a = random.randint(0,k-1)
            
        else:
            a = np.argmax(Q_e1)
        r = np.random.normal(loc=q_star[a],scale=1)
        choice_e1[a] +=1
        Q_e1[a] = Q_e1[a] + 1/(choice_e1[a]) * (r - Q_e1[a])
        mean_rew_epsilon1[j] = mean_rew_epsilon1[j] + 1/(i+1) * (r - mean_rew_epsilon1[j])

        if random.random() < e2:
            a2 = random.randint(0,k-1)
        else:
            a2 = np.argmax(Q_e2)

        r = np.random.normal(loc=q_star[a2],scale=1)
        choice_e2[a2] +=1
        Q_e2[a2] = Q_e2[a2] + 1/(choice_e2[a2]) * (r - Q_e2[a2])
        mean_rew_epsilon2[j] = mean_rew_epsilon2[j] + 1/(i+1) * (r - mean_rew_epsilon2[j]) 

plt.plot(mean_rew_greedy,label=f"greedy")
plt.plot(mean_rew_epsilon1,label=f"epsilon={e1}")
plt.plot(mean_rew_epsilon2,label=f"epsilon={e2}")
plt.legend()
plt.show()