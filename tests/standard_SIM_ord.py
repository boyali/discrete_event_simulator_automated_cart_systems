
import datetime
import logging

import numpy as np
import pandas as pd

from env import environmentOrd2

end_time = 14 * 3600  # 5 hours of simulation
operation_start_time = datetime.time(7, 0, 0)  # in the morning h:m:s

BETA = 0.01
env = environmentOrd2.Env(num_of_carts=2, end_time=end_time, beta=BETA)

observations = env.reset()
done = False

env.step()

avg_mean = []

N = 1

formatter = logging.Formatter('%(name)s:  %(message)s')
file_handler_sim = logging.FileHandler('../tests/logs/log_std_sim_N_Ord_{}.txt'.format(N))
open('../tests/logs/log_std_sim_N_Ord_{}.txt'.format(N), 'w').close()
file_handler_sim.setFormatter(formatter)
file_handler_sim.flush()

open('../tests/logs/log_std_sim_N_Ord_{}.txt'.format(N), 'w').close()
logger_sim = logging.getLogger('../tests/logs/log_std_sim_N_Ord_{}.txt'.format(N))
logger_sim.setLevel(logging.INFO)
logger_sim.addHandler(file_handler_sim)

avg_accupancy = np.zeros(shape=(N, env.num_of_carts))
dist_travelled = np.zeros(shape=(N, env.num_of_carts))
ring_counters = np.zeros(shape=(N, env.num_of_carts))
charge_used = np.zeros(shape=(N, env.num_of_carts))

frames_csate_reward = []

frames_passengers = []
frames_sars = []
frames_agents = []

for i in range(N):

    _ = env.reset()
    done = False

    env.step()
    while not done:
        try:
            action, _ = env.current_cart.stdAction()  # 502
            # action, _ = env.current_cart.sampleAction()     # 503
            s_, r_, done = env.step(action)

        except done:
            env.close()

    print(env._pass_mean_time)
    avg_mean.append(env._pass_mean_time)

    for (j, cart) in env.carts.items():
        print(cart.dt_avg_occupancy)
        avg_accupancy[i, j] = cart.dt_avg_occupancy
        dist_travelled[i, j] = cart._total_distance
        ring_counters[i, j] = next(cart._cart_ring_counter) - 1
        charge_used[i, j] = cart._charge_used_cumsum

        frames_csate_reward.append(pd.DataFrame(env._cstate_reward_dict[env.carts[j].cartID],
                                                columns=env.cstate_reward_tuple._fields))

        frames_passengers.append(pd.DataFrame(cart.stats_list, columns=env.carts[j]._stats_tuple._fields))

        frames_sars.append(env.sars_history_constrained[env.carts[j].cartID])
        frames_agents.append(pd.DataFrame(env.csars_agent_all[env.carts[j].cartID],
                                          columns=env.csars_tuple._fields))

env.close_funcs()

print('Average mean over {} simulations is :{}'.format(N, sum(avg_mean) / N))
print('Average occupancy over {} simulations is :{}'.format(N, avg_accupancy.mean(axis=0)))
print('Distance travelled {} simulations is :{}'.format(N, dist_travelled.mean(axis=0)))
print('Total charge used {} simulations is :{}'.format(N, charge_used.mean(axis=0)))

print('Number of rings {}'.format(ring_counters.mean(axis=0)))

logger_sim.info('Average mean over {} simulations is :{}'.format(N, sum(avg_mean) / N))
logger_sim.info('Average occupancy over {} simulations is :{}'.format(N, avg_accupancy.mean(axis=0)))
logger_sim.info('Distance travelled {} simulations is :{}'.format(N, dist_travelled.mean(axis=0)))
logger_sim.info('Total charge used {} simulations is :{}'.format(N, charge_used.mean(axis=0)))

cstate_reward_pd = pd.concat(frames_csate_reward)
passenger_waiting_pd = pd.concat(frames_passengers)
sars_pd = pd.concat(frames_sars)
agents_pd = pd.concat(frames_agents)

cstate_reward_pd.to_csv('../tests/logs/cars_cstate_reward_pd_collected_{}_ord.csv'.format(N), sep="\t")
passenger_waiting_pd.to_csv('../tests/logs/passenger_waiting_pd_collected_{}_ord.csv'.format(N), sep="\t")
sars_pd.to_csv('../tests/logs/sars_collected_{}_ord.csv'.format(N), sep="\t")
agents_pd.to_csv('../tests/logs/cars_agent_stats_pd_collected_{}_ord.csv'.format(N), sep="\t")
