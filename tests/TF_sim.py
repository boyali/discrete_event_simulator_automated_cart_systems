import datetime
import gc
import logging
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf

# from numpy import linalg as LA
from env import environment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

train_opt_dict = {0: 'egreedy', 1: 'boltzman', 2: 'dropout', 3: 'bayesian'}

train_opt = train_opt_dict[2]
cpu = 'cpu:0'
EPOCHS = 181
N = 51

formatter = logging.Formatter('%(name)s:  %(message)s')
file_handler_sim = logging.FileHandler('../tests/logs/log_sim_N_{}_{}.txt'.format(N, train_opt))
open('../tests/logs/log_sim_N_{}_{}.txt'.format(N, train_opt), 'w').close()
file_handler_sim.setFormatter(formatter)
file_handler_sim.flush()

logger_sim = logging.getLogger('../tests/logs/log_sim_N_{}_{}.txt'.format(N, train_opt))
logger_sim.setLevel(logging.INFO)
logger_sim.addHandler(file_handler_sim)

q_minmax = namedtuple('q_values', 'q_min, q_max, action_ind, accumulated_cost, action_taken event')
minQ_tuple = namedtuple('minQ_ed', 'state action_ind minQ')

BETA = 0.01
"""
    Beta is the dicount factor, 
        0.01 corresponds to 0.54 for per minute
        0.001 corresponds to 0.94
        0.0017 corrsponds t0 0.90
"""

end_time = 14 * 3600  # 5 hours of simulation
operation_start_time = datetime.time(7, 0, 0)  # in the morning h:m:s

env = environment.Env(num_of_carts=2, end_time=end_time, beta=BETA)

# ## Restore Model
# Remove the previous weights and bias
tf.reset_default_graph()

sess_sim = tf.Session()

# read checkpoint
model_path = "../models/model_{}_{}/".format(EPOCHS, train_opt, EPOCHS)
with open(model_path + 'checkpoint', 'r') as f:
    first_line = f.readline()

checkpoint = 160
model_name = "../models/model_{}_{}/trained_{}.ckpt-{}".format(EPOCHS, train_opt, EPOCHS, checkpoint)

meta_name = first_line.split(':')[1].strip().split('"')[1::2][0]

# saver = tf.train.import_meta_graph(model_name + '.meta')
saver = tf.train.import_meta_graph(model_path + meta_name+'.meta')
saver.restore(sess_sim, tf.train.latest_checkpoint(model_path))
# saver.restore(sess_sim, model_name)

# all_vars = tf.trainable_variables()
# for v in all_vars:
#     print(v.name)

# variables_names = [v.name for v in tf.trainable_variables() if v.name == 'dueling/target_net/Advantage/b2:0']
# values = sess.run(variables_names)
# print(values)

# for k,v in zip(variables_names, values):
#     print(k, v)


graph = sess_sim.graph
predict_op = graph.get_tensor_by_name('dueling/eval_net/Q_values/out:0')
# predict_op = graph.get_operation_by_name('dueling/eval_net/Q_values/out')
state_ph = graph.get_tensor_by_name('dueling/state:0')
keep_ph = graph.get_tensor_by_name('dueling/keep_prob_ph:0')
q_distribution = graph.get_tensor_by_name('dueling/eval_net/q_distribution:0')
temp_ph = graph.get_tensor_by_name('dueling/eval_net/temp_ph:0')
saver.restore(sess_sim, tf.train.latest_checkpoint(model_path))


def predict(observation):
    observation = observation[np.newaxis, :]
    with tf.device('/{}'.format(cpu)):
        actions_values = sess_sim.run(predict_op, feed_dict={state_ph: observation, keep_ph: 1.0})
        action_ind = np.argmin(actions_values)
        action = env.current_cart.get_action(action_ind)

    return actions_values[0], action, action_ind


def simulate_net():
    minQ_list = []
    q_minmax_list = []
    acc_r = [0]

    observation = env.reset()
    _, _, done = env.step()

    # just to avoid pycharm warnings
    action_ind = []
    waiting_cost = None
    action = None
    ta = 0

    while not done:

        observation = env.get_state
        minQ, action, action_ind = predict(observation)
        # action_values, action, action_ind = predict_sim(predict_op, [state_placeholder, keep_ph], observation)
        ta = env._sim_time

        try:
            observation_, waiting_cost, done = env.step(action)
            # print(waiting_cost)
            # a = 1

        except done:
            env.close()

        ta_prime = env._sim_time

        acc_r.append(waiting_cost + acc_r[-1])  # accumulated waiting_cost
        q_minmax_list.append(q_minmax(q_min='{:4.2f}'.format(np.min(minQ)),
                                      q_max='{:4.2f}'.format(np.max(minQ)),
                                      action_taken=action,
                                      action_ind=action_ind,
                                      accumulated_cost=acc_r[-1],
                                      event=env.current_event.event_type))
        minQ_list.append(
            minQ_tuple(state=np.append(observation, action_ind)[np.newaxis, :], action_ind=action_ind, minQ=minQ))

        if len(observation_) > 0:
            observation = observation_

    qvalues_pd = pd.DataFrame(q_minmax_list, columns=q_minmax._fields)
    qvalues_pd.to_csv('../tests/logs/simulation_q_values.csv', sep='\t')
    minQ_pd = pd.DataFrame(minQ_list, columns=minQ_tuple._fields)
    return minQ_pd, acc_r


avg_mean = []

avg_accupancy = np.zeros(shape=(N, env.num_of_carts))
dist_travelled = np.zeros(shape=(N, env.num_of_carts))
ring_counters = np.zeros(shape=(N, env.num_of_carts))
charge_used = np.zeros(shape=(N, env.num_of_carts))

frames_csate_reward = []
frames_passengers = []
frames_minQ = []
frames_sars = []
frames_agents = []

for i in range(N):

    minQ_pd, cumulative_cost = simulate_net()

    print(env._pass_mean_time)
    avg_mean.append(env._pass_mean_time)

    frames_minQ.append(minQ_pd)

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
minQ_training_pd = pd.concat(frames_minQ)
sars_pd = pd.concat(frames_sars)
agents_pd = pd.concat(frames_agents)

cstate_reward_pd.to_csv('../tests/logs/cars_cstate_reward_pd_collected_{}_{}_sim.csv'.format(N, train_opt), sep="\t")
passenger_waiting_pd.to_csv('../tests/logs/passenger_waiting_pd_collected_{}_{}_sim.csv'.format(N, train_opt), sep="\t")

minQ_training_pd.to_csv('../tests/logs/minQ_pd_collected_{}_{}_sim.csv'.format(N, train_opt), sep="\t")
sars_pd.to_csv('../tests/logs/sars_collected_{}_{}_sim.csv'.format(N, train_opt), sep="\t")
agents_pd.to_csv('../tests/logs/cars_agent_stats_pd_collected_{}_{}_sim.csv'.format(N, train_opt), sep="\t")

sess_sim.close()
gc.collect()
