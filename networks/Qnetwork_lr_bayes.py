from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import linalg as LA
import random


# import tensorflow.contrib.slim as slim
# np.random.seed(1)
# tf.set_random_seed(1)


class Q_network():
    '''Double Duelling Network'''
    _training_params = namedtuple('training_prams',
                                  'global_step batch_size learning_rate temperature epsilon loss meanQ BNNloss dropout_keep')

    def __init__(self, num_of_inputs=28,
                 num_of_hidden_units=(256, 512, 512, 64),
                 num_of_actions=2,
                 e_greedy_max=0.5,
                 e_greedy_min=0.1,
                 learning_rate_max=1e-3,
                 learning_rate_min=1e-4,
                 replace_target_iter=100,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_steps=5e4,
                 temperature_steps=3e4,
                 output_graph=False,
                 dueling=True,
                 sess=None,
                 time_diff=0,
                 beta=0.01,
                 env=None,
                 keep_prop=None,
                 edward_model=None,
                 cpu=0,
                 action_opt=None,
                 model_name_to_restore=None,
                 prioritized=True, reload=False):
        '''
            Double Dueling Network
            Network Structure: Input --> Hidden --> Hidden Output --> Output

            Hidden Output Layer Contains Q = Value (1) + Advantage (2)
            Output Layer --> Two units for action Q-values

            time_diff = for computing gamma, discount
        '''

        # Input and Output Units
        self.n_input = num_of_inputs
        self.n_actions = num_of_actions

        # Learning Parameters
        self.learning_rate_max = learning_rate_max
        self.learning_rate_min = learning_rate_min
        self.lr = None

        self.epsilon_max = e_greedy_max
        self.epsilon_min = e_greedy_min
        self.epsilon = e_greedy_max

        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_steps = e_greedy_steps
        self.sample_space = np.arange(0, self.memory_size)
        self.cost = 0
        self.batch_counter = 0

        self.dueling = dueling  # decide to use dueling network
        self.learning_step_counter = 0
        self.beta = beta
        self.env = env
        self.keep_prob = keep_prop
        self.cpu = cpu
        self.prediction = None

        self.temperature = np.array([2.0])
        self.temp_max_steps = temperature_steps
        self.decay_iter = 0
        self.training_trajectory = []
        self.learning_step = 6e4  # 1 / (np.log(self.learning_rate_max) / self.learning_rate_max)
        self.batch_increase = 6e4

        # Layer parameters
        self.nhidden_layer1 = num_of_hidden_units  # number of hidden layer units
        self.model_name_to_restore = model_name_to_restore
        self.action_opt = action_opt
        self.bnnloss = 0
        self.prioritized = prioritized
        self.dropout_keep_prob = np.array([0.5])

        # Memory size for (s, a, s', r)
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros(shape=(self.memory_size, num_of_inputs * 2 + 3), dtype=np.float32)

        self._build_net()

        # Network parameters for updating target and evaluation networks
        t_params = tf.get_collection(key='target_net_params')
        e_params = tf.get_collection(key='eval_net_params')

        # with tf.get_default_graph().as_default():
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        config = tf.ConfigProto(device_count={"CPU": 8},
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=1)

        if sess is None:
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

        else:
            self.sess = sess

        if output_graph:
            # tf.summary.FileWriter('logs', self.sess.graph)
            # tensorboard --logdir=./graphs

            self.writer = tf.summary.FileWriter('./graphs', self.sess.graph)

        self.lost_hist = []
        self.qmean_hist = []
        self.edward_model = edward_model

        # restore options
        self.reload = reload

        if self.reload:
            EPOCHS = 301
            train_opt = 'dropout'
            model_path = "../models/model_{}_{}/".format(EPOCHS, train_opt, EPOCHS)
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

    def _build_net(self):
        def build_layers(state, c_names, n_l1, w_initializer, b_initializer):
            '''
                state = placeholder, c_names = collection_names
            '''
            nhu_1 = n_l1[0]
            nhu_2 = n_l1[1]
            nhu_3 = n_l1[2]
            nhu_4 = n_l1[3]

            with tf.device('/{}'.format(self.cpu)):
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable(name='w1', shape=[self.n_input, nhu_1],
                                         initializer=w_initializer, collections=c_names)

                    b1 = tf.get_variable(name='b1', shape=[1, nhu_1],
                                         initializer=b_initializer, collections=c_names)

                    # relu_layer1 = tf.nn.softplus(tf.matmul(state, w1) + b1)
                    relu_layer1 = tf.nn.elu(tf.matmul(state, w1) + b1)

                    l1 = tf.nn.dropout(relu_layer1, self.keep_prob)

                with tf.variable_scope('l2'):
                    w1a = tf.get_variable(name='w1a', shape=[nhu_1, nhu_2],
                                          initializer=w_initializer, collections=c_names)

                    b1a = tf.get_variable(name='b1a', shape=[1, nhu_2],
                                          initializer=b_initializer, collections=c_names)

                    relu_layer2 = tf.nn.elu(tf.matmul(l1, w1a) + b1a)

                    l2 = tf.nn.dropout(relu_layer2, self.keep_prob)

                with tf.variable_scope('l3'):
                    w1b = tf.get_variable(name='w1b', shape=[nhu_2, nhu_3],
                                          initializer=w_initializer, collections=c_names)

                    b1b = tf.get_variable(name='b1b', shape=[1, nhu_3],
                                          initializer=b_initializer, collections=c_names)

                    relu_layer3 = tf.nn.elu(tf.matmul(l2, w1b) + b1b)

                    l3 = tf.nn.dropout(relu_layer3, self.keep_prob)

                with tf.variable_scope('l4'):
                    w1c = tf.get_variable(name='w1b', shape=[nhu_3, nhu_4],
                                          initializer=w_initializer, collections=c_names)

                    b1c = tf.get_variable(name='b1b', shape=[1, nhu_4],
                                          initializer=b_initializer, collections=c_names)

                    relu_layer4 = tf.nn.elu(tf.matmul(l3, w1c) + b1c)

                    l4 = tf.nn.dropout(relu_layer4, self.keep_prob)

            if self.dueling:
                ''' Dueling DQN '''

                with tf.device('/{}'.format(self.cpu)):
                    with tf.variable_scope('Value'):
                        w2 = tf.get_variable(name='w2', shape=[nhu_4, 1], initializer=w_initializer,
                                             collections=c_names)
                        b2 = tf.get_variable(name='b2', shape=[1, 1], initializer=b_initializer, collections=c_names)
                        # self.V = tf.nn.relu(tf.matmul(l3, w2) + b2)
                        self.V = tf.matmul(l4, w2) + b2

                    with tf.variable_scope('Advantage'):
                        w2 = tf.get_variable('w2', [nhu_4, self.n_actions], initializer=w_initializer,
                                             collections=c_names)
                        b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                        # self.A = tf.nn.relu(tf.matmul(l3, w2) + b2)
                        self.A = tf.matmul(l4, w2) + b2

                    with tf.variable_scope('Q_values'):
                        # out = tf.nn.relu(tf.add(self.V, (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True)),
                        #              name="out"), name='out_relu')  # Q(s, a) = V(s) + A(s, a)

                        out = tf.add(self.V, (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True)),
                                     name="out")  # Q(s, a) = V(s) + A(s, a)

            else:

                with tf.variable_scope('Q_values'):
                    w2 = tf.get_variable('w2', [nhu_3, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                    out = tf.add(tf.matmul(l3, w2), b2, name="out")
                    # out = tf.nn.relu(tf.matmul(l3, w2)+ b2, name = 'out')

            return out

        # ----------------------------- build_evaluation_net ------------------------------------- #

        # Define input-outputs
        # with tf.variable_scope('input_output'):
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input], name='state')  # input states
        self.q_target = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions])  # output actions q-values
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=(), name='keep_prob_ph')
        # self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        # configuration of layers
        with tf.device('/{}'.format(self.cpu)):
            with tf.variable_scope('eval_net'):
                c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
                n_l1 = self.nhidden_layer1
                # w_initializer = tf.random_normal_initializer(mean=0.0, stddev=1)
                w_initializer = tf.truncated_normal_initializer(mean=0., stddev=0.5)
                b_initializer = tf.random_uniform_initializer(0., 0.5)
                # w_initializer = tf.contrib.layers.xavier_initializer()
                # b_initializer = tf.constant_initializer(0.1)

                self.q_eval = build_layers(self.state, c_names, n_l1, w_initializer, b_initializer)
                self.temp_ph = tf.placeholder(tf.float32, shape=[1], name='temp_ph')
                self.Q_distribution = tf.nn.softmax((self.q_eval / tf.norm(self.q_eval)) / self.temp_ph,
                                                    name='q_distribution')

        # Operations
        with tf.device('/{}'.format(self.cpu)):
            with tf.variable_scope('Loss'):

                if self.prioritized:
                    self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)  # for updating Sumtree
                    training_loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
                    # vars = tf.trainable_variables()
                    # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.005
                    #
                    # self.loss_func = tf.reduce_mean(training_loss + lossL2)
                    self.loss_func = tf.reduce_mean(training_loss)

                else:
                    # training_loss = tf.losses.huber_loss(self.q_target, self.q_eval, delta=2)
                    training_loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
                    # vars = tf.trainable_variables()
                    # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.005

                    # self.loss_func = tf.reduce_mean(training_loss + lossL2)
                    self.loss_func = tf.reduce_mean(training_loss)

            with tf.variable_scope('Train'):
                # self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
                # self._train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

                self.learning_rate = tf.train.exponential_decay(self.learning_rate_max, self.global_step,
                                                                decay_steps=self.learning_step, decay_rate=0.96,
                                                                staircase=True)

                self._train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_func,
                                                                                                   global_step=self.global_step)

        with tf.variable_scope('Summaries'):
            tf.summary.scalar('loss', self.loss_func)
            tf.summary.scalar('learning_rate', self.learning_rate)
            self.summary_op = tf.summary.merge_all()

            # ----------------------------- build_target_net ------------------------------------- #

        self.next_state = tf.placeholder(tf.float32, [None, self.n_input], name='next_state')

        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.next_state, c_names, n_l1, w_initializer, b_initializer)

            # ----------------------------- Helper Functions ------------------------------------- #

    def store_transitions(self, experience):
        s, a, r, tdiff, s_ = experience

        if self.prioritized:  # prioritized replay
            transition = np.hstack((s, a, r, tdiff, s_))
            self.memory.store(transition)  # have high priority for newly arrived transition
        else:  # random replay

            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0

            if self.batch_counter >= self.batch_size:
                self.batch_counter = 0

            transition = np.hstack((s, a, r, tdiff, s_)).astype(np.float32)
            index = self.memory_counter % self.memory_size
            # index = self.sampled_index

            self.memory[index, :] = transition
            self.memory_counter += 1

    def exponential_decay(self, step, total_step, initial, final, rate=1e-4, stairs=None):

        if stairs is not None:
            step = stairs * tf.floor(step / stairs)

        scale, offset = 1. / (1. - rate), 1. - (1. / (1. - rate))
        progress = np.array([step]).astype(np.float32) / total_step,
        value = np.array([(initial - final) * scale * rate]) ** progress + offset + final
        lower, upper = np.min([initial, final]), np.max([initial, final])
        return np.array([np.max([lower, np.min([value[0], upper])])])

    def choose_action(self, observation):

        option = self.action_opt
        observation = observation[np.newaxis, :]
        action = None
        action_ind = None
        actions_values = None

        if option == 'egreedy':
            if np.random.rand(1) > self.epsilon:

                # actions_values = self.sess.run(self.q_eval, feed_dict={self.state: observation})
                actions_values = self.sess.run(self.q_eval,
                                               feed_dict={self.state: observation,
                                                          self.keep_prob_ph: 1.0})

                action_ind = np.argmin(actions_values)
                action = self.env.current_cart.get_action(action_ind)

            else:

                actions_values = self.sess.run(self.q_eval, feed_dict={self.state: observation})
                action, action_ind = self.env.current_cart.sampleAction()

            self.epsilon = self.exponential_decay(self.learning_step_counter, total_step=self.epsilon_steps,
                                                  initial=self.epsilon_max, final=self.epsilon_min, rate=1e-2,
                                                  stairs=None)[0]

            actions_values = actions_values[0]

        if option == 'boltzman':
            ''' 
                Since we minimize q-learning the action should be chosen with min probability, not max.
                so the action probabilities should be inverted p(a1) = 1 - p(a2)
                
            '''
            self.temperature = self.exponential_decay(self.learning_step_counter, total_step=self.temp_max_steps,
                                                      initial=2.0, final=1.0, rate=1e-2, stairs=None)

            action_probs, actions_values = self.sess.run([self.Q_distribution, self.q_eval],
                                                         feed_dict={self.state: observation,
                                                                    self.temp_ph: self.temperature})
            actions_values = actions_values[0]

            action_probs = 1.0 - action_probs  # revert probabilities so that the min action can be chosen with high prob

            action_chosen = np.random.choice(action_probs[0], p=action_probs[0])
            action_ind = np.argmax(action_probs[0] == action_chosen)
            action = self.env.current_cart.get_action(action_ind)

        if option == 'dropout':
            self.dropout_keep_prob = 1.0 - self.exponential_decay(self.learning_step_counter,
                                                                  total_step=self.epsilon_steps,
                                                                  initial=0.5, final=0.1, rate=1e-2, stairs=None)

            actions_values = self.sess.run(self.q_eval,
                                           feed_dict={self.state: observation,
                                                      self.keep_prob_ph: self.dropout_keep_prob[0]})
            actions_values = actions_values[0]
            # averages = np.zeros((5, 2))
            # for i in range(5):
            #     actions_values = self.sess.run(self.q_eval,
            #                                    feed_dict={self.state: observation,
            #                                               self.keep_prob_ph: self.dropout_keep_prob[0]})
            #
            #     averages[i,:] = actions_values
            #
            # actions_values = averages.mean(axis=0)

            action_ind = np.argmin(actions_values)

            action = self.env.current_cart.get_action(action_ind)

        if option == 'bayesian':
            network_input0 = np.append(observation, [0])[np.newaxis, :]
            network_input1 = np.append(observation, [1])[np.newaxis, :]
            R0 = self.edward_model.predict(network_input0)[:, np.newaxis]
            R1 = self.edward_model.predict(network_input1)[:, np.newaxis]

            actions_values = np.array([np.append(R0, R1)])

            action_ind = np.argmin(actions_values)
            action = self.env.current_cart.get_action(action_ind)

        return actions_values[action_ind], action, action_ind

    def predict(self, observation):
        observation = observation[np.newaxis, :]
        with tf.device('/{}'.format(self.cpu)):
            action_probs, actions_values = self.sess.run([self.Q_distribution, self.q_eval],
                                                         feed_dict={self.state: observation,
                                                                    self.temp_ph: np.array([1]),
                                                                    self.keep_prob_ph: 1})

            action_probs = 1.0 - action_probs
            action_ind = np.argmax(action_probs)
            # action_ind = np.argmin(actions_values)
            action = self.env.current_cart.get_action(action_ind)

        return actions_values[0][action_ind], action, action_ind

    def learn(self):
        with tf.device('/{}'.format(self.cpu)):
            if self.learning_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.replace_target_op)
                print('\n', 'target_params_replaced')

            if self.prioritized:
                tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)

            else:

                sample_space = np.arange(0, self.memory_size)
                self.sampled_index = np.random.choice(sample_space, size=self.batch_size)

                batch_memory = self.memory[self.sampled_index, :]

            # q-values of the next stated

            if self._train_op == 'dropout1':
                q_next = np.zeros(( self.batch_size, 2))
                for h in range(4):
                    q_next += self.sess.run(self.q_next, feed_dict={self.next_state:
                                                                       batch_memory[:, -self.n_input:],
                                                                   self.keep_prob_ph: 0.8})

                q_next = q_next / 4

            else:

                q_next = self.sess.run(self.q_next, feed_dict={self.next_state:
                                                                    batch_memory[:, -self.n_input:]})

            q_eval = self.sess.run(self.q_eval, {self.state: batch_memory[:, :self.n_input]})

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_input].astype(int)  # n_input column is chosen actions

            reward = batch_memory[:, self.n_input + 1]
            # reward = reward / 635456

            # Compute gamma
            tdiff = batch_memory[:, -self.n_input - 1] / 60  # change time unit to the minutes
            self.gamma = np.exp(-self.beta * tdiff)

            q_next_min = np.min(q_next, axis=1)
            q_target[batch_index, eval_act_index] = reward + self.gamma * q_next_min  # chosen actions updated

            # # standardize q_target
            # for i in range(self.batch_size):
            #     mean_qval, std_qval = self.env._func_procs['q_vals'].send(q_target[batch_index, eval_act_index][i])
            #
            # if std_qval !=0:
            #     q_target[batch_index, eval_act_index] = (q_target[batch_index, eval_act_index] - mean_qval)/std_qval
            #
            #
            # q_target[batch_index, eval_act_index] = self.clip(q_target[batch_index, eval_act_index])

            if self.prioritized:

                _, batch_loss, learning_rate, abs_errors = self.sess.run([self._train_op, self.loss_func,
                                                                          self.learning_rate, self.abs_errors],
                                                                         feed_dict={
                                                                             self.state: batch_memory[:, :self.n_input],
                                                                             self.q_target: q_target,
                                                                             self.keep_prob_ph: self.keep_prob,
                                                                             self.ISWeights: ISWeights})

                self.memory.batch_update(tree_idx, abs_errors)  # update priority

            else:
                _, batch_loss, learning_rate = self.sess.run([self._train_op, self.loss_func, self.learning_rate],
                                                             feed_dict={self.state: batch_memory[:, :self.n_input],
                                                                        self.q_target: q_target,
                                                                        self.keep_prob_ph: self.keep_prob})
            self.lr = learning_rate
            self.lost_val = batch_loss  # / total_batch

            self.lost_hist.append(self.lost_val)
            self.qmean_hist.append(q_target[batch_index, eval_act_index].mean())

            # self.memory = np.delete(self.memory, self.sampled_index, axis=0)
            # self.memory = np.append(np.zeros((self.batch_size, self.memory.shape[1])), self.memory, axis=0)

            self.learning_step_counter += 1

            if self.learning_step_counter % self.batch_increase == 0 and self.batch_size <= 256:
                self.batch_size = int(self.batch_size * 1.2)

            temp = self._training_params(global_step='{:6.0f}'.format(self.learning_step_counter),
                                         batch_size=self.batch_size,
                                         learning_rate='{:.3E}'.format(learning_rate),
                                         temperature='{:5.5f}'.format(self.temperature[0]),
                                         epsilon=self.epsilon,
                                         loss='{:2.2f}'.format(batch_loss),
                                         meanQ='{:4.2f}'.format(self.qmean_hist[-1]),
                                         BNNloss=self.bnnloss,
                                         dropout_keep=self.dropout_keep_prob[0])

            self.training_trajectory.append(temp)

            print('Step: {} - Loss: {}'.format(self.learning_step_counter, self.lost_val))
            # Tensorboard summary writer
            # write log
            # self.writer.add_summary(summary, self.learning_step_counter)

    def print_report(self, epoch=200):
        training_trajectory_pd = pd.DataFrame(self.training_trajectory, columns=self._training_params._fields)
        training_trajectory_pd.to_csv('../tests/logs/training_trajectory_{}_{}.csv'.format(epoch, self.action_opt),
                                      sep='\t')

    def restore_from(self):
        saver = tf.train.import_meta_graph(self.model_name_to_restore + '.meta')
        saver.restore(self.sess, self.model_name_to_restore)

    def train_BNN(self, x_batch, y_batch):
        bnnloss = []
        for j in range(y_batch.shape[0] - 1):
            x = x_batch[j, :][np.newaxis, :]
            y = np.array([y_batch[j]])[np.newaxis, :] / 200

            self.edward_model.update_belief(x, y)
            bnnloss.append(self.edward_model.info_dict['loss'])

        self.bnnloss = sum(bnnloss) / self.batch_size

    def clip(self, val, min=-10.0, max=10.0):

        val[val < min] = min
        val[val > max] = max

        return val


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Store the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
