import queue
from collections import namedtuple
from itertools import count

import numpy as np
import pandas as pd

from Cart import cartStd as mn
from env import process_time as dt
from stops import stops as stp
from passengers import passengers as psg

# SEED = 1  # 3438  ##
# np.random.seed(SEED)

NUMBER_OF_BUS_STOPS = 8
Event = namedtuple('Event', 'time proc action')


def wrapped_coroutine(func):
    pass

    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)

        next(gen)
        return gen

    return wrapper


class Env(object):
    ''' http://rllab.readthedocs.io/en/latest/user/implement_env.html '''
    sars_tuple = namedtuple('action_reward_state',
                            'sim_time cartID current_state next_state cart_position occupancy carried call_received'
                            ' action reward delta_reward time_since event_type reason')

    csars_tuple = namedtuple('action_reward_state',
                             'sim_time cartID current_state action cart_positions '
                             'reward delta_reward time_since occupancy charge reward_bat event_type reason current_stop')

    cstate_reward_tuple = namedtuple('cstate_reward', 'state action reward_pass state_next charge reward_bat')
    action_state_tuple = namedtuple('action_state', 'state action reward timediff state_next')
    running_mean = namedtuple('running_mean', 'mean std')

    _num_of_envs = count(0)

    def __init__(self, num_of_carts=1, end_time=14 * 3600, beta=0.001, convex_alpha=0.6):

        self._envID = next(self._num_of_envs)

        self._observation_space = None
        # self._seed = SEED

        self._number_of_carts = num_of_carts
        self._beta = beta

        self._carts = {i: mn.Cart(self) for i in range(num_of_carts)}
        self._carts_decision_times = {cart.cartID: [] for (_, cart) in self._carts.items()}
        self._action_counter_dict = {cart.cartID: [] for (_, cart) in self._carts.items()}
        self._const_action_counter_dict = {cart.cartID: [] for (_, cart) in self._carts.items()}
        self._cstate_reward_dict = {cart.cartID: [] for (_, cart) in self._carts.items()}
        self._action_states = {cart.cartID: [] for (_, cart) in self._carts.items()}

        self._action_type_counter = {'wait': 0, 'move': 0, 'stop': 0, 'continue_past_next': 0,
                                     'park': 1, 'launch': 1}

        self._carts_state_of_charge = {cart.cartID: [] for (_, cart) in self._carts.items()}

        self._stops = {i: stp.Stops(self) for i in range(NUMBER_OF_BUS_STOPS)}
        self._stops_vec = np.arange(NUMBER_OF_BUS_STOPS)

        self._events = queue.PriorityQueue()
        self._current_event = None
        self._current_event_type = None
        self._current_cart = self._carts[0]
        self.current_proc_id = None
        self._current_action = None
        self._current_reason = None
        self._pass_mean_time = 0

        # simulation parameters
        self._end_time = end_time
        self._sim_time = 0
        self._sim_daytime = 0
        self._done = False
        self._action_counter = [0] * self.num_of_carts

        # Keep statistics
        self._state = []
        self._reward = 0
        self._delta_reward = 0
        self._convex_alpha = convex_alpha

        self.sars_history_agent = {}
        self.sars_history_constrained = {}
        self.csars_agent_all = {}

        for (key, cart) in self.carts.items():
            self.sars_history_agent[cart.cartID] = []  # pd.DataFrame(columns=sars_tuple._fields)
            self.sars_history_constrained[cart.cartID] = []  # pd.DataFrame(columns=sars_tuple._fields)
            self.csars_agent_all[cart.cartID] = []

        # add function coroutines - in the reset section
        self._func_procs = None
        self._running_means_dict = {'reward': (0.0, 0.0), 'q_vals': (0.0, 0.0)}  # mean, std
        self._func_procs = {'reward': self.running_average(), 'q_vals': self.running_average()}

        for _, proc in sorted(self._func_procs.items()):
            next(proc)

    def _initialize_processes(self):
        for _, proc in sorted(self._procs.items()):
            first_event = next(proc)
            self._events.put(first_event)
            # print(str(first_event) + ' \n ')

    def reset(self):
        ''' Empty queues'''
        while not self._events.empty():
            try:
                self._events.get(False)

            except self._events.queue.Empty:
                self._events.task_done()

        # reset carts
        for i in range(len(self.carts)):
            self.carts[i].reset()

        # reset stops
        for i in range(len(self.stops)):
            self.stops[i].reset()
        # reset processes

        self._procs = {self._stops[i].stopID: self._stops[i].passenger_process()
                       for i in range(NUMBER_OF_BUS_STOPS)}

        for _, cart in self._carts.items():
            '''Append the carts to the proc disctionary'''
            self._procs.update({cart.cartID: cart.cart_process()})

        self._initialize_processes()

        # simulation parameters
        self._end_time = self._end_time
        self._sim_time = 0
        self._sim_daytime = 0
        self._done = False
        self._action_counter = [0] * self.num_of_carts

        # Keep statistics
        self._state = []
        self._reward = 0
        self._delta_reward = 0
        self._pass_mean_time = 0

        self.sars_history_agent = {}
        self.sars_history_constrained = {}

        self.csars_agent_all = {}

        self._carts_decision_times = {cart.cartID: [] for (_, cart) in self._carts.items()}
        self._action_counter_dict = {cart.cartID: [] for (_, cart) in self._carts.items()}
        self._const_action_counter_dict = {cart.cartID: [] for (_, cart) in self._carts.items()}
        self._cstate_reward_dict = {cart.cartID: [] for (_, cart) in self._carts.items()}
        self._action_states = {cart.cartID: [] for (_, cart) in self._carts.items()}
        self._action_type_counter = {'wait': 0, 'move': 0, 'stop': 0, 'continue_past_next': 0,
                                     'park': 1, 'launch': 1}

        self._carts_state_of_charge = {cart.cartID: [] for (_, cart) in self._carts.items()}

        # Env.reset_cls()

        for (key, cart) in self.carts.items():
            self.sars_history_agent[cart.cartID] = []  # pd.DataFrame(columns=sars_tuple._fields)
            self.sars_history_constrained[cart.cartID] = []  # pd.DataFrame(columns=sars_tuple._fields)
            self.csars_agent_all[cart.cartID] = []

        state = self.get_state

        return state

    def passenger_vector(self):
        passengerVec = np.array([len(cart_stops.passengers) for (_, cart_stops) in sorted(self._stops.items())])
        return passengerVec

    def one_hot_loc(self, loc):
        loc_vec = np.zeros(shape=(NUMBER_OF_BUS_STOPS,), dtype=int)

        if len(loc) >= 1:
            for i in range(len(loc)):
                loc_vec[loc[i]] += 1

        return np.array(loc_vec).astype(np.float32)

    # @classmethod
    # def reset_cls(cls):
    #     cls._ARRIVAL_MEANS = np.random.randint(low=0, high=6, size=8)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state = val

    @property
    def other_cart_poss(self):
        other_cart_pos = []
        if len(self.carts) > 1:
            other_cart_pos = [cart.current_stop for (key, cart) in self.carts.items()
                              if self.current_cart.cartID != cart.cartID]

        else:
            other_cart_pos = [0]

        return other_cart_pos

    @property
    def get_state(self):

        other_cart_in_service = [1.0 if cart._in_service is True else 0.0  for (key, cart) in self.carts.items()
                              if self.current_cart.cartID != cart.cartID]
        other_cart_pos = self.other_cart_poss
        passvec = self.passenger_vector()

        if passvec.sum() > 0:
            passvecnorm = passvec / 10

        else:
            passvecnorm = passvec

        time_slot = self._sim_time // 3600

        try:
            # state_val = np.hstack((time_slot / 14, self.current_cart._charge_used_cumsum / 4, passvecnorm,
            #                        self.one_hot_loc([self.current_cart.current_stop]),
            #                        self.current_cart.occupancy / self.current_cart._capacity,
            #                        self.current_cart._cart_inverse_state[self.current_cart.current_state],
            #                        self.one_hot_loc(other_cart_pos)))  # tot_size = 8 + 8 +1 + 8

            state_val = np.hstack((time_slot / 14, self.current_cart._charge_used_cumsum / 4, passvecnorm,
                                   self.one_hot_loc([self.current_cart.current_stop]),
                                   other_cart_in_service[0],
                                   self.current_cart._cart_inverse_state[self.current_cart.current_state],
                                   self.one_hot_loc(other_cart_pos)))  # tot_size = 8 + 8 +1 + 8
        except ValueError:
            state_val = np.array([0])

        return state_val.astype(np.float32)

    def render(self):
        print('current state:', self._state)

    def close(self):
        for _, proc in self._procs.items():
            proc.close()

        # for _, proc in self._func_procs.items():
        #     proc.close()

    def close_funcs(self):
        for _, proc in sorted(self._func_procs.items()):
            proc.close()

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        return self.observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.observation_space = value

    @property
    def stops(self):
        return self._stops

    @property
    def carts(self):
        return self._carts

    @property
    def events(self):
        return self._events

    @property
    def current_event(self):
        return self._current_event

    @current_event.setter
    def current_event(self, current_event):
        self._current_event = current_event

    @property
    def current_cart(self):
        return self._current_cart

    @current_cart.setter
    def current_cart(self, current_cart):
        self._current_cart = current_cart

    @property
    def procs(self):
        return self._procs

    @property
    def num_of_carts(self):
        return self._number_of_carts

    @property
    def cart_decision_times(self):
        return self._carts_decision_times

    @staticmethod
    def print_bus_loc(busloc, occupancy):
        up = ['__'] * 8
        bus_list = [' | '] * 8

        bus_list[busloc] = ' |' + str(busloc) + '({})'.format(occupancy) + '| '

        # print('\n')
        # print('BUS LOCATION')
        print(' ' + '-' * 40)
        print("  ".join(bus_list))
        print('', '-' * 40)

    @staticmethod
    def print_passengers(a):
        up = ['__'] * 8
        pass_list = []

        for i in range(a.size):
            pass_list.append(' |' + str(a[i]))

        # print('\n')
        # print('PASSENGERS LOCATION')
        print(' ' + '-' * 40)
        print("  ".join(pass_list) + '   |')
        print('-' * 40)

    def print_other_carts(self):
        other_carts = []

        if len(self.carts) > 1:
            other_carts = {key: cart for (key, cart) in self.carts.items()
                           if self.current_cart.cartID != cart.cartID}

        pos_txt = [str(t) for t in range(NUMBER_OF_BUS_STOPS)]
        # pos_txt = [0]*NUMBER_OF_BUS_STOPS

        key_list = []

        if len(self.carts) > 1:
            for _, cart in other_carts.items():
                status = '>' if cart._state == 'In_Motion' else '|'
                occupancy = cart.occupancy
                key_list.append(cart.current_stop)

                pos_txt[cart.current_stop] = str(pos_txt[cart.current_stop]) + 'T({}){}'.format(occupancy, status)

        # add current cart

        status = '>' if self.current_cart._state == 'In_Motion' else '|'
        occupancy = self.current_cart.occupancy

        pos_txt[self.current_cart.current_stop] = str(pos_txt[self.current_cart.current_stop]) + \
                                                  'C-({}){}'.format(occupancy, status)

        return pos_txt

    def issueCall(self, current_stop):
        # Issue call to the closest cart (closest stop is zero ind)
        # puts the current stop at the end of the index

        closest_vect = np.roll(self._stops_vec[::-1], current_stop)
        cart_positions = [[cart.cartID, cart.current_stop, cart.occupancy,
                           cart.passengers_to_getoff_at(current_stop), cart._state]
                          for (_, cart) in self.carts.items()]

        cart_ind = sorted([(np.where(closest_vect == cart_positions[j][1])[0][0], j)
                           for j in range(len(cart_positions))])

        if current_stop == closest_vect[cart_ind[0][1]]:
            '''Check if bus stop assign call to a moving car whose stop is not changed yet'''
            if cart_positions[cart_ind[0][1]][4] == 'In_Motion':
                closest_cart = cart_positions[cart_ind[1][1]][0]

        else:
            closest_cart = cart_positions[cart_ind[0][1]][0]

        self.carts[int(closest_cart.split('_')[1])]._call_received_bool = True
        self.carts[int(closest_cart.split('_')[1])]._call_received_from = 'S_{}'.format(current_stop)

        self.stops[current_stop]._call_assigned = True
        self.stops[current_stop]._call_assigned_to_cart = closest_cart

        return closest_cart

    def event_run(self):
        '''Execute the current event'''

        current_event = self.current_event
        sim_time, proc_id, event_type, previous_action = current_event
        stop_index = int(proc_id.split('_')[1])

        def schedule_next_stop_event():

            next_time = sim_time + self.stops[stop_index].rvExpoVariate(sim_time)

            if next_time != sim_time and next_time < self._end_time:
            # if next_time < self._end_time:
                next_event = self.procs[proc_id].send(next_time)
                self.events.put(next_event)

            if next_time == sim_time:
                time_to_next_hour = 60 - self._sim_daytime.minute
                next_time = sim_time + time_to_next_hour * 60 + 1
                next_event = self.stops[stop_index]._passengerEvent(next_time, self.stops[stop_index].stopID,
                                                                    'stop_initialized', 'passenger generated')

                self.events.put(next_event)


        def schedule_next_cart_event(next_time, next_event_type):
            next_event = self.procs[proc_id].send((next_time + sim_time, next_event_type))
            self.events.put(next_event)

        def run_stop_event():
            current_stop_ind = int(proc_id.split('_')[1])
            # issue a call to the closest cart

            if event_type == 'stop_initialized':
                print('Stop is ready to generate')
                schedule_next_stop_event()

            elif event_type == 'passenger_arrives':
                self._stops[current_stop_ind].generate_passenger(sim_time)
                print('*** Passenger Arrived ***')
                schedule_next_stop_event()

            if len(self._stops[current_stop_ind].passengers) > 0:
                called_cart = self.issueCall(current_stop_ind)
                print('Closest cart {} is called'.format(called_cart))

        def run_cart_event():
            cart_index = int(proc_id.split('_')[1])

            if event_type in ['car_arrival_stop', 'car_arrival_past', 'alighting', 'boarding',
                              'decision_time_charging']:
                self.carts[cart_index].updateState(event_type, sim_time)

            if event_type == 'car_arrival_past':
                action, reason = ('NAction', 'Car continues past')
                next_time, next_event_type = self.carts[cart_index].generateNext_TimeEvent(action)
                schedule_next_cart_event(next_time, next_event_type)

            print('-' * 36 + '-' * 36)
            # print('\n')
            print(dt.time_handle(sim_time))

            # if self.passenger_vector().sum() > 0:
            #     print('Passenger locations before update')
            #     self.print_passengers(self.passenger_vector())

            print('Updated Event for {}'.format(self.carts[cart_index].cartID))
            print('Current stop of the cart {} is -{}-'.format(self.carts[cart_index].cartID,
                                                               self.carts[cart_index].current_stop))

            print('Current event is -{}- at time {} in STATE -{}-'.format(event_type, dt.
                                                                          time_handle(sim_time),
                                                                          self.carts[cart_index].state))

            print('-' * 36 + '-' * 36, '\n')

        options = {'S': run_stop_event, 'C': run_cart_event}

        return options[proc_id.split('_')[0]]()

    def step(self, chosen_action=None):

        if chosen_action:
            '''agent_action will choose here'''
            action = chosen_action
            reason = 'Sampled Action'

            cart_index = int(self.current_proc_id.split('_')[1])
            self.current_cart = self.carts[cart_index]

            # self._action_counter[cart_index] += self._constrained_action_counter[cart_index]

            passengerVec = self.passenger_vector()
            cart_position = self.carts[cart_index].current_stop
            occupancy = self.carts[cart_index].occupancy
            carried = self.carts[cart_index].tot_passengers_carried

            # Update state here
            self.state = self.get_state

            # House keeping decision times
            self.cart_decision_times[self.carts[cart_index].cartID].append(self._sim_time)

            charge = self.carts[cart_index]._charge_indicator
            self._carts_state_of_charge[self.carts[cart_index].cartID].append(charge)  # add state of charge
            self._action_counter_dict[self.carts[cart_index].cartID].append(
                self._action_counter[cart_index])

            temp_sars_hist_agent = list(
                self.sars_tuple(sim_time=self._sim_time,
                                cartID=self.current_cart.cartID,
                                current_state=self.state,  # passengerVec,
                                next_state='next state',
                                cart_position=cart_position,
                                occupancy=occupancy,
                                carried=carried,
                                call_received=self.carts[cart_index]._call_received_from,
                                action='{:8.8s}'.format(action),
                                reward='{:4.3f}'.format(0), delta_reward=0,
                                time_since=0,
                                event_type='{:20s}'.format(self._current_event_type),
                                reason=self._current_reason))

            self.sars_history_agent[self.current_cart.cartID].append(temp_sars_hist_agent)

            temp_cars = list(self.csars_tuple(sim_time='{:5}'.format(self._sim_time),
                                              cartID='{:3s}'.format(self.current_cart.cartID),
                                              current_state=passengerVec,
                                              action='{:8.8s}'.format('F_' + action),
                                              cart_positions=self.print_other_carts(),
                                              reward=0,
                                              delta_reward=0,
                                              time_since=0,
                                              occupancy='{:4.2f}'.format(self.current_cart.occupancy),
                                              event_type='{:20s}'.format(self._current_event_type),
                                              charge='{:1.2f}'.format(self.current_cart._charge_indicator),
                                              reward_bat=None,
                                              reason=reason, current_stop=self.current_cart.current_stop))

            self.csars_agent_all[self.current_cart.cartID].append(temp_cars)

            temp_reward = list(
                self.cstate_reward_tuple(state=self.state, action=action, reward_pass=None, state_next=None,
                                         charge='{:1.2f}'.format(self.current_cart._charge_indicator),
                                         reward_bat=None))

            self._cstate_reward_dict[self.current_cart.cartID].append(temp_reward)

            # Set training data
            action_ind = 1 if action in ['wait', 'stop'] else 0
            temp_state = self.action_state_tuple(state=self.state, reward=None, action=action_ind,
                                                 timediff=self._sim_time, state_next=None)
            self._action_states[self.carts[cart_index].cartID] = list(temp_state)

            # Schedule next time, next event
            self._action_counter[cart_index] += 1
            next_time, next_event = self.carts[cart_index].generateNext_TimeEvent(action)
            next_event_full = self.procs[self.current_proc_id].send(
                (next_time + self._sim_time, next_event))
            self.events.put(next_event_full)

            print('=' * 36 + '=' * 36, '\n')
            print('={}= is chosen because {}'.format(action, reason))
            print('Next event will be -{}- at time {}'.format(next_event,
                                                              dt.time_handle(self._sim_time + next_time)))
            print('Passenger locations after update')
            self.print_passengers(self.passenger_vector())

            for i in range(len(self.carts)):
                print('\n', 'Location of Cart {}'.format(self.carts[i].cartID))
                self.print_bus_loc(self.carts[i].current_stop, self.carts[i].occupancy)

                if self.carts[i].current_state == 'In_Motion':
                    print(' ' * 5 * self.carts[i].current_stop, '---->')

            print('=' * 36 + '=' * 36, '\n')

        while self._sim_time < self._end_time:

            if self.events.empty():
                print('*** end of the events ***')
                self.close()
                break

            # get current event and run for required actions on time
            self.current_event = self.events.get()
            self._sim_time, self.current_proc_id, self._current_event_type, info = self.current_event
            self._sim_daytime = dt.time_handle(time_in_seconds=self._sim_time)

            # Run the current event, update the states and schedule the next event
            self.event_run()

            if self.current_proc_id.split('_')[0] == 'C':
                cart_index = int(self.current_proc_id.split('_')[1])
                self.current_cart = self.carts[cart_index]

                passengerVec = self.passenger_vector()
                cart_position = self.carts[cart_index].current_stop
                occupancy = self.carts[cart_index].occupancy
                carried = self.carts[cart_index].tot_passengers_carried

                if self._current_event_type != 'car_arrival_past':
                    '''
                        Carts only take actions before the stops while moving
                        and when there is no get-in get-off when stopping
                    '''
                    try:
                        action, reason = self.carts[cart_index].chooseAction()

                    except TypeError:
                        a = 1

                    # vectfunc = np.vectorize(self.carts[cart_index].chooseAction(), otypes=[np.float], cache=False)

                    self._current_action = action
                    self._current_reason = reason

                    if action != 'agent_action':
                        reward = 0.0
                        delta_reward = 0.0

                        self._const_action_counter_dict[self.carts[cart_index].cartID].append(
                            self._action_counter[cart_index])

                        temp_sars_hist = list(
                            self.sars_tuple(sim_time=self._sim_time, cartID=self.current_cart.cartID,
                                            current_state=passengerVec,
                                            next_state='next state',
                                            cart_position=cart_position,
                                            occupancy=occupancy,
                                            carried=carried,
                                            call_received=self.carts[cart_index]._call_received_from,
                                            action='{:8.8s}'.format(action),
                                            reward=0, delta_reward=0,
                                            time_since=0,
                                            event_type='{:20s}'.format(self._current_event_type),
                                            reason=reason))

                        self.sars_history_constrained[self.current_cart.cartID].append(temp_sars_hist)

                        temp_cars = list(self.csars_tuple(sim_time='{:5}'.format(self._sim_time),
                                                          cartID=self.current_cart.cartID,
                                                          current_state=passengerVec,
                                                          action='{:8.8s}'.format('_' + action),
                                                          cart_positions=self.print_other_carts(),
                                                          reward='{:4.3f}'.format(0.0),
                                                          delta_reward='{:4.3f}'.format(0.0),
                                                          time_since='{:4.2}'.format(0.0),
                                                          occupancy='{:4.2f}'.format(
                                                              self.current_cart.occupancy),
                                                          event_type='{:20s}'.format(self._current_event_type),
                                                          charge='{:1.2f}'.format(self.current_cart._charge_indicator),
                                                          reward_bat='{:4.3f}'.format(0.0),
                                                          reason=reason, current_stop=self.current_cart.current_stop))

                        self.csars_agent_all[self.current_cart.cartID].append(temp_cars)
                        # self._constrained_action_counter[cart_index] += 1

                        if len(self.sars_history_constrained[self.current_cart.cartID]) >= 2:
                            update_ind = self._const_action_counter_dict[self.current_cart.cartID][-2]

                            self.sars_history_constrained[self.current_cart.cartID][-2][3] = passengerVec

                            self.sars_history_constrained[self.current_cart.cartID][-2][9] = '{:4.2f}'.format(
                                reward)
                            self.sars_history_constrained[self.current_cart.cartID][-2][10] = '{:4.2f}'.format(
                                delta_reward)

                            temp = self.csars_agent_all[self.current_cart.cartID][update_ind][2]

                            self.csars_agent_all[self.current_cart.cartID][update_ind][2] = '{}-A->{}'.format(
                                temp, passengerVec)

                        # Schedule next time, next event
                        next_time, next_event = self.carts[cart_index].generateNext_TimeEvent(action)
                        next_event_full = self.procs[self.current_proc_id].send(
                            (next_time + self._sim_time, next_event))

                        self.events.put(next_event_full)
                        self._action_counter[cart_index] += 1

                        print('=' * 36 + '=' * 36, '\n')
                        print('={}= is chosen because {}'.format(action, reason))
                        print('Next event will be -{}- at time {}'.format(next_event,
                                                                          dt.time_handle(self._sim_time + next_time)))
                        print('Passenger locations after update')
                        self.print_passengers(self.passenger_vector())

                        for i in range(len(self.carts)):
                            print('\n', 'Location of Cart {}'.format(self.carts[i].cartID))
                            self.print_bus_loc(self.carts[i].current_stop, self.carts[i].occupancy)

                            if self.carts[i].current_state == 'In_Motion':
                                print(' ' * 5 * self.carts[i].current_stop, '---->')

                        print('=' * 36 + '=' * 36, '\n')

                    # if self.current_cart._state == self.current_cart._cart_state_space[2]:
                    #     a = 1

                    if action == 'agent_action':
                        # observation = self.get_state
                        ta_prime = self._sim_time
                        self.current_cart.carriage_hist(self.current_cart.tot_passengers_carried)

                        # check if previous action taken
                        if len(self._carts_decision_times[self.current_cart.cartID]) >= 1:
                            ta = self._carts_decision_times[self.current_cart.cartID][-1]

                            charge_ta = self._carts_state_of_charge[self.current_cart.cartID][-1]
                            charge_ta_prime = self.current_cart._charge_indicator

                            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            # previous_state = self._action_states[self.carts[cart_index].cartID][0][1:9].sum()
                            # previous_action = self._cstate_reward_dict[self.current_cart.cartID][-1][1]
                            #
                            # reward = 0
                            #
                            # if previous_action == 'park' or 'launch':
                            #     ''' This is penalty to the continue_past_next when there is no passenger'''
                            #
                            #     # self._action_type_counter  ={key:0 if key is not previous_action else
                            #     #                                              value  for (key, value) in
                            #     #                                              self._action_type_counter.items()}
                            #
                            #
                            #     #
                            #     self._action_type_counter[previous_action] +=1
                            #
                            #     # reward = -psg.Passengers.waiting_cost_cls(beta=self._beta, ta=0, ta_prime=120)
                            #     # reward = -0.1*self._action_type_counter[previous_action]reward = self.current_cart.computeCost_R(ta, ta_prime) / self.current_cart._charge_indicator
                            #     reward = 5 + self.current_cart.computeCost_R(ta, ta_prime)
                            #
                            #
                            # elif sum(passengerVec)+previous_state == 0 and previous_action =='continue_past_next':
                            #     #zero others
                            #
                            #     # self._action_type_counter  ={key:0 if key is not previous_action else
                            #     #                                              value  for (key, value) in
                            #     #                                              self._action_type_counter.items()}
                            #     #
                            #     self._action_type_counter[previous_action] += 1
                            #
                            #     # reward = psg.Passengers.waiting_cost_cls(beta=self._beta, ta=ta, ta_prime=ta_prime)
                            #     # reward = psg.Passengers.waiting_cost_cls(beta=self._beta, ta=0, ta_prime=10)
                            #     '''ta_prime 1; -0.04, 2; 0.042,  5; 0.162, 10; 0.25'''
                            #
                            #     # reward = 0.1 * self._action_type_counter[previous_action]
                            #     reward = 5 + self.current_cart.computeCost_R(ta, ta_prime)
                            #
                            #
                            # elif sum(passengerVec)+previous_state == 0 and previous_action =='move':
                            #
                            #     # self._action_type_counter  ={key:0 if key is not previous_action else
                            #     #                                              value  for (key, value) in
                            #     #                                              self._action_type_counter.items()}
                            #     #
                            #     self._action_type_counter[previous_action] +=1
                            #
                            #     # reward = psg.Passengers.waiting_cost_cls(beta=self._beta, ta=ta, ta_prime=ta_prime)
                            #     # reward = psg.Passengers.waiting_cost_cls(beta=self._beta, ta=0, ta_prime=10)
                            #     '''ta_prime 1; -0.04, 2; 0.042,  5; 0.162, 10; 0.25'''
                            #     reward = self.current_cart.computeCost_R(ta, ta_prime)
                            #
                            # else:
                            #     reward = self.current_cart.computeCost_R(ta, ta_prime)
                            #
                            #
                            # # reward = self.current_cart.computeCost_R(ta, ta_prime)
                            #
                            # ''' No need to compte delta-reward here, it is put here to debug only'''
                            # # delta_reward = self.current_cart.computeCost_deltaR(ta, ta_prime)
                            # delta_reward = 0
                            #
                            # # # current step rewards
                            # # mean_reward, std_reward = self._func_procs['reward'].send(reward)
                            # # self._running_means_dict['reward'] = (mean_reward, std_reward)
                            # #
                            # # if std_reward != 0:
                            # #     self._reward =self.clip((reward-mean_reward)/std_reward)
                            #
                            #
                            # # self._reward = self.clip(reward/self.current_cart._charge_indicator)
                            # # max_count = np.max([self._action_type_counter['park'], self._action_type_counter['park']])
                            # max_count = 1;
                            # self._reward = self.clip(reward*max_count)
                            # #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                            delta_reward = 0  # no need to compute here
                            self._delta_reward = delta_reward

                            reward_passenger = self.current_cart.computeCost_R(ta,
                                                                               ta_prime)  # all rewards are the costs
                            reward_battery = self.current_cart.battery_charge_cost(charge_ta, charge_ta_prime)
                            reward = (self._convex_alpha) * reward_passenger + (
                                    1.0 - self._convex_alpha) * reward_battery

                            self._reward = reward / 10

                            previous_action_index = self._action_counter_dict[self.current_cart.cartID][-1]

                            # if len(self.sars_history_agent[self.current_cart.cartID]) >= 1:
                            #
                            # csars_tuple = namedtuple('action_reward_state',
                            #                          'sim_time cartID current_state action cart_positions '
                            #                          'reward delta_reward time_since average_occupancy charge
                            # reward_bat event_type reason')

                            self.sars_history_agent[self.current_cart.cartID][-1][
                                3] = self.get_state,  # self.passenger_vector()

                            self.sars_history_agent[self.current_cart.cartID][-1][9] = '{:4.3f}'.format(self._reward)
                            self.sars_history_agent[self.current_cart.cartID][-1][10] = '{:4.3f}'.format(
                                delta_reward)

                            self.sars_history_agent[self.current_cart.cartID][-1][11] = '{:4.2f}'.format(
                                ta_prime - ta)

                            # Update training set
                            # action_state_tuple = namedtuple('action_state', 'state action reward timediff state_next')
                            self._action_states[self.carts[cart_index].cartID][2] = self._reward

                            temp_time = self._action_states[self.carts[cart_index].cartID][3]
                            self._action_states[self.carts[cart_index].cartID][3] = self._sim_time - temp_time
                            self._action_states[self.carts[cart_index].cartID][4] = self.get_state

                            # if len(self.csars_agent_all[self.current_cart.cartID]) >= 1:
                            try:
                                temp = self.csars_agent_all[self.current_cart.cartID][previous_action_index][2]

                                self.csars_agent_all[self.current_cart.cartID][previous_action_index][2] = \
                                    '{}-->{}'.format(temp, self.passenger_vector())

                                self.csars_agent_all[self.current_cart.cartID][previous_action_index][
                                    5] = '{:4.3f}'.format(self._reward)

                                self.csars_agent_all[self.current_cart.cartID][previous_action_index][
                                    6] = '{:4.3f}'.format(delta_reward)

                                self.csars_agent_all[self.current_cart.cartID][previous_action_index][7] = \
                                    '{:4.2f}'.format(ta_prime - ta)

                                # all occupancies
                                occupancy_all = [cart.occupancy for (key, cart) in self.carts.items()]

                                self.csars_agent_all[self.current_cart.cartID][previous_action_index][8] = \
                                    '{:4.2f}'.format(sum(occupancy_all))
                                self.csars_agent_all[self.current_cart.cartID][previous_action_index][10] = \
                                    '{:2.2f}'.format(reward_battery)

                                self._cstate_reward_dict[self.current_cart.cartID][-1][2] = self._reward
                                self._cstate_reward_dict[self.current_cart.cartID][-1][3] = self.get_state
                                self._cstate_reward_dict[self.current_cart.cartID][-1][5] = reward_battery



                            except IndexError:
                                print(IndexError('There is something wrong'))

                        return self._action_states[self.carts[cart_index].cartID], self._reward, self._done

        if self._sim_time >= self._end_time:
            self._done = True
            self.close()
            msg = '*** end of simulation time: {} events pending ***'

            print('\n', msg.format(self.events.qsize()))
            print('Env {} ended'.format(self._envID), '\n\n')

            frames_cart_pass = []
            frames_sars_constrained = []
            frames_sars_agent = []

            frames_carts_agent = []
            frames_csate_reward = []
            meanTimeEnv = []

            for (i, cart) in self.carts.items():
                cart.stats = pd.DataFrame(cart.stats_list,
                                          columns=self.carts[i]._stats_tuple._fields)

                meanTime = []

                for jtemp in range(len(cart.stats_list)):
                    # cart.stats_list[0].waitingDuration
                    meanTime.append(float(cart.stats_list[jtemp].waitingDuration))

                if len(meanTime) != 0:
                    meanTime = sum(meanTime) / len(meanTime)

                else:
                    meanTime = 0.0

                #
                # else:
                #     meanTime = 0

                meanTimeEnv.append(meanTime)

                cart_agent = pd.DataFrame(self.csars_agent_all[cart.cartID],
                                          columns=self.csars_tuple._fields)

                cart_agent.to_csv(
                    '../tests/logs/cart_{}_agent_stats_env_{}.csv'.format(self.carts[i].cartID, self._envID), sep="\t")

                cart.stats.to_csv(
                    '../tests/logs/cart_{}_passenger_stats_env_{}.csv'.format(self.carts[i].cartID, self._envID),
                    sep="\t")

                frames_cart_pass.append(cart.stats)
                frames_csate_reward.append(pd.DataFrame(self._cstate_reward_dict[self.carts[i].cartID],
                                                        columns=self.cstate_reward_tuple._fields))

                self.sars_history_agent[cart.cartID] = pd.DataFrame(list(self.sars_history_agent[cart.cartID]),
                                                                    columns=self.sars_tuple._fields)
                self.sars_history_agent[cart.cartID].to_csv(
                    '../tests/logs/sars_history_agent_cart{}_env_{}_.csv'.format(i, self._envID),
                    sep='\t')

                self.sars_history_constrained[cart.cartID] = pd.DataFrame(self.sars_history_constrained[cart.cartID],
                                                                          columns=self.sars_tuple._fields)
                self.sars_history_constrained[cart.cartID].to_csv(
                    '../tests/logs/sars_history_constrained_cart{}_env_{}_.csv'.format(i, self._envID),
                    sep='\t')

                frames_carts_agent.append(cart_agent)
                frames_sars_agent.append(self.sars_history_agent[cart.cartID])
                frames_sars_constrained.append(self.sars_history_constrained[cart.cartID])

            carts_passenger_stats_pd = pd.concat(frames_cart_pass)
            sars_cart_const_stat_pd = pd.concat(frames_sars_constrained)
            sars_cart_agent_stat_pd = pd.concat(frames_sars_agent)
            cars_agent_stats_pd = pd.concat(frames_carts_agent)
            cstate_reward_pd = pd.concat(frames_csate_reward)

            carts_passenger_stats_pd.to_csv('../tests/logs/carts_passenger_stats_env_{}.csv'.format(self._envID),
                                            sep="\t")
            sars_cart_const_stat_pd.to_csv('../tests/logs/sars_history_constrained_cart_env_{}.csv'.format(self._envID),
                                           sep="\t")
            sars_cart_agent_stat_pd.to_csv('../tests/logs/sars_history_agent_cart_env_{}.csv'.format(self._envID),
                                           sep="\t")
            cars_agent_stats_pd.to_csv('../tests/logs/cars_agent_stats_pd_{}.csv'.format(self._envID), sep="\t")
            cstate_reward_pd.to_csv('../tests/logs/cars_cstate_reward_pd_{}.csv'.format(self._envID), sep="\t")

            self._pass_mean_time = sum(meanTimeEnv) / self.num_of_carts

        return self._action_states[self.carts[cart_index].cartID], self._reward, self._done

    def running_average(self):
        mu = 0
        count = 0

        # second moment
        sq_sum = 0
        std = 0
        var = 0

        while True:

            x = yield self.running_mean(mu, std)

            # if x is None:
            #     break

            count += 1
            if count > 0:
                mu_new = mu + (x - mu) / count
                sq_sum += (x - mu) * (x - mu_new)  # squared sum
                mu = mu_new

                var = sq_sum / count
                std = var ** (1 / 2)

            else:
                mu = 0

    def clip(self, val, min=-10.0, max=10.0):
        if val < min:
            val = min

        if val > max:
            val = max

        return val
