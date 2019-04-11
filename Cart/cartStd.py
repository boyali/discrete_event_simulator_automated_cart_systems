import logging
import random
from collections import Counter
from collections import namedtuple
from itertools import count

import numpy as np
import pandas as pd
import scipy.stats as stats

from env import process_time as dt


class DecisionTaken(namedtuple('Decision_Taken', "time bus_stop decision")):
    """Something special. usage ; a = Decision_Taken.from_defaults() """

    @classmethod
    def from_defaults(cls, time, bus_stop=None, decision=5):
        return cls(time, bus_stop, decision)


class Cart(object):
    _num_of_carts = count(0)
    _cart_ring_counter = count(0)
    _capacity = 8
    _total_distance_epoch = 0
    _total_distance = 0
    _charge_capacity = 200000  # 50 km with one charge

    _decision_interval = 30
    _charge_duration = 1 * 3600 / 12  # two hours

    _speed = 12 / 3.6  # m/s
    _acc = 1  # m/s^2 acceleration, deceleration magnitude

    _cartEvent = namedtuple('cartEvent', 'time cartID event_type info')

    _cart_event_space = {0: 'decision_time_stopped', 1: 'alighting', 2: 'boarding',
                         3: 'decision_time_motion', 4: 'car_arrival_stop', 5: 'car_arrival_past',
                         6: 'decision_time_charging'}

    _cart_state_space = {0: 'Stopped', 1: 'In_Motion', 2: 'Charging'}
    _cart_inverse_state = {'Stopped': 0, 'In_Motion': 1, 'Charging': 2}
    _cart_action_space = {'Stopped': ['move', 'wait'],
                          'In_Motion': ['continue_past_next', 'stop'],
                          'Charging': ['park', 'launch']}

    _cart_policy = {0: 'run whenever there is a call'}

    _distance_vector = {'01': 435, '12': 352, '23': 180,
                        '34': 170, '45': 190, '56': 163, '67': 393, '70': 435}
    num_of_stops = len(_distance_vector)

    _stats_tuple = namedtuple('passengerStats',
                              ['passID', 'passengerType', 'daytime', 'arrival_time', 'fromStop', 'toStop',
                               'getIn_time', 'getOffTime', 'waitingDuration', 'carriedBy', 'stop_states'])

    formatter = logging.Formatter('%(name)s:  %(message)s')

    _passengerTypes = {1: 'single', 2: 'couple', 3: 'triple', 4: 'family'}
    _passengerGroupNum = {'single': 1, 'couple': 2, 'triple': 3, 'family': 4}

    def __init__(self, env,
                 current_stop=0,  # station_id
                 motion_status='stopped',  # parked, in_motion
                 passengers=None,
                 ):

        # Prepare variables
        self._cartID = next(self._num_of_carts)
        self._env = env

        self._current_stop = current_stop
        self._next_stop = (current_stop + 1) % len(self._distance_vector)
        self._previous_stop = current_stop

        self._motion_status = motion_status
        self._occupancy = 0
        self._state = self._cart_state_space[0]

        self._previous_action = ''
        self._current_action = ''
        self._current_reason = ''

        self._current_event = self._cart_event_space[0]
        self._previous_event = self._cart_event_space[0]

        self._next_event = ''
        self._next_time = 0
        self._previous_sim_time = 0

        if passengers is None:
            self._passengers = []
        else:
            self._passengers = list(passengers)

        self._distance_matrix = self.make_distance_matrix()
        self._passenger_destinations = np.array([-1] * self._capacity)

        self.stats = pd.DataFrame(columns=self._stats_tuple._fields)
        self.stats_list = []

        self._cost_increment = 0  # deltaR
        self._call_received_bool = False
        self._call_received_from = ''

        self.num_of_getin = self._capacity

        self._passengers_carried = count(0)
        self._total_passenger_carried = 0
        self._carriage_hist = [0] * 2  # keeps occupancy during the actions  using num of carried
        self._dt_avg_occupancy = 0
        self._dt_x_occupancy = 0

        # cart incremantal discounted cost
        self._total_discounted_cost = 0

        # charging parameters
        self._charge_start_time = None
        self._charge_finish_time = None
        self._charge_max = 1.0
        self._charge_min = 0.1
        self._charge_indicator = self._charge_max
        self._charge_used_cumsum = 0.0

        self._in_service = True  # if charging False

        # Logger events
        open('../tests/logs/log_cart_{}.txt'.format(self.cartID), 'w').close()
        self.file_handler_cart = logging.FileHandler('../tests/logs/log_cart_{}.txt'.format(self.cartID))
        self.file_handler_cart.setFormatter(self.formatter)
        self.file_handler_cart.flush()

        self.logger_cart = logging.getLogger('../tests/logs/log_cart_{}.txt'.format(self.cartID))
        self.logger_cart.setLevel(logging.INFO)
        self.logger_cart.addHandler(self.file_handler_cart)

    @property
    def dt_avg_occupancy(self):
        """Returns average occupancy between the actions"""

        if self._previous_sim_time > 0:
            self._dt_avg_occupancy = self._dt_x_occupancy / self._previous_sim_time
        return (1 - self._dt_avg_occupancy / self._capacity)

    def carriage_hist(self, val):
        self._carriage_hist.pop(0)
        self._carriage_hist.append(val)

    def reset(self):
        # env = self._env
        # self.__init__(env)
        self._current_stop = 0
        self._passengers = []
        self._state = self._cart_state_space[0]
        self._previous_action = ''
        self._current_action = ''
        self._current_reason = ''
        self._previous_state = 'Stopped'

        self._current_event = self._cart_event_space[0]
        self._previous_event = self._cart_event_space[0]

        self._next_event = ''
        self._next_time = 0

        self._passenger_destinations = np.array([-1] * self._capacity)

        # keep statistics
        self.stats = pd.DataFrame(columns=self._stats_tuple._fields)
        self.stats_list = []
        # self.cart_sars = []

        self._cost_increment = 0  # deltaR
        self._call_received_bool = False
        self._call_received_from = ''

        self.num_of_getin = self._capacity

        self._passengers_carried = count(0)
        self._total_passenger_carried = 0
        self._carriage_hist = [0] * 2  # keeps occupancy during the actions  using num of carried
        self._dt_avg_occupancy = 0
        self._dt_x_occupancy = 0

        # cart incremantal discounted cost
        self._total_discounted_cost = 0

        # total distance
        self._charge_start_time = None
        self._charge_finish_time = None
        self._charge_indicator = self._charge_max
        self._charge_used_cumsum = 0.0

        self._cart_ring_counter = count(0)
        self._total_distance_epoch = 0
        self._total_distance = 0

    def cart_process(self):
        # event_types = {0: 'cart_initialized', 1: 'decision_time',
        #                2: 'car_arrival'}

        print('Cart {} is ready to receive a call at time {}'.format(self.cartID, 0))

        event = yield self._cartEvent(0, self.cartID, 'cart_initialized', 'ready to receive call at time {}'.format(0))
        time, cart_event_type = event

        while True:

            if time >= 0:

                event = yield self._cartEvent(time, self.cartID, cart_event_type,
                                              'next event will be {} at time {}'.format(cart_event_type,
                                                                                        dt.time_handle(time)))
                time, cart_event_type = event

            else:
                print('Cart Process is terminated at the cart stop {} at time {}'.format(
                    self.current_stop, time))
                break

    @property
    def passengers(self):
        return self._passengers

    @property
    def passenger_destinations(self):
        self._passenger_destinations = [p.destination for p in self._passengers]
        return self._passenger_destinations

    @property
    def distance_matrix(self):
        return self._distance_matrix

    @property
    def motion_status(self):
        return self._motion_status

    @motion_status.setter
    def motion_status(self, value):
        self._motion_status = value

    @property
    def call_received(self):
        return self._call_received_bool

    @call_received.setter
    def call_received(self, value):
        self._call_received_bool = value

    @property
    def previous_stop(self):

        if self.current_stop == 0:
            previous_stop = self.num_of_stops - 1

        else:
            previous_stop = self.current_stop - 1
        return previous_stop

    @property
    def current_stop(self):
        return self._current_stop

    @current_stop.setter
    def current_stop(self, value):
        self._current_stop = value % len(self._distance_vector)

    @property
    def next_stop(self):
        return (self._current_stop + 1) % len(self._distance_vector)

    @property
    def next_next_stop(self):
        return (self._current_stop + 2) % len(self._distance_vector)

    @property
    def occupancy(self):
        self._occupancy = len(self._passengers)
        return self._occupancy

    @occupancy.setter
    def occupancy(self, value):
        self._occupancy = value

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def acceleration(self):
        return self._acc

    @acceleration.setter
    def acceleration(self, value):
        self._acc = value

    @property
    def current_state(self):
        return self._state

    @current_state.setter
    def current_state(self, value):
        self._state = value

    @property
    def previous_state(self):
        return self._previous_state

    @previous_state.setter
    def previous_state(self, value):
        self._previous_state = value

    @property
    def cartID(self):
        return 'C_' + str(self._cartID)

    @property
    def available_seats(self):
        return self._capacity - self.occupancy

    @property
    def tot_passengers_carried(self):

        return self._total_passenger_carried

    @property
    def passengers_to_getoff(self):
        destination_counter = dict(Counter(self.passenger_destinations))

        if len(destination_counter) != 0:
            passenger_get_off = [v for (k, v) in destination_counter.items() if self.current_stop == k]

        else:
            passenger_get_off = []

        return passenger_get_off

    @property
    def passengers_to_getoff_next(self):

        destination_counter = dict(Counter(self.passenger_destinations))

        if len(destination_counter) != 0:
            passenger_get_off = [v for (k, v) in destination_counter.items() if self.next_stop == k]

        else:
            passenger_get_off = []

        return passenger_get_off

    def passengers_to_getoff_at(self, stop_num):
        destination_counter = dict(Counter(self.passenger_destinations))

        if stop_num in destination_counter.keys():
            pass_get_off_at = True

        else:
            pass_get_off_at = False

        return pass_get_off_at

    def distance_between(self, ind1, ind2):
        """ From ind1 to ind2 """
        N = len(self._distance_vector)
        indices = []
        distances = []

        if ind1 == ind2:
            indices.append([0, 0])
            distances.append(0)
        else:
            for i in range(ind1, ind1 + N):
                indices.append((i % N, (i + 1) % N))
                distances.append(self._distance_vector[str(i % N) + str((i + 1) % N)])
                if (i + 1) % N == ind2:
                    break

        # print(indices)
        # print(distances)

        return sum(distances)

    def stops_between(self, ind1, ind2):
        """ From ind1 to ind2 """
        N = len(self._distance_vector)
        indices = []
        distances = []

        if ind1 == ind2:
            indices.append(0)
            distances.append(0)
        else:
            for i in range(ind1, ind1 + N):
                indices.append((i + 1) % N)
                # distances.append(str(i % N) + str((i + 1) % N))
                if (i + 1) % N == ind2:
                    break
        return indices

    def make_distance_matrix(self):

        distance_matrix = np.zeros(shape=(self.num_of_stops, self.num_of_stops))

        for i in range(8):
            for j in range(8):
                distance_matrix[i, j] = self.distance_between(i, j)

        return distance_matrix

    def sampleAction(self):
        # _cart_state_space = {0: 'Stopped', 1: 'In_Motion'}
        # _cart_action_space = {'Stopped': ['move', 'wait'],
        #                       'In_Motion': ['continue_past_next', 'stop']}
        action = None
        action_ind = None

        if self._state != self._cart_state_space[2]:
            state_actions = self._cart_action_space[self._state]
            action = random.sample(state_actions, 1)[0]
            action_ind = random.sample([0, 1], 1)[0]

        else:
            state_actions = self._cart_action_space[self._state]
            action = state_actions[0]
            action_ind = 0

        return action, action_ind

    def stdAction(self):

        if self._state != self._cart_state_space[2]:
            state_actions = self._cart_action_space[self._state]
            action_ind = 1
        else:
            state_actions = self._cart_action_space[self._state]
            action = state_actions[0]
            action_ind = 0

        return state_actions[action_ind], action_ind

    def exponential_decay(self, step, total_step, initial, final, rate=1e-4, stairs=None):

        if stairs is not None:
            step = stairs * np.floor(step / stairs)

        scale, offset = 1. / (1. - rate), 1. - (1. / (1. - rate))
        progress = np.array([step]).astype(np.float32) / total_step,
        value = np.array([(initial - final) * scale * rate]) ** progress + offset + final
        value = value[0][0]
        lower, upper = min([initial, final]), max([initial, final])
        return max([lower, min(value, upper)])

    def get_action(self, ind):
        # _cart_state_space = {0: 'Stopped', 1: 'In_Motion'}
        # _cart_action_space = {'Stopped': ['move', 'wait'],
        #                       'In_Motion': ['continue_past_next', 'stop']}

        state_actions = self._cart_action_space[self._state]

        return state_actions[ind]

    def erlang_rv(self, num_of_pass):
        '''
            num_of_passengers is the number of passengers getting in getting out from the cart
            1 passengers getting in and out of the cart takes time with mean = 3 and standard
            truncated standard deviation 2-7 seconds


            in scipy
            A common parameterization for expon is in terms of the rate parameter lambda,
            such that pdf = lambda * exp(-lambda * x). This parameterization corresponds
            to using scale = 1 / lambda

            exponential distribution is f(x) = lambda*exp(-lambda*x) or
            f(x) = (1/lambda)*exp(-x/lambda), scale = 1/lambda

            loc is used to shift the value of x,
            we can do dis, expon.pdf(y) / scale with y = (x - loc) / scale.


        '''

        lower, upper, scale = 5, 12, 3
        X = stats.truncexpon(b=(upper - lower) / scale, loc=lower, scale=scale)
        data = X.rvs(num_of_pass)
        # math.ceil(max(data))
        return max(data)

    @property
    def other_cart_poss(self):
        other_cart_pos = []
        if len(self._env.carts) > 1:
            other_cart_pos = [cart.current_stop for (key, cart) in self._env.carts.items()
                              if cart.cartID != self.cartID]

        return other_cart_pos

    def generateNext_TimeEvent(self, action):
        '''
            _cart_event_space = {0: 'decision_time_stopped', 1: 'alighting', 2: 'boarding',
                                 3:'decision_time_motion', 4: 'car_arrival_stop', 5: 'car_arrival_past'}

            _cart_state_space = {0: 'Stopped', 1: 'In_Motion'}

            _cart_action_space = {'Stopped': ['move', 'wait'],
                                  'In_Motion': ['continue_past_next', 'stop']}

            Updates Cart State when the action changes the state
        '''

        distance_to_next = self.distance_matrix[self.current_stop][self.next_stop]

        ta = self.speed / self.acceleration  # time to accelerate to the max speed
        td = self.speed / self.acceleration  # time to decelerate to zero

        xa = self.speed * ta / 2
        xd = self.speed * td / 2

        def Stopped_state_event():
            '''In stopped state, next event and next time'''

            # _cart_event_space = {0: 'decision_time_stopped', 1: 'alighting', 2: 'boarding',
            #                      3: 'decision_time_motion', 4: 'car_arrival_stop', 5: 'car_arrival_past'}
            if self._in_service:
                if action == 'move':
                    self._next_event = self._cart_event_space[3]  # decision_time_motion
                    tc = (distance_to_next - xa) / self.speed
                    self._next_time = ta + tc

                    self.current_state = self._cart_state_space[1]  # switch to in_motion state immediately


                elif action == 'wait':
                    num_of_waiting = self._env.stops[self.current_stop].passengers_waiting
                    num_of_alighting = self.passengers_to_getoff

                    # check if next stop is occupied
                    num_of_carts = self._env.num_of_carts

                    # is_next_occupied = self._env.stops[self.next_stop].occupied_by_acart
                    is_next_occupied = [True if (cart.current_stop == self.next_stop) else False
                                        for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID]

                    is_there_cart_heading = [
                        True if (cart.state == 'In_Motion' and cart.next_stop == self.next_stop) else False
                        for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID]

                    t_get_in = None
                    t_get_off = None

                    if not (num_of_alighting or num_of_waiting > 0) or (
                            num_of_carts > 1 and (any(is_next_occupied) or any(is_there_cart_heading))):
                        self._next_event = self._cart_event_space[0]  # decision_time_stop
                        self._next_time = self._decision_interval

                    else:
                        if num_of_alighting:
                            self._next_event = self._cart_event_space[1]  # alighting
                            # t_get_off = num_of_alighting[0] * 3  # will change to Gamma - Erlang Distribution
                            t_get_off = self.erlang_rv(num_of_alighting)
                            self._next_time = t_get_off

                        elif num_of_waiting and not num_of_alighting:
                            # a = self.num_of_getin
                            self._next_event = self._cart_event_space[2]  # boarding

                            if num_of_waiting > self.available_seats:
                                # t_get_in = self.available_seats * 3  # will change to Gamma - Erlang Distribution
                                t_get_in = self.erlang_rv(self.available_seats)

                            else:
                                # t_get_in = num_of_waiting * 3  # will change to Gamma - Erlang Distribution
                                t_get_in = self.erlang_rv(num_of_waiting)

                            self._next_time = t_get_in

            else:  # not in service
                if action == 'move':
                    self._next_event = self._cart_event_space[3]  # decision_time_motion
                    tc = (distance_to_next - xa) / self.speed
                    self._next_time = ta + tc
                    self.current_state = self._cart_state_space[1]  # switch to in_motion state immediately

                elif action == 'wait':
                    num_of_alighting = self.passengers_to_getoff

                    if not num_of_alighting:
                        if self.current_stop != 0:
                            self._next_event = self._cart_event_space[0]  # decision_time_stop
                            self._next_time = self._decision_interval

                        else:  # now in charging state
                            self._next_event = self._cart_event_space[6]  # decision_time_in_charging
                            self._next_time = self._decision_interval

                    else:
                        self._next_event = self._cart_event_space[1]  # alighting
                        # t_get_off = num_of_alighting[0] * 3  # will change to Gamma - Erlang Distribution
                        t_get_off = self.erlang_rv(num_of_alighting)
                        self._next_time = t_get_off

            if self._next_event == '':
                raise ValueError('Next Event is not defined')

            return (round(self._next_time), self._next_event)

        def In_Motion_state_event():
            # _cart_event_space = {0: 'decision_time_stopped', 1: 'alighting', 2: 'boarding',
            #                      3: 'decision_time_motion', 4: 'car_arrival_stop', 5: 'car_arrival_past'}

            '''In motion state, next event and next time'''
            if self._in_service:
                if action == 'stop':
                    self._next_event = self._cart_event_space[4]  # car_arrival_stop
                    self._next_time = td

                elif action == 'continue_past_next':
                    self._next_event = self._cart_event_space[5]  # car_arrival_past
                    self._next_time = xd / self.speed

                elif action == 'move':
                    self._next_event = self._cart_event_space[3]  # decision_time_motion
                    self._next_time = xd / self.speed

                elif action == 'NAction':
                    self._next_event = self._cart_event_space[3]  # car_arrival_past
                    self._next_time = (distance_to_next - xd) / self.speed

            else:  # going to charging

                if action == 'continue_past_next':
                    # a = self.current_stop
                    # b = self.occupancy
                    self._next_event = self._cart_event_space[5]  # car_arrival_past
                    self._next_time = xd / self.speed

                elif action == 'move':
                    self._next_event = self._cart_event_space[3]  # decision_time_motion
                    self._next_time = xd / self.speed

                elif action == 'stop':
                    self._next_event = self._cart_event_space[4]  # car_arrival_stop
                    self._next_time = td

                elif action == 'NAction':
                    self._next_event = self._cart_event_space[3]  # car_arrival_past
                    self._next_time = (distance_to_next - xd) / self.speed

            return (round(self._next_time), self._next_event)

        def Charging_state_event():

            # num_of_carts = self._env.num_of_carts
            # is_next_occupied = [True if (cart.current_stop == self.next_stop) else False
            #                     for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID]
            #
            # is_there_cart_heading = [
            #     True if (cart.state == 'In_Motion' and cart.next_stop == self.next_stop) else False
            #     for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID]

            # if action == 'park':
            self._next_event = self._cart_event_space[6]  # decision_time_charging
            self._next_time = self._charge_duration
            self._charge_finish_time = self._charge_start_time + self._charge_duration

            # self._total_distance += self._total_distance_epoch

            # elif action == 'launch':
            #
            #     self._next_event = self._cart_event_space[0]  # decision_time_stop
            #     self._next_time = self._decision_interval
            #
            #     self.current_state = self._cart_state_space[1]  # switch to in_motion state immediately
            #     self._in_service = True

            # else: # parked
            #     self._next_event = self._cart_event_space[0]  # decision_time_stop
            #     self._next_time = self._decision_interval
            #     self.current_state = self._cart_state_space[0]  # switch to stopped state immediately
            #
            #     self._in_service = True

            return (round(self._next_time), self._next_event)

        options = {'Stopped': Stopped_state_event,
                   'In_Motion': In_Motion_state_event,
                   'Charging': Charging_state_event}

        return options[self.state]()

    def chooseAction(self):
        '''Return an action with a reason'''

        # Implements call received if the current is closer than the other carts
        # call_received = self._env.passenger_vector().sum()  #
        call_received = self._call_received_bool

        num_of_waiting = self._env.stops[self.current_stop].passengers_waiting
        num_of_alighting = self.passengers_to_getoff

        num_of_waiting_next = self._env.stops[self.next_stop].passengers_waiting
        num_of_alighting_next = self.passengers_to_getoff_next

        num_of_carts = self._env.num_of_carts

        def Stopped_next_action():
            '''
                check if there is a call from any cart stop if cart is empty
                check if there is other cart stopped, alighting, boarding or stopped
            '''

            # _cart_action_space = {'Stopped': ['move', 'wait'],
            #                       'In_Motion': ['continue_past_next', 'stop']}

            is_next_occupied = [True if (cart.current_stop == self.next_stop and cart.current_stop != 0) else False
                                for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID]

            is_there_cart_heading = [True if (cart.state == 'In_Motion' and cart.next_stop == self.next_stop) else False
                                     for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID]

            is_there_cart_behind = [
                True if (cart.current_stop == self.previous_stop) else False
                for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID
                                                             and self.current_stop not in [0]]

            action = None
            reason = ''

            if self._in_service:
                try:
                    if not (num_of_waiting or num_of_alighting or call_received):
                        '''No waiting, no alighting or no call'''

                        if self.occupancy > 0:
                            if not (any(is_next_occupied) or any(is_there_cart_heading)):
                                action = 'move'
                                reason = 'passengers are waiting in the cart'

                            else:
                                action = 'wait'
                                reason = 'Next bus stop is occupied or there is cart heading to next'

                        elif self.occupancy == 0:
                            if not (any(is_next_occupied) or any(is_there_cart_heading)):
                                if self.current_stop not in [0]:
                                    if not any(is_there_cart_behind):

                                        action = 'agent_action'
                                        reason = 'sampled action - free to choose {}'.format(all(is_there_cart_behind))
                                    else:
                                        action = 'move'
                                        reason = 'Behind there is a cart'
                                else:
                                    action = 'agent_action'
                                    reason = 'sampled action - free to choose'
                            else:
                                action = 'wait'
                                reason = 'Next bus stop is occupied or there is cart heading to next'

                    else:

                        if num_of_alighting:
                            action = 'wait'
                            reason = 'there are people in the cart getting-off'

                        elif num_of_waiting:
                            if self.occupancy < self._capacity:
                                # check if available seats will be sum of group lengths
                                groups = self._env.stops[self.current_stop].passenger_groups()
                                current_stop_groups = [z[0] * [z[0]] for z in groups for i in range((len(z) // z[0]))]
                                group_lengths = [i[0] for i in current_stop_groups]

                                available_seats = self.available_seats
                                can_get_passengers = False

                                for i, j in enumerate(group_lengths):

                                    if available_seats == 0:
                                        break

                                    elif group_lengths[i] == 1:
                                        available_seats -= 1
                                        can_get_passengers = True

                                    elif available_seats >= group_lengths[i] and group_lengths[i] != 1:
                                        available_seats -= group_lengths[i]
                                        can_get_passengers = True

                                    elif available_seats < group_lengths[i] and can_get_passengers:
                                        pass

                                if not can_get_passengers:
                                    if (num_of_carts > 1 and not any(is_next_occupied) and not any(
                                            is_there_cart_heading)) or \
                                            num_of_carts == 1:

                                        action = 'move'
                                        reason = 'nobody wants to get in from the stop'

                                    else:
                                        action = 'wait'
                                        reason = 'Next bus stop is occupied or there is cart heading to next'

                                else:
                                    'can get passenger'
                                    action = 'wait'
                                    reason = 'there are people waiting at the current stop and can get-in'

                                self.num_of_getin = self.available_seats - available_seats

                            if self.occupancy == self._capacity:
                                if not (any(is_next_occupied) and any(is_there_cart_heading)):
                                    action = 'move'
                                    reason = 'there are people waiting but no seats'

                                else:
                                    action = 'wait'
                                    reason = 'the next bus stop is occupied by a cart'

                        elif call_received:
                            if not (any(is_next_occupied) or any(is_there_cart_heading)):
                                action = 'move'
                                reason = 'Call received other than current stop'

                            else:
                                action = 'wait'
                                reason = 'Next bus stop is occupied or there is cart heading to next'

                except ValueError:
                    print(ValueError)

            else:  # not in service

                if self.occupancy > 0:
                    if not num_of_alighting:
                        if not (any(is_next_occupied) or any(is_there_cart_heading)):
                            action = 'move'
                            reason = 'Out of service for charge - will drop the passengers first'

                        else:
                            action = 'wait'
                            reason = 'waiting, next is occupied while going to charging'

                    else:
                        action = 'wait'
                        reason = 'there are people in the cart getting-off'

                else:
                    if self.current_stop != 0:
                        if not (any(is_next_occupied) or any(is_there_cart_heading)):
                            action = 'move'
                            reason = 'Going to charging station from stopped'

                        else:
                            action = 'wait'
                            reason = 'waiting, next is occupied while going to charging'

                    else:
                        action = 'wait'
                        reason = 'arrived to charging station'
                        self._charge_start_time = self._env._sim_time
                        self.current_state = self._cart_state_space[2]  # switch to in_motion state immediately

            if action is None:
                raise ValueError('conditions are not met')

            self._previous_action = self._current_action
            self._current_action = action

            # results = (action, reason)
            return action, reason

        def In_Motion_next_action():
            '''Return an action with a reason'''

            is_next_next_occupied = [
                True if (cart.current_stop == self.next_next_stop and cart.current_stop != 0) else False
                for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID]

            is_there_cart_heading_next = [
                True if (cart.state == 'In_Motion' and cart.next_stop == self.next_next_stop) else False
                for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID]

            is_there_cart_behind = [
                True if (cart.current_stop == self.previous_stop) else False
                for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID
                                                             and self.current_stop not in [0]]

            action = None
            reason = ''

            # passenger_vector = self._env.passenger_vector()
            if self._in_service:
                if not (num_of_waiting_next or num_of_alighting_next or call_received):
                    if not (any(is_next_next_occupied) or any(is_there_cart_heading_next)):
                        if self.occupancy > 0:
                            action = 'continue_past_next'
                            reason = 'passengers are waiting in the cart, no call, no passenger at the next'

                        else:

                            # if (not any(is_there_cart_behind) or (any(is_there_cart_behind) and self.current_stop == 0)):
                            action = 'agent_action'
                            reason = 'sampled action - free to choose {}'.format(any(is_there_cart_behind))

                            # else:
                            #     action = 'move'
                            #     reason = 'Behind there is cart '

                    else:
                        action = 'stop'
                        reason = 'Next bus stop is occupied or there is cart heading to next'

                else:
                    if num_of_alighting_next:
                        action = 'stop'
                        reason = 'there are passengers to get-off at the next station'

                    # elif num_of_waiting_next > 0 and self.occupancy < 4:
                    #     action = 'agent_action'
                    #     reason = 'sampled action - free to choose'

                    elif num_of_waiting_next > 0:
                        if self.occupancy < self._capacity:
                            action = 'stop'
                            reason = 'There are people waiting at the next stop'
                            # action = 'agent_action'
                            # reason = 'We left the agent to decide whether to pick up passengers'

                        elif self.occupancy == self._capacity and not num_of_alighting_next:
                            if not (any(is_next_next_occupied) or any(is_there_cart_heading_next)):
                                action = 'continue_past_next'
                                reason = 'Even there are passengers waiting at the next stop to get-in and there is NO seats'
                            else:
                                action = 'stop'
                                reason = 'Next bus stop is occupied or there is cart heading to next'



                    elif call_received:
                        if not (any(is_next_next_occupied) or any(is_there_cart_heading_next)):
                            action = 'continue_past_next'
                            reason = 'Call received and no get-in and get-off'

                        else:
                            action = 'stop'
                            reason = 'Next bus stop is occupied or there is cart heading to next'

                    else:
                        if not (any(is_next_next_occupied) or any(is_there_cart_heading_next)):
                            action = 'agent_action'
                            reason = 'sampled action - free to choose'

            else:  # not in service

                if self.occupancy != 0:
                    if not num_of_alighting_next:
                        if not (any(is_next_next_occupied) or any(is_there_cart_heading_next)):
                            action = 'continue_past_next'
                            reason = 'Out of service for charge - will drop the passengers first'

                        else:
                            action = 'stop'
                            reason = 'Next is occupied while going to charging station - there are passengers in'

                    else:
                        action = 'stop'
                        reason = 'there are passengers to get-off at the next station'

                else:
                    if self.next_stop == 0:
                        action = 'stop'
                        reason = 'Next stop is the charging station'

                    else:
                        action = 'continue_past_next'
                        reason = 'Going to charging station'

            if action == None:
                raise ValueError('action is not chosen')

            self._previous_action = self._current_action
            self._current_action = action
            self._current_reason = reason

            return (action, reason)

        def Charging_next_action():
            # action = None
            # reason = ''
            #
            # is_next_occupied = [True if (cart.current_stop == self.next_stop and cart.current_stop != 0) else False
            #                     for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID]
            #
            # is_there_cart_heading = [True if (cart.state == 'In_Motion' and cart.next_stop == self.next_stop) else False
            #                          for (_, cart) in self._env._carts.items() if cart.cartID != self.cartID]

            self._total_distance_epoch = 0.0
            self._charge_indicator = self._charge_max
            self._in_service = True

            # if not(any(is_next_occupied) or any(is_there_cart_heading)):
            action = 'wait'
            reason = 'Fully charged'

            # Change cart state to waiting at the 0-th stop
            self.current_state = self._cart_state_space[0]

            # else:
            #     action = 'wait'
            #     reason = ' Fully charged but next is occupied'

            return (action, reason)

        options = {'Stopped': Stopped_next_action,
                   'In_Motion': In_Motion_next_action,
                   'Charging': Charging_next_action}

        return options[self._state]()

    def updateState(self, current_event, sim_time=0):
        '''Update Cart and Environment States'''
        current_stop = self._env.stops[self.current_stop]
        num_of_alighting = self.passengers_to_getoff

        self._previous_event = current_event
        self._dt_x_occupancy += self.occupancy * (sim_time - self._previous_sim_time)
        self._previous_sim_time = sim_time

        self._previous_state = self.current_state

        def Stopped_update():
            # all the get-in, get-off is completed, update the cart stops

            # _distance_vector = {'01': 435, '12': 352, '23': 180,
            #                     '34': 170, '45': 190, '56': 163, '67': 393, '70': 435}

            self._previous_stop = self.current_stop
            self.current_stop += 1
            self.current_state = self._cart_state_space[0]  # Stopped

            # distance travelled
            if self._previous_stop != self.current_stop:

                next(self._cart_ring_counter)
                dist_travelled = self._distance_vector['{}{}'.format(self.previous_stop, self.current_stop)]

                self._total_distance_epoch += dist_travelled
                self._total_distance += dist_travelled

                current_charge = self.exponential_decay(step=self._total_distance_epoch,
                                                        total_step=self._charge_capacity,
                                                        initial=self._charge_max, final=self._charge_min,
                                                        rate=1e-1)

                delta_charge = self._charge_indicator - current_charge

                if np.absolute(delta_charge) < 0.6:
                    self._charge_used_cumsum += delta_charge

                self._charge_indicator = current_charge

        def Passed_update():

            self._previous_stop = self.current_stop
            self.current_stop += 1
            self.current_state = self._cart_state_space[1]  # In_Motion

            # distance travelled
            if self._previous_stop != self.current_stop:

                next(self._cart_ring_counter)
                dist_travelled = self._distance_vector['{}{}'.format(self.previous_stop, self.current_stop)]

                self._total_distance_epoch += dist_travelled
                self._total_distance += dist_travelled

                current_charge = self.exponential_decay(step=self._total_distance_epoch,
                                                        total_step=self._charge_capacity,
                                                        initial=self._charge_max, final=self._charge_min,
                                                        rate=1e-1)

                delta_charge = self._charge_indicator - current_charge

                if np.absolute(delta_charge) < 0.6:
                    self._charge_used_cumsum += delta_charge

                self._charge_indicator = current_charge

        def Alighting_update():

            passengers_to_alight = [p for p in self._passengers if p.destination == self.current_stop]

            for i in range(num_of_alighting[0]):
                '''Update passenger time'''
                passenger_getting_off = passengers_to_alight[i]

                passenger_getting_off.state = 'going to home'
                passenger_getting_off.getoff_time = sim_time
                current_stop._passengers_got_off_here.append(passenger_getting_off)

                self._total_passenger_carried = next(self._passengers_carried)
                # self._carriage_hist.pop(0)  # keeps occupancy during the actions  using num of carried
                # self._carriage_hist.append(self._total_passenger_carried)

                print('=' * 36 + '=' * 36)
                print('Number of passengers in the cart is {}'.format(self.occupancy))

                self._passengers.remove(passengers_to_alight[i])

                print('Passenger -{}- gets-off the cart'.format(passenger_getting_off.passengerID))
                self.logger_cart.info(
                    '{} Passenger -{} {}- gets-off the cart at stop {}'.format(dt.time_handle(sim_time),
                                                                               passenger_getting_off.passengerID,
                                                                               passenger_getting_off.passengerType,
                                                                               self.current_stop))

                print('After the passenger got-off the occupancy is {}'.format(self.occupancy))
                print('=' * 36 + '=' * 36)

                # _stats_tuple = namedtuple('passengerStats',
                #                           ['passID', 'arrival_time', 'fromStop', 'toStop'
                #                             'getIn_time', 'getOffTime', 'waitingDuration'])

                p = passenger_getting_off

                temp_stats = self._stats_tuple(p.passengerID, p._passengerType, dt.time_handle(sim_time),
                                               '{:4.0f}'.format(p.arrival_time), p.origin,
                                               '---> S_' + str(p.destination), p.getin_time, p.getoff_time,
                                               '{:4.2f}'.format((p.getin_time -p.arrival_time) / 60), self.cartID,
                                               self._env.passenger_vector().tolist())

                self.stats_list.append(temp_stats)

            # self._average_occupancy = self._carriage_hist[1] - self._carriage_hist[0]

        def Boarding_update():
            'First comes First Out'

            if self.cartID == current_stop._call_assigned_to_cart:
                '''Update call states of the cart and stop pairs'''

                self._call_received_bool = False
                self._call_received_from = ''

                current_stop._call_assigned_to_cart = ''
                current_stop._call_assigned_to = False

            available_seats = self.available_seats
            groups_org = current_stop.passenger_groups()
            # groups_srt = sorted(groups_org, key=len, reverse=True)

            current_stop_groups = [z[0] * [z[0]] for z in groups_org for i in range((len(z) // z[0]))]
            # current_stop_groups = [z[0] * [z[0]] for z in groups_srt for i in range((len(z) // z[0]))]
            # group_lengths = [i[0] for i in current_stop_groups]

            # if groups:
            #     if groups[0][0] != len(groups[0]):
            #         raise ValueError('Invalid Pickup')

            for i in range(len(current_stop_groups)):
                passengerType = self._passengerTypes[current_stop_groups[i][0]]

                if current_stop_groups[i][0] == 1:
                    maxIJ = min(len(current_stop_groups[i]), self.available_seats)

                    for maxlen in range(maxIJ):
                        pick_passenger(passengerType)

                if current_stop_groups[i][0] > 1 and self.available_seats >= current_stop_groups[i][0]:

                    for j in range(current_stop_groups[i][0]):
                        pick_passenger(passengerType)

                        # if j == current_stop_groups[i][0] - 1:
                        #     passengerType = self._passengerTypes[current_stop_groups[i][0]]

                elif current_stop_groups[i][0] > 1 and self.available_seats < current_stop_groups[i][0]:
                    pass

                if self.available_seats == 0:
                    break

                    # _cart_event_space = {0: 'decision_time_stopped', 1: 'alighting', 2: 'boarding',
                    # 3:'decision_time_motion', 4: 'car_arrival_stop', 5: 'car_arrival_past'}
                    # if i == len(current_stop_groups) - 1:
                    #     groups = current_stop.passenger_groups()
                    #     if groups:
                    #         if groups[0][0] != len(groups[0]):
                    #             raise ValueError('Invalid Pickup')

        def pick_passenger(passengerType):
            'pick the passenger when the passengerType is the first instance'
            # passenger_getting_in = current_stop.passengers[0]

            passenger_getting_in = next(passenger for passenger in current_stop.passengers
                                        if passenger.passengerType == passengerType)

            passenger_getting_in.state = 'in cart'
            passenger_getting_in.getin_time = sim_time

            # passenger_getting_in.waiting_cost = passenger_getting_in.waiting_cost(sim_time)

            current_stop.passengers.remove(passenger_getting_in)
            self._passengers.append(passenger_getting_in)

            txt1 = '{} Passenger -{} {}- got-in the bus'.format(dt.time_handle(sim_time),
                                                                passenger_getting_in.passengerID,
                                                                passenger_getting_in.passengerType)

            txt2 = 'with destination to -{}-'.format(passenger_getting_in.destination)
            txt3 = 'at the bus stop {}'.format(self.current_stop)

            print('=' * 36 + '=' * 36)
            print(' '.join([txt1, txt2, txt3]))
            self.logger_cart.info(' '.join([txt1, txt2, txt3]))

            print('This passenger waited {} seconds in total'.format(
                passenger_getting_in.waiting_duration(sim_time)))
            print('Number of passengers in the cart is {}'.format(self.occupancy))
            print('=' * 36 + '=' * 36, '\n')

            if len(current_stop.passengers) == 0:
                current_stop._call_assigned = False
                current_stop._call_assigned_to_cart = ''

        # _cart_event_space = {0: 'decision_time_stopped', 1: 'alighting', 2: 'boarding',
        #                      3: 'decision_time_motion', 4: 'car_arrival_stop', 5: 'car_arrival_past'}

        def Decision_stopped_updates():
            '''We need to update costs'''
            pass

        def Decision_motion_updates():
            '''We need to update costs'''
            pass

        def Decision_time_charging():
            '''Update chargin'''
            pass

        # after alighting, boarding, update if the cart needs to charge

        if self._charge_indicator <= 0.1:
            # self.current_state = self._cart_state_space[2]  # Cart enters the charge state
            self._in_service = False

        options = {'car_arrival_stop': Stopped_update,
                   'car_arrival_past': Passed_update,
                   'alighting': Alighting_update,
                   'boarding': Boarding_update,
                   'decision_time_stopped': Decision_stopped_updates,
                   'decision_time_motion': Decision_motion_updates,
                   'decision_time_charging': Decision_time_charging}

        return options[current_event]()

    def computeCost_R(self, ta=0, ta_prime=0):
        '''
            ta is the action time when the action is selected
            ta_prime is the time when the maximizing action selected


        Algorithm:

        - at time t_prime, look all the passengers and get the passengers waiting times at the end of t_prime
        who waited between **** ta and ta_prime ***

                lst = [(j, k) for j in s1 for k in s2]
                p = [stop.passengers[i].waiting_duration for stop in self.env.stop
                     for i in range(len(stop.passengers))]


        :param ta:
        :param ta_prime:
        :return: R
        '''

        # do after computing R
        # - reset the incremental cost deltaR
        self._cost_increment = 0

        R = [stop.passengers[i].waiting_cost(ta, ta_prime, type='stop') for (_, stop) in self._env.stops.items()
             for i in range(len(stop.passengers)) if (stop.passengers and stop.passengers[i].arrival_time <= int(ta))]

        deltaR = self.computeCost_deltaR(ta, ta_prime)
        # deltaR = 0

        Jsum = sum(R)

        # If condition comes from the fact that, we need the waiting time of the all passengers
        # who was already waited at the time of starting action time until the end of the next action
        # During this episode A NEW EVENT MAY OCCUR, these events are handled in the deltaR
        # cart incremantal discounted cost

        # if not Jsum:
        #     Jsum = 0.0

        # do after computing R
        # - reset the incremental cost deltaR

        self._total_discounted_cost = Jsum + deltaR
        # self._total_discounted_cost = (Jsum + deltaR) * self._dt_avg_occupancy

        # Update discounted cost
        # self._total_discounted_cost += self._total_discounted_cost
        #
        # if self._total_discounted_cost > 2:
        #     self._total_discounted_cost = 2

        if self._total_discounted_cost == 0:
            self._total_discounted_cost = 0
        #
        else:
            # total_cost = 1 / total_cost
            self._total_discounted_cost = np.log10(self._total_discounted_cost)
            # self._total_discounted_cost = -1 / self._total_discounted_cost
            # self._total_discounted_cost = np.log(self._total_discounted_cost)/10

        # if  self._dt_avg_occupancy !=0:
        #     self._total_discounted_cost  /= self._dt_avg_occupancy

        return self._total_discounted_cost

    def computeCost_deltaR(self, ta=None, ta_prime=None):
        '''
            The aim of computing the deltaR is not to miss interval events
            Passengers starts to wait after ta
            Passengers waiting ends before ta_prime

            WHERE TO LOOK
            ** Passenger start to wait at the stops after ta and they are still at the stops
            ** Passengers ceased to wait after ta, and they are either in one of the cars
               or they are put in the stop's passenger arrived list

        '''

        deltaR = 0.0

        # look at the stops if there are passengers arrived between ta - tprime
        Crstops = [stop.passengers[i] for (_, stop) in self._env.stops.items()
                   for i in range(len(stop.passengers)) if (stop.passengers and
                                                            stop.passengers[i].arrival_time > int(ta) and
                                                            stop.passengers[i].arrival_time <= int(ta_prime))]

        if Crstops:
            Rstops = [passenger.waiting_cost(ta=ta, ta_prime=ta_prime, type='arrival') for passenger in Crstops]
            deltaR += sum(Rstops)

        # look at if there are passengers in the cart or the passenger arrived list of the stops
        # these passengers stopped to wait, ts is stopping time

        Crincart = [cart.passengers[i] for (_, cart) in self._env.carts.items()
                    for i in range(len(cart.passengers)) if (cart.passengers and
                                                             cart.passengers[i].arrival_time >= int(ta) and
                                                             cart.cartID != self.cartID)]

        if Crincart:
            Rincart = [passenger.waiting_cost(ta=ta, ta_prime=ta_prime, type='picked-by') for passenger in Crincart]
            deltaR += sum(Rincart)

        # there is no need to look at the passengers who got-off NO THE IS NEED
        # if passenger started to wait between event and get-in then get-off

        Crdelivered = [stop._passengers_got_off_here[i] for (_, stop) in self._env.stops.items()
                       for i in range(len(stop._passengers_got_off_here)) if (stop._passengers_got_off_here and
                                                                              stop._passengers_got_off_here[
                                                                                  i].arrival_time > int(ta))]

        if Crdelivered:
            Rdelivered = [passenger.waiting_cost(ta=ta, ta_prime=ta_prime, type='delivered') for passenger in
                          Crdelivered]
            deltaR += sum(Rdelivered)

        return deltaR

    def battery_charge_cost(self, ta_c, ta_prime_c):
        '''
            Depleted Charge wp0 - Charge already depleted and this will increase during the epoch
            Charge difference measured charge --> tau_ch minus inititial charge tau_0
            ta - ta_prime = delta charge per action pair
        '''

        scaler = 20  # % percent
        beta = self._env._beta
        wp0 = self._charge_used_cumsum * scaler

        wp1 = wp0 + np.absolute(ta_c - ta_prime_c) * scaler

        ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
        cost0 = np.exp(-beta * wp0) * ctau0

        ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
        cost1 = np.exp(-beta * wp1) * ctau1

        total_cost = cost0 - cost1

        if total_cost == 0:
            total_cost = 0
        else:
            total_cost = np.log10(total_cost)

        return total_cost
