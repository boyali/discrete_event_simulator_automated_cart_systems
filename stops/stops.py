
import logging
import random
from collections import namedtuple
from itertools import count, groupby

import numpy as np

from env import process_time as dt
from passengers import passengers as psg
fname = "../stops/arrival_means"



class Stops(object):
    # _ARRIVAL_MEANS = np.random.randint(low=0, high=5, size=(8, 14)) # passengers per hour

    # SEED = 1  # 3438  ##
    # np.random.seed(SEED)
    # random.seed(SEED)

    _mean_gamma = 4
    _std_gamma = 3

    _gamma_alpha = (_mean_gamma / _std_gamma) ** 2
    _gamma_beta = _mean_gamma / _std_gamma ** 2

    # _ARRIVAL_MEANS = np.random.gamma(_gamma_alpha, 1 / _gamma_beta, size=(8, 15)).astype(int)  # passengers per hour

    _ARRIVAL_MEANS = np.load(fname + '.npy')
    # PASSNGR_TYPE_PROBS = [0.8, 0.1, 0.05, 0.05]  # Expected val = 1*0.75 + 0.1*2 + 0.1*3 + 0.05*4 - Compound Poisson
    PASSNGR_TYPE_PROBS = [0.7, 0.2, 0.05, 0.05]

    _num_of_stops = count(0)
    _passengerEvent = namedtuple('passengerEvent', 'time stopID event_type info ')

    _passengerTypes = {1: 'single', 2: 'couple', 3: 'triple', 4: 'family'}
    _passengerGroupNum = {'single': 1, 'couple': 2, 'triple': 3, 'family': 4}
    _stop_event_space = {0: 'stop_initialized', 1: 'passenger_arrives'}

    formatter = logging.Formatter('%(name)s:  %(message)s')

    # _event_types = {0: 'stop_process_initialized', 1: 'passenger_arrives'}

    def __init__(self, env, passengers=None):

        self._stopID = next(self._num_of_stops)
        self._passengers = []
        self._passengers_got_off_here = []

        self._arrival_mean = self._ARRIVAL_MEANS[self._stopID, :]
        self._env = env
        self._beta = env._beta
        self._call_assigned = False
        self._call_assigned_to_cart = ''

        if passengers is None:
            self._passengers = []
        else:
            self._passengers = list(passengers)

        # self if parked
        self._occupied_by_cart = False  # there is a cart stopping or not

        # log files
        open('../tests/logs/log_stop_stop{}.txt'.format(self.stopID), 'w').close()
        self.file_handler_stop = logging.FileHandler('../tests/logs/log_stop_stop{}.txt'.format(self.stopID))
        self.file_handler_stop.setFormatter(self.formatter)
        self.file_handler_stop.flush()

        self.logger_stop = logging.getLogger('../tests/logs/log_stop_stop{}.txt'.format(self.stopID))
        self.logger_stop.setLevel(logging.INFO)
        self.logger_stop.addHandler(self.file_handler_stop)

    @property
    def stopID(self):
        return 'S_' + str(self._stopID)

    @property
    def occupied_by_acart(self):
        return self._occupied_by_cart

    @occupied_by_acart.setter
    def occupied_by_acart(self, bvalue):
        self._occupied_by_cart = bvalue

    @property
    def passengers(self):
        return self._passengers

    def reset(self):

        self._passengers = []
        self._passengers_got_off_here = []

        self._call_assigned = False
        self._call_assigned_to_cart = ''
        self._occupied_by_cart = False  # there is a cart stopping or not
        Stops.reset_cls()

    @classmethod
    def reset_cls(cls):
        # cls._ARRIVAL_MEANS = np.random.randint(low=0, high=6, size=(8, 14))
        # cls._ARRIVAL_MEANS = np.random.gamma(cls._gamma_alpha, 1 / cls._gamma_beta, size=(8, 15)).astype(int)  # passengers per hour

        cls._ARRIVAL_MEANS = np.load(fname + '.npy')

    def passenger_process(self, time=0):

        print('Stop {}  passenger process started at time {}'.format(self._stopID, time))

        # put dummy initialization here
        # this .....V....here yields when initialized

        event = self._passengerEvent(time, self.stopID, 'stop_initialized',
                                     'passenger process initialized')
        time = yield event

        while True:
            if time >= 0:

                # _event_types = {0: 'stop_process_initialized', 1: 'passenger_arrives'}
                event = self._passengerEvent(time, self.stopID, 'passenger_arrives', 'passenger generated')
                time = yield event

            else:
                print('Process is terminated at the cart stop {} at time {}'.format(self.stopID, time))
                break

                # print('Process Closed')

    def generate_passenger(self, time):
        daytime = dt.time_handle(time_in_seconds=time)
        # _passengerTypes = {1:'single', 2:'couple', 3:'triple', 4:'family'}
        rand_type = np.random.choice([1, 2, 3, 4], 1, p=self.PASSNGR_TYPE_PROBS)

        passengerType = self._passengerTypes[rand_type[0]]

        destination = self.generate_destination()

        for i in range(rand_type[0]):
            new_passenger = psg.Passengers(self, time, passengerType=passengerType, destination=destination)

            self._passengers.append(new_passenger)

            self.logger_stop.info('Passenger {} {} is generated at time {}'
                                  ' at bus stop {} to destination stop {}'.format(new_passenger.passengerID,
                                                                                  new_passenger.passengerType,
                                                                                  daytime,
                                                                                  new_passenger.cart_stop.stopID,
                                                                                  new_passenger.destination))

    def generate_destination(self):
        bus_stops = list(range(8))
        destination = list(filter(lambda x: x != self._stopID, bus_stops))

        self.destination = destination

        return random.choice(destination)

    def passenger_groups(self):
        groups = [self._passengerGroupNum[self.passengers[i].passengerType]
                  for i in range(len(self.passengers))]

        return [list(j) for i, j in groupby(groups)]

    def rvExpoVariate(self, sim_time):
        'Generate next arrival time'
        time_ind = sim_time // 3600
        _random_time = 0
        arrival_mean = self._arrival_mean[time_ind]

        if arrival_mean != 0:

            try:
                _random_time = random.expovariate(arrival_mean / 3600)  # random seconds
                # print(int(_random_time))

            except OverflowError:
                print('Overflow error {}'.format(_random_time))

            else:
                return int(_random_time)  # mean arrival per minute

            if _random_time is None:
                raise ValueError('Random time is None')

        #
        #
        return _random_time

    @property
    def passengers_waiting(self):
        return len(self._passengers)
