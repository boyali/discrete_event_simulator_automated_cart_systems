from itertools import count

import numpy as np


class Passengers(object):
    """Passenger Object"""
    __num_of_passengers = count(0)
    __beta = 0.0002

    def __init__(self, cart_stop, arrival_time=0,
                 current_state='waiting at the stop', passengerType='single', destination=None):
        self._passengerID = next(self.__num_of_passengers)
        self._passengerType = passengerType  # couple, triple, family
        self._arrival_time = arrival_time
        self._getin_time = None
        self._getoff_time = None

        self._waiting_duration = 0
        self._current_state = current_state  # ['going home', 'in cart']
        self._cart_stop = cart_stop

        self._destination = destination

        if cart_stop is not None:
            self._beta = self.cart_stop._beta  # for waiting cost, discount factor
        else:
            self._beta = 0.0002

        self._waiting_cost = None

    @property
    def origin(self):
        return self._cart_stop.stopID

    @property
    def destination(self):
        return self._destination

    @destination.setter
    def destination(self, value):
        self._destination = value

    @property
    def passengerType(self):
        return self._passengerType

    @passengerType.setter
    def passengerType(self, value):
        self._passengerType = value

    @property
    def passengerID(self):
        return self._passengerID

    @property
    def cart_stop(self):
        return self._cart_stop

    @property
    def getin_time(self):
        return self._getin_time

    @getin_time.setter
    def getin_time(self, value):
        self._getin_time = value

    @property
    def getoff_time(self):
        return self._getoff_time

    @getoff_time.setter
    def getoff_time(self, value):
        self._getoff_time = value

    @property
    def arrival_time(self):
        return self._arrival_time

    @arrival_time.setter
    def arrival_time(self, value):
        self._arrival_time = value

    def waiting_duration(self, sim_time):
        # waiting = None
        #
        # if self.getin_time:
        #     waiting = self.getin_time - self.arrival_time
        # else:
        #     waiting = sim_time - self.arrival_time

        waiting = sim_time - self.arrival_time
        return waiting

    # def waiting_cost(self, ta=None, ta_prime=None, type='stop'):
    #     '''
    #         type = stop for R, arrival for inter-decision passenger arrival
    #                            picked-by for picked passengers
    #                            delivered for delivered passengers
    #
    #         When an action is taken (tx), this value is called
    #         After that, we have a look at the new state, and take the
    #         maximizing action (ty) and call this value again.
    #
    #     ** THE COST at t0 and t1 is only computed passengers waited from tx to ty **
    #
    #         action x (tx), chosen, --> New state --> Maximizing action y (ty)
    #
    #         if new event occurs between these to time values
    #         cost = R is updated by computing deltaR for each car
    #
    #         NEW EVENTS;
    #
    #                     * a passenger comes between the actions and picked up
    #                       by the other cart
    #
    #                     * passenger gets-in the car
    #                     * passenger gets-off the car
    #                     * passenger arrives
    #                     * decision is made
    #
    #                     R + dR of this new event
    #
    #                     t0 is the time of last event
    #                     t1 is the time of the current event
    #
    #         BARTO PAPER THE INTEGRAL IS missing detail
    #     '''
    #     cost0 = 0
    #     cost1 = 0
    #     normalizing_constant = 1
    #     beta = self._beta
    #
    #     if type == 'stop':
    #         wp0 = self.waiting_duration(ta_prime)
    #         wp1 = wp0 + ta_prime - ta
    #
    #         ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
    #         cost0 = np.exp(-beta * wp0) * ctau0 * normalizing_constant
    #
    #         ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
    #         cost1 = np.exp(-beta * wp1) * ctau1 * normalizing_constant
    #
    #     elif type == 'arrival':
    #         '''
    #             first event is passenger arrival t0
    #             second event is decision time ta_prime - still waiting
    #             since time ta
    #         '''
    #         wp0 = 0  # waiting time at the arrival
    #         wp1 = wp0 + ta_prime - self.arrival_time  # waiting time until the ta_prime
    #
    #         ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
    #         cost0 = np.exp(-beta * (self.arrival_time - ta)) * ctau0 * normalizing_constant
    #
    #         ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
    #         cost1 = np.exp(-beta * (ta_prime - ta)) * ctau1 * normalizing_constant
    #
    #
    #     elif type == 'picked-by':
    #         '''
    #             first event is passenger arrival t0
    #             second event is get-in time - ceased to wait
    #             since time ta
    #         '''
    #         wp0 = 0  # waiting time at the arrival
    #         wp1 = wp0 + self.getin_time - self.arrival_time  # waiting time until get-in
    #
    #         ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
    #         cost0 = np.exp(-beta * (self.arrival_time - ta)) * ctau0 * normalizing_constant
    #
    #         ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
    #         cost1 = np.exp(-beta * (self.getin_time - ta)) * ctau1 * normalizing_constant
    #
    #     elif type == 'delivered':
    #         '''
    #             first event is passenger arrival t0
    #             second event is get-in time - ceased to wait
    #             since time ta
    #         '''
    #         wp0 = 0  # waiting time at the arrival
    #         wp1 = wp0 + self.getin_time - self.arrival_time  # waiting time until get-in
    #
    #         ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
    #         cost0 = np.exp(-beta * (self.arrival_time - ta)) * ctau0 * normalizing_constant
    #
    #         ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
    #         cost1 = np.exp(-beta * (self.getin_time - ta)) * ctau1 * normalizing_constant
    #
    #     elif type == 'static':
    #         ''' Given waiting time how much is the cost, I added this as passenger independent func'''
    #         wp0 = 0
    #         wp1 = wp0 + ta_prime - ta
    #
    #         ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
    #         cost0 = np.exp(-beta * wp0) * ctau0 * normalizing_constant
    #
    #         ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
    #         cost1 = np.exp(-beta * wp1) * ctau1 * normalizing_constant
    #
    #     total_cost = cost0 - cost1
    #     return total_cost

    def waiting_cost(self, ta=None, ta_prime=None, type='stop'):
        '''
            type = stop for R, arrival for inter-decision passenger arrival
                               picked-by for picked passengers
                               delivered for delivered passengers

            When an action is taken (tx), this value is called
            After that, we have a look at the new state, and take the
            maximizing action (ty) and call this value again.

        ** THE COST at t0 and t1 is only computed passengers waited from tx to ty **

            action x (tx), chosen, --> New state --> Maximizing action y (ty)

            if new event occurs between these to time values
            cost = R is updated by computing deltaR for each car

            NEW EVENTS;

                        * a passenger comes between the actions and picked up
                          by the other cart

                        * passenger gets-in the car
                        * passenger gets-off the car
                        * passenger arrives
                        * decision is made

                        R + dR of this new event

                        t0 is the time of last event
                        t1 is the time of the current event

            BARTO PAPER THE INTEGRAL IS missing detail
        '''
        cost0 = 0
        cost1 = 0
        normalizing_constant = 1e-5
        beta = self._beta

        if type == 'stop':
            wp0 = self.waiting_duration(ta_prime) / 60
            wp1 = wp0 + (ta_prime - ta) / 60

            ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
            cost0 = np.exp(-beta * wp0) * ctau0 * normalizing_constant

            ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
            cost1 = np.exp(-beta * wp1) * ctau1 * normalizing_constant

        elif type == 'arrival':
            '''
                first event is passenger arrival t0
                second event is decision time ta_prime - still waiting
                since time ta
            '''
            wp0 = 0  # waiting time at the arrival
            wp1 = (ta_prime - self.arrival_time) / 60  # waiting time until the ta_prime

            ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
            cost0 = np.exp(-beta * wp0) * ctau0 * normalizing_constant

            ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
            cost1 = np.exp(-beta * wp1) * ctau1 * normalizing_constant


        elif type == 'picked-by':
            '''
                first event is passenger arrival t0
                second event is get-in time - ceased to wait
                since time ta
            '''
            wp0 = self.waiting_duration(ta) / 60  # waiting time at the arrival
            wp1 = wp0 + (self.getin_time - ta) / 60  # waiting time until get-in

            ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
            cost0 = np.exp(-beta * wp0) * ctau0 * normalizing_constant

            ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
            cost1 = np.exp(-beta *wp1) * ctau1 * normalizing_constant

        elif type == 'delivered':
            '''
                first event is passenger arrival t0
                second event is get-in time - ceased to wait
                since time ta
            '''
            wp0 = 0  # waiting time at the arrival
            wp1 = (wp0 + self.getin_time - self.arrival_time) / 60  # waiting time until get-in

            ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
            cost0 = np.exp(-beta * wp0) * ctau0 * normalizing_constant

            ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
            cost1 = np.exp(-beta * wp1) * ctau1 * normalizing_constant

        elif type == 'static':
            ''' Given waiting time how much is the cost, I added this as passenger independent func'''
            wp0 = 0
            wp1 = (wp0 + ta_prime - ta) / 60

            ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
            cost0 = np.exp(-beta * wp0) * ctau0 * normalizing_constant

            ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
            cost1 = np.exp(-beta * wp1) * ctau1 * normalizing_constant

        total_cost = cost0 - cost1
        return total_cost

    @classmethod
    def waiting_cost_cls(cls, beta, ta, ta_prime):
        # beta = cls._beta
        normalizing_constant = 1.0

        wp0 = 0
        wp1 = (wp0 + ta_prime - ta) / 60

        ctau0 = (2 / beta ** 3 + 2 * wp0 / beta ** 2 + (wp0 ** 2) / beta)
        cost0 = np.exp(-beta * wp0) * ctau0 * normalizing_constant

        ctau1 = (2 / beta ** 3 + 2 * wp1 / beta ** 2 + (wp1 ** 2) / beta)
        cost1 = np.exp(-beta * wp1) * ctau1 * normalizing_constant

        total_cost = cost0 - cost1

        return np.log10(total_cost) / 10
