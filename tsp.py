import random
import math
import itertools
import logging
import numpy as np
import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from sklearn import datasets
from scipy.misc import imread
import time
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.distance import vincenty
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TravelingTourist(object):
    """" Applies Simulated Annealing to a traveling salesman problem
    """
    def __init__(self):
        self.temperature_initial = 1
        self.temperature_terminal = 0.0001
        self.cooling_rate = 0.99
        self.X = None
        self.y = None
        self.trip = None
        self.trip_neighbor = None
        self.state_evolution = []
        self.distances = None
        self.latlong = None

    def define_problem(self, cities, round_trip=True, shuffle=False):
        """ Define the problem
        :param cities: A list of city names as strings
        :param round_trip: True if Start==End
        :param shuffle: choose a random permutation
        """
        if round_trip:
            self.trip = cities
            if shuffle:
                random.shuffle(self.trip)

    @staticmethod
    def retrieve_latlong_for_city(city):
        """ Retrieve the distances for all given cities
        Output as matrix
        """
        geolocator = Nominatim()
        c = geolocator.geocode(city)
        return [c.longitude, c.latitude]

    def retrieve_distance_between_cities(self, city1, city2, unit='km'):
        geolocator = Nominatim()
        c1 = geolocator.geocode(city1)
        c2 = geolocator.geocode(city2)
        if unit == 'km':
            return vincenty(c1.point, c2.point).km
        if unit == 'miles':
            return vincenty(c1.point, c2.point).miles

    def make_latlong(self, load_from_file=None):
        """ Generate a data frame containing the previously defined cities with their respective longitude
        and latitude
        """
        if load_from_file:
            self.distances = pd.read_csv(load_from_file)
        else:
            assert self.trip is not None, 'The problem has not been defined. Run define_problem().'
            latlong = pd.DataFrame(self.trip)
            latlong[1] = latlong[0].apply(t.retrieve_latlong_for_city)
            latlong[['Longitude', 'Latitude']] = pd.DataFrame([x for x in latlong[1]])
            latlong.index = latlong[0]
            del latlong[0]
            del latlong[1]
            self.latlong = latlong

    def make_distances(self, load_from_file=None):
        if load_from_file:
            self.distances = pd.read_csv(load_from_file)
        else:
            assert self.trip is not None, 'The problem has not been defined. Run define_problem().'
            distances = pd.DataFrame(np.zeros([len(self.trip), len(self.trip)]))
            distances = distances.replace(0, -1)
            distances.columns = self.trip
            distances.index = self.trip
            self.distances = distances
            for city in self.trip:
                self.distances.loc[city].loc[city] = 0
            for city_tuple in itertools.combinations(self.trip, 2):
                logging.debug(city_tuple)
                if self.distances.loc[city_tuple[0]].loc[city_tuple[1]] < 0 \
                        or self.distances.loc[city_tuple[1]].loc[city_tuple[0]] < 0:
                    try:
                        dist = self.retrieve_distance_between_cities(city_tuple[0], city_tuple[1])
                    except GeocoderUnavailable:
                        logging.warning('Geopy Geocoder was not available.')
                        dist = np.nan
                    except HTTPError:
                        logging.warning('Too many requests for Geopy. Waiting a few seconds')
                        time.sleep(30)
                        dist = np.nan
                    self.distances.loc[city_tuple[0]].loc[city_tuple[1]] = dist
                    self.distances.loc[city_tuple[1]].loc[city_tuple[0]] = dist

    def energy(self, trip):
        """ Energy function of the annealing progress
        Calculate the total distance of the journey with the current order of cities
        """
        return self.journey_distance(trip)

    def journey_distance(self, trip, round_trip=True):
        """ Take a list with cities and calculate the corresponding total distance of the round trip """
        assert self.distances is not None, 'No distance matrix loaded'
        distances = []
        if round_trip:
            for i in range(len(trip)-1):
                distances.append(self.distances.loc[trip[i]].loc[trip[i+1]])
        return np.array(distances).sum()

    @staticmethod
    def acceptance_probability(energy_old, energy_new, T):
        """ Compare energy states with respect to current temperature"""
        return np.exp((energy_old-energy_new)/T)

    def generate_neighbor(self):
        """ Switch two random cities"""
        trip_neighbor = self.trip.copy()
        sw = random.sample(range(len(self.trip)), 2)
        trip_neighbor[sw[1]], trip_neighbor[sw[0]] = trip_neighbor[sw[0]], trip_neighbor[sw[1]]
        self.trip_neighbor = trip_neighbor

    def anneal(self):
        """ Simulated Annealing
        Work through the classical process of
        """
        assert self.distances is not None, 'No distance file present'
        assert self.trip is not None, 'No problem was defined. Run self.define_problem().'
        self.state_evolution.append(self.trip)  # state_evolution only serves to follow up on the states chosen
        T = self.temperature_initial
        while T > self.temperature_terminal:
            for n in range(len(self.trip)):
                self.generate_neighbor()
                energy_old = self.energy(self.trip)
                energy_new = self.energy(self.trip_neighbor)
                if self.acceptance_probability(energy_old, energy_new, T) > random.random():
                    self.trip = self.trip_neighbor  # the alternative state was accepted
                    self.state_evolution.append(self.trip)
            T *= self.cooling_rate
        return self.trip

    # VISUALIZATION
    def visualize_one_state(self, state):
        """ Draw one state as map"""
        if self.latlong is None:
            self.make_latlong()
        # City grid
        self.latlong.plot.scatter('Longitude', 'Latitude', zorder=1)
        for n in range(len(state)-1):
            self.draw_line_between(state[n], state[n+1])
        for city in self.trip:
            self.annotate_city(city)
        return plt

    def draw_line_between(self,city1, city2):
        """ Draw a line between two cities

        :param city1: As string
        :param city2: As string
        """
        plt.plot([self.latlong.loc[city1].loc['Longitude'], self.latlong.loc[city2].loc['Longitude']],
                 [self.latlong.loc[city1].loc['Latitude'], self.latlong.loc[city2].loc['Latitude']])
        return plt

    def visualize_all_states(self, folder=''):
        """ Save all different states as png"""
        i = 0
        for state in self.state_evolution:
            plt = self.visualize_one_state(state)
            plt.savefig(folder+'tsp_' + str(i))
            i += 1

    def annotate_city(self, city):
        """ Add city labels to the visualization"""
        x_lab = self.latlong.loc[city].loc['Long'] + 0.5
        y_lab = self.latlong.loc[city].loc['Lat'] + 0.5
        return plt.annotate(city, xy=(x_lab, y_lab), xytext=(x_lab, y_lab))



t = TravelingTourist()
t.define_problem(['Barcelona', 'Belgrade', 'Berlin', 'Brussels', 'Bucharest', 'Budapest', 'Copenhagen', 'Dublin',
                  'Paris', 'Lisbon', 'Madrid', 'Cologne', 'Bern', 'Amsterdam', 'London', 'Manchester', 'Oslo',
                  'Rome', 'Sicily', 'Montpellier', 'Zurich', 'Vienna', 'Athina'])


# s.anneal()
# s.visualize_one_state(s.trip)
#
#
# plt.show()
