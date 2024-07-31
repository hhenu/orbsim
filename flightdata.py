"""
File for a ProjectileData object which is the object that's actually
used in the simulations
"""

import utils
import numpy as np

from rocket import Rocket


class FlightData:
    def __init__(self, rocket: Rocket, coords: np.ndarray, vel: np.ndarray,
                 dt: int | float, c_d: np.ndarray = None, re: np.ndarray = None) -> None:
        """
        :param rocket: A Rocket object whose flight was simulated
        :param coords: Coordinates of the trajectory
        :param vel: Velocity of the projectile at different timepoints
        :param dt: Timestep used in the simulations [s]
        :param c_d: Drag coefficient at different timepoints
        :param re: Reynolds number at different timepoints
        :return:
        """
        self.rocket = rocket
        self.coords = coords
        self.vel = vel
        self.c_d = c_d
        self.re = re
        self.dt = dt
        x, y = coords[:, 0], coords[:, 1]
        self.x_dist = x[-1] - x[0]
        self.y_max = np.max(y)
        self.time = x.shape[0] * dt
        self.speed = utils.speed(vel=vel)
        try:
            self.ke = .5 * rocket.m * np.power(self.speed, 2)
        except AttributeError:
            pass

