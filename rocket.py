"""
Rocket object or sheit
"""

import numpy as np
from typing import Callable


class Rocket:
    """
    The rocket is here assumed to essentially be a shell shaped thing that
    has a changing mass and non-zero thrust.
    """
    def __init__(self, p0: np.ndarray, d: int | float, length: int | float,
                 mass_fun: Callable, thrust_fun: Callable,
                 drag_fun: Callable, name: str = None) -> None:
        """
        :param p0: Initial position [m]
        :param d: Diameter [m]
        :param length: Length [m]
        :param mass_fun: Function that describes the mass of the rocket
        as a function of time [kg]
        :param thrust_fun: Function that describes the thrust produced
        by the rocket as a function of time [N]
        :param drag_fun: Function that returns current drag
        :param name: Optional name for the projectile, will be used in the legend
        of the plots so that the projectile can be identified.
        :return:
        """
        self.p0 = p0
        self.d = d
        self.r = d / 2
        self.length = length
        self.mass_fun = mass_fun
        self.thrust_fun = thrust_fun
        self.drag_fun = drag_fun  # TODO: Design this function
        self.name = name
        self.size = d

    @property
    def proj_area(self) -> int | float:
        """
        The projected area of the rocket (assumed to be parallel to the flow)
        :return:
        """
        return np.pi * self.r * self.r

    @property
    def surf_area(self) -> int | float:
        """
        The surface area of the rocket (here assumed to be a cylinder)
        :return:
        """
        return self.proj_area * 2 + np.pi * self.d * self.length

    @property
    def volume(self) -> int | float:
        """
        Volume of a cube
        :return:
        """
        return self.proj_area * self.length

    def get_cd(self, re: int | float) -> int | float:
        """
        Calculates the drag coefficient based on the Reynolds number using
        the drag correlation attribute
        :param re: Reynolds number [-]
        :return:
        """
        return self.drag_fun(re)

    def get_mass(self, t: int | float) -> int | float:
        """
        Gets the mass at the given time
        :param t: Time [s]
        :return:
        """
        return self.mass_fun(t)

    def get_thrust(self, t: int | float) -> int | float:
        """
        Gets the thrust at the given 01:42
        :param t: Time [s]
        :return:
        """
        return self.thrust_fun(t)

    def __str__(self) -> str:
        """
        :return:
        """
        if self.name is None:
            return f"{self.__class__.__name__}"
        return self.name

