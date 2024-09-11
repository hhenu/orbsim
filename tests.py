"""
Contains some kind of tests of the modelling functionality. Right now only
plots the atmospheric data so as to make it available for visual inspection.
"""

import atmos
import numpy as np
import matplotlib.pyplot as plt


def _plot(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str) -> None:
    """
    :param x:
    :param y:
    :param xlabel:
    :param ylabel:
    :return:
    """
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


def plot_data(h: np.ndarray) -> None:
    """
    :param h:
    :return:
    """
    t_arr = np.zeros(shape=h.shape)
    p_arr = np.zeros(shape=h.shape)
    rho_arr = np.zeros(shape=h.shape)
    vis_arr = np.zeros(shape=h.shape)
    for i, hv in enumerate(h):
        t, p, rho = atmos.get_atmos_data(h=hv)
        t_arr[i] = t
        p_arr[i] = p
        rho_arr[i] = rho
        vis_arr[i] = atmos.visc(rho=rho, temp=t)
    _plot(x=h, y=t_arr, xlabel="Altitude [m]", ylabel="Temperature [K]")
    _plot(x=h, y=p_arr, xlabel="Altitude [m]", ylabel="Pressure [Pa]")
    _plot(x=h, y=rho_arr, xlabel="Altitude [m]", ylabel="Density [kg/m^3]")
    # TODO: Add unit for viscosity
    _plot(x=h, y=vis_arr, xlabel="Altitude [m]", ylabel="Viscosity [someunit]")
    plt.show()


def main() -> None:
    a, b, dy = 0, 1500e3, 30
    altitudes = np.linspace(a, b, int(b - a / dy))
    plot_data(h=altitudes)


if __name__ == "__main__":
    main()

