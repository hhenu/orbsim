"""
Ballistic simulation for some kind of a rocket or missile or some shit. Takes into
account the changing atmospheric properties as a function of altitude, and the Earth's
curvature is also taken into account (at some point maybe). The projectile can have
some thrust, which can also be time-dependent. The depletion of the mass during
flight is also taken into account, on some crude level of detail.
"""

import atmos
import utils
import constants
import functools
import numpy as np
import matplotlib.pyplot as plt

from solvers import rk4
from typing import Callable
from rocket import Rocket
from flightdata import FlightData
from drag_correlations import HaiderLevenspiel, HolzerSommerfeld


def flight_angle(t: int | float) -> int | float:
    """
    Returns the angle that the rocket is pointing towards as a function of time,
    90 degrees is directly away from the Earth, 0 degrees is parallel to ground.
    :param t: Flight time [s]
    :return: 
    """
    if t < 30:
        return 90
    t -= 30
    turning_rate = .25  # [deg/s]
    return max(10, 90 - turning_rate * t)


def f9_mass_fun(t: int | float, payload: int | float) -> int | float:
    """
    Returns the mass of a Spacex Falcon 9 rocket as a function
    of flight time

    Specs from from https://en.wikipedia.org/wiki/Falcon_9
    :param t: Flight time [s]
    :param payload: Payload mass [kg]
    :return:
    """
    if t < 0:
        raise ValueError(f"Time must be positive, now got {t}")
    stage1_empty_mass = 25600  # [kg]
    stage1_fuel_mass = 395700
    stage1_burn_time = 162  # [s]
    stage2_empty_mass = 3900
    stage2_fuel_mass = 92670
    stage2_burn_time = 397
    total = stage1_empty_mass + stage1_fuel_mass + stage2_empty_mass + \
        stage2_fuel_mass + payload
    # Stage 1
    if t <= stage1_burn_time:
        return total - t * (stage1_fuel_mass / stage1_burn_time)
    # Stage 2, stage 1 has been dropped
    elif stage1_burn_time < t <= (stage1_burn_time + stage2_burn_time):
        t -= stage1_burn_time
        loss_rate = stage2_fuel_mass / stage2_burn_time
        return (stage2_empty_mass + stage2_fuel_mass + payload) - t * loss_rate
    else:
        return stage2_empty_mass + payload


def f9_thrust_fun(t: int | float, angle_fun: Callable) -> np.ndarray:
    """
    Returns the thrust of a Spacex Falcon 9 rocket as a function of flight time

    Specs from from https://en.wikipedia.org/wiki/Falcon_9
    :param t: Flight time [s]
    :param angle_fun: Function that returns the rocket direction as a function of
    time
    :return:
    """
    if t < 0:
        raise ValueError(f"Time must be positive, now got {t}")
    stage1_burn_time = 162  # [s]
    stage2_burn_time = 397
    stage1_thrust = 7607e3  # At sea level [N]
    stage2_thrust = 981e3  # [N]
    angle = angle_fun(t)
    direc = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    if t <= stage1_burn_time:
        return direc * stage1_thrust
    elif stage1_burn_time < t <= (stage1_burn_time + stage2_burn_time):
        return direc * stage2_thrust
    else:
        return np.array([0, 0])


def dist_to_earth(point: np.ndarray) -> int | float:
    """
    Finds the shortest distance from the given point to Earth
    :param point: 
    :return:
    """
    # Earth is here assumed to be at (0, 0)
    dist = utils.vec_len(point)
    return dist - constants.r_e


def _inverse_clamp(val: int | float, low: int | float, high: int | float) -> int | float:
    """
    Makes sure that the given value is outside the range [low, high]
    :param val:
    :param low:
    :param high:
    :return:
    """
    mid = (high + low) / 2
    if low <= val <= mid:
        return low
    if mid <= val <= high:
        return high
    return val


def prandtl_glauert(c0: int | float, v: int | float, t: int | float) -> int | float:
    """
    Based on

    https://pages.vassar.edu/magnes/2019/05/12/computational-simulation-
    of-rocket-trajectories
    
    :param c0: The "initial" drag coefficient, which is to be modified
    by the model
    :param speed: Speed of the rocket [m/s]
    :param t: Temperature [K]
    :return:
    """
    epsilon = 1e-4
    c_sound = np.sqrt(constants.gamma * constants.r_gas * t)
    mach = v / c_sound
    # There's a singularity with mach == 1
    mach = _inverse_clamp(val=mach, low=1 - epsilon, high=1 + epsilon)
    if mach < 1:
        beta = np.sqrt(1 - mach * mach)
    else:
        beta = np.sqrt(mach * mach - 1)
    return c0 / beta


def _diff_eq(y0: np.ndarray, t: int | float, m: int | float, drag: int | float,
             gravity: int | float, thrust: int | float) -> np.ndarray:
    """
    Differential equation for an object following ballistic trajectory
    :param y0: Initial conditions as a vector of vectors [position, velocity]
    :param t: Time [s] (unused)
    :param m: Mass of the projectile [kg]
    :param drag: Drag force affecting the projectile [N]
    :param gracity: Gravitational force affecting the projectile [N]
    :param thrust: Thrust force [N]
    :return:
    """
    _ = t # To mark the param unused 
    _, v = y0
    dydt = [v, (drag + gravity + thrust) / m]
    return np.array(dydt)


def _solve(rocket: Rocket, solver: Callable, dt: float,
           time: int | float) -> FlightData:
    """
    :param rocket: A Rocket object
    :param solver: The solver used to solve the differential equation of motion
    :param dt: Timestep size [s]
    :param time: Simulation time [s]
    :return: A FlightData object containing the simulation results
    """
    steps = int(time / dt) + 1
    pos = np.zeros(shape=(steps, 2), dtype=float)
    vel = np.zeros(shape=(steps, 2), dtype=float)
    cds = np.zeros(shape=(steps, ), dtype=float)
    res = np.zeros(shape=(steps, ), dtype=float)
    drag = np.zeros(shape=(steps, ), dtype=float)
    alt = np.zeros(shape=(steps, ), dtype=float)
    pos[0] = rocket.p0
    cds[0] = rocket.get_cd(re=0)
    res[0] = 0
    alt[0] = dist_to_earth(point=rocket.p0)
    for n in range(1, steps):
        t = dt * n 
        mass = rocket.mass_fun(t)
        thrust = rocket.thrust_fun(t)
        grav_f = utils.grav_force(pos=pos[n - 1], m=mass)
        temp, _, rho = atmos.get_atmos_data(h=alt[n - 1])
        re = utils.reynolds(rocket.size, rho=rho, temp=temp, vel=vel[n - 1])
        c_d = rocket.get_cd(re=re)
        c_d = prandtl_glauert(c0=c_d, v=utils.vec_len(vel[n - 1]), t=temp)
        drag_f = utils.drag_force(rho=rho, area=rocket.proj_area, c_d=c_d, vel=vel[n - 1])
        n_pos, n_vel = solver(diff_eq=_diff_eq, y0=np.vstack((pos[n - 1], vel[n - 1])), t=t,
                              dt=dt, m=mass, drag=drag_f, gravity=grav_f, thrust=thrust)
        pos[n] = n_pos
        vel[n] = n_vel
        cds[n] = c_d
        res[n] = re
        drag[n] = utils.vec_len(drag_f)
        alt[n] = dist_to_earth(point=pos[n])
        if alt[n] < 0:
            print(f"INFO: Simulation ended early due to height being < 0")
            break

    return FlightData(rocket=rocket, coords=pos[:n], vel=vel[:n], dt=dt, c_d=cds[:n],
                      re=res[:n], alt=alt[:n])


def simulate(*args: Rocket, solver: Callable, dt: int | float,
             time: int | float) -> list[FlightData]:
    """
    Calculates the trajectory of the projectile. The simulation is continued until
    the projectile hits the ground or the max_steps amount of timesteps are
    simulated.
    :param args: Arbitrary amount of Rocket objects to run the simulation for
    :param solver: The solver function used to solve the differential equation
    of motion
    :param dt: Timestep [s]
    :param time: The length of the simulation time [s]
    :return: List of FlightDataData objects containing the simulation results
    for all given args
    """
    data_objs = []
    for rocket in args:
        data_obj = _solve(rocket=rocket, solver=solver, dt=dt, time=time)
        data_objs.append(data_obj)
    return data_objs


def _plot_earth(points: int = 200) -> None:
    """
    :return:
    """
    angles = np.linspace(0, 2 * np.pi, points, endpoint=True)
    radius = constants.r_e * 1e-3
    x, y = np.cos(angles) * radius, np.sin(angles) * radius
    plt.plot(x, y, label="Earth")


def display_results(*args: FlightData) -> None:
    """
    Prints out and plots some key characteristics of the trajectory
    :param args:
    :return:
    """
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots() 
    fig3, ax3 = plt.subplots()
    for data_obj in args:
        tspan = np.linspace(0, data_obj.time, int(data_obj.time / data_obj.dt))
        print(f"Flight data for {data_obj.rocket}:")
        print(f"    Highest point: {np.max(data_obj.alt) * 1e-3:.3f} km")
        print(f"    Flight time: {data_obj.time:.3f} s")
        print()
        
        # Trajectory with Earth plotted too
        plt.figure(fig1)
        _plot_earth()
        plt.plot(data_obj.coords[:, 0] * 1e-3, data_obj.coords[:, 1] * 1e-3,
                 label=f"{data_obj.rocket} trajectory")

        # Altitude
        plt.figure(fig2)
        plt.plot(tspan, data_obj.alt * 1e-3, label=f"{data_obj.rocket}")
        
        # Velocities
        plt.figure(fig3)
        plt.plot(tspan, data_obj.vel[:, 0], label=f"X velocity, {data_obj.rocket}")
        plt.plot(tspan, data_obj.vel[:, 1], label=f"Y velocity, {data_obj.rocket}")
        plt.plot(tspan, data_obj.speed, linestyle="--",
                 label=f"Total velocity, {data_obj.rocket}")
        
    ax1.set_title("Trajectory")
    ax1.set_xlabel("x [km]")
    ax1.set_ylabel("y [km]")
    ax1.legend()
    ax1.grid()

    ax2.set_title("Altitude as a function of time")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Altitude [km]")
    ax2.legend()
    ax2.grid()

    ax3.set_title("Velocities and speed as a function of time")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Speed [m/s]")
    ax3.legend()
    ax3.grid()

    plt.show()


def main() -> None:
    # Initial conditions and shit
    p0 = np.array([0, constants.r_e])  # Initial position [m]
    d = 3.7  # Diameter [m]
    length = 69.8  # Total length [m]
    payload = 20e3  # [kg]
    dt = 1  # [s]
    time = 60 * 120 # [s]
    solver = rk4

    # Projectile creation with functions and stuff
    m_fun = functools.partial(f9_mass_fun, payload=payload)
    t_fun = functools.partial(f9_thrust_fun, angle_fun=flight_angle)
    # Note: HaiderLevenspiel probably overestimates drag coefficient,
    # HolzerSommerfeld might underestimate it
    drag_corr = HolzerSommerfeld 
    rocket = Rocket(p0=p0, d=d, length=length, mass_fun=m_fun, thrust_fun=t_fun,
                    drag_corr=drag_corr, name="Falcon 9")
   
    # Simulate
    flight_data = simulate(rocket, solver=solver, dt=dt, time=time)
    display_results(*flight_data)



if __name__ == "__main__":
    main()

