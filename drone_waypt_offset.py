import os
import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #Initialize the simulation
    INIT_XYZS = np.array([[0.0, 0.0, 1.0]])
    INIT_RPYS = np.array([[0, 0, 0]])
    FINAL_XYZS = np.array([[0.0, 20.0, 1.0]])
    PERIOD = 10
    NUM_WP = control_freq_hz * PERIOD

    #Obstacle Parameters
    OBSTACLE_CENTER = np.array([0.0, 8.0, 1.0])
    OBSTACLE_BARRIER = 0.5  #"Safe" margin in meters (covers box and a bit more)
    DETOUR_OFFSET = 0.12     # How far to go around the box in x direction

    drone_traj = []
    
    #Generate waypoints and insert detour if needed
    waypoints = np.linspace(INIT_XYZS[0], FINAL_XYZS[0], NUM_WP)
    
    
    #Find index where path gets close to the obstacle
    dists_to_obstacle = np.linalg.norm(waypoints[:, :2] - OBSTACLE_CENTER[:2], axis=1)
    danger_idxs = np.where(dists_to_obstacle < (OBSTACLE_BARRIER))[0]
    print(danger_idxs)
    if len(danger_idxs) > 0:
        #Insert detour at the first dangerous index
        #detour_points = np.zeros((len(danger_idxs), 3))
        #detour_points = waypoints.copy()
        for detour_idx in danger_idxs:
            delta_y = waypoints[detour_idx, 1] - OBSTACLE_CENTER[1]
            x_on_barrier = np.sqrt(OBSTACLE_BARRIER**2 - (delta_y)**2)
            waypoints[detour_idx, 0] = x_on_barrier + DETOUR_OFFSET

    # Plot waypoints path
    #ax.plot(waypoints[:, 0], waypoints[:, 1], 'b-o', label="Waypoints Path")

    '''
    # Plot obstacle as a rectangle
    obstacle_center = OBSTACLE_CENTER
    #half_x = 0.1
    #half_y = 0.1
    circ = patches.Circle(
        (obstacle_center[0], obstacle_center[1]),
        #2*half_x,
        #2*half_y,
        radius=OBSTACLE_BARRIER,
        linewidth=1,
        edgecolor='r',
        facecolor='r',
        alpha=0.5,
        label="Obstacle"
    )
    ax.add_patch(circ)

    # Set plot limits a bit beyond start and goal
    margin = 3
    ax.set_xlim(INIT_XYZS[0,0] - margin, FINAL_XYZS[0,0] + margin)
    ax.set_ylim(INIT_XYZS[0,1] - margin, FINAL_XYZS[0,1] + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Waypoint Path with Obstacle and Detour')
    ax.legend()
    ax.grid(True)
    plt.show()
    '''

    wp_counters = np.array([0])

    #Create the environment
    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=user_debug_gui)

    PYB_CLIENT = env.getPyBulletClient()
    
    #Create Box Obstacle
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_SPHERE, radius=0.5, physicsClientId=PYB_CLIENT),
        basePosition=[0.0, 8.0, 1.0],
        physicsClientId=PYB_CLIENT)

    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab)

    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    action = np.zeros((num_drones, 4))
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)
        
        #Hardcoded arc offset to avoid the obstacle
        for j in range(num_drones):
            drone_traj.append(obs[j, :3])
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=waypoints[wp_counters[j]],
                target_rpy=INIT_RPYS[j, :]
            )

        for j in range(num_drones):
            if wp_counters[j] < (len(waypoints) - 1):
                wp_counters[j] += 1

        for j in range(num_drones):
            logger.log(
                drone=j,
                timestamp=i / env.CTRL_FREQ,
                state=obs[j],
                control=np.hstack([waypoints[wp_counters[j]], INIT_RPYS[j, :], np.zeros(6)])
            )

        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    drone_traj = np.array(drone_traj)

    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale.'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    # Plot trajectory
    ax.plot(drone_traj[:, 0], drone_traj[:, 1], drone_traj[:, 2], 'b-', label='Trajectory')

    #Plot Spherical Obstacle
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = OBSTACLE_CENTER[0] + (OBSTACLE_BARRIER) * np.cos(u) * np.sin(v)
    y = OBSTACLE_CENTER[1] + (OBSTACLE_BARRIER) * np.sin(u) * np.sin(v)
    z = OBSTACLE_CENTER[2] + (OBSTACLE_BARRIER) * np.cos(v)
    ax.plot_surface(x,y,z, color='r', alpha=0.6, label='Obstacle')
    

    # Labels and aspect
    set_axes_equal(ax)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Drone Trajectory with Obstacle')
    plt.legend()
    plt.show()
    
    #logger.save()
    #logger.save_as_csv("pid")
    if plot:
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int, help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool, help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool, help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool, help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int, help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
