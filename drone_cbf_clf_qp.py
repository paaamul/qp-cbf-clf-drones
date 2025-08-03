import os
import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cp

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
DEFAULT_DURATION_SEC = 30
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def clf_cbf_qp_setpoint(cur_pos, cur_vel, goal, obs_center, obs_radius, dt=0.05):
    """
    Computes the next safe target position using CLF-CBF-QP for a 2D drone.
    cur_pos: np.array([x, y])
    cur_vel: np.array([vx, vy])
    goal:    np.array([x_goal, y_goal])
    obs_center: np.array([x_obs, y_obs])
    obs_radius: float (safety margin)
    Returns: np.array([x_next, y_next])
    """
    c = 1.0   # CLF rate
    alpha = 2.0  # CBF rate
    max_speed = 7.0   # [m/s] max step per control (tune as needed)
    safety_margin = 0.3  # [m] on top of obstacle

    u = cp.Variable(2)  # desired velocity (delta per step)
    s = cp.Variable(1)
    
    ''' This produces a quadratic constraint for CLF
    # CLF: Move toward goal
    V = 0.5 * cp.sum_squares(cur_pos + u*dt - goal)
    LfV = (cur_pos - goal).T @ cur_vel
    LgV = (cur_pos - goal).T * dt
    clf_constraint = LfV + LgV @ u + c*V <= 0
    '''

    #Linearized CLF constraint
    clf_constraint = (cur_pos - goal) @ (u * dt)  <= s
    slack_penalty = 50.0
    #box_constraints = [u >= -max_speed, u <= max_speed]

    ''' This produces a quadratic constraint with h 
    
    dist_vec = cur_pos + u*dt - p_obs
    h = cp.sum_squares(dist_vec) - (obs_radius + safety_margin)**2
    Lfh = 2 * (cur_pos - p_obs).T @ cur_vel
    Lgh = 2 * (cur_pos - p_obs).T * dt
    cbf_constraint = Lfh + Lgh @ u + alpha*h >= 0
    
    '''
    ''' Linearized CBF constraint '''

    # CBF: Keep outside obstacle
    p_obs = obs_center
    h_cur = np.sum((cur_pos - p_obs)**2) - (obs_radius + safety_margin)**2
    grad_h = 2 * (cur_pos - p_obs)

    cbf_constraint = grad_h @ (u * dt) + alpha * h_cur >= 0

    #constraints = [clf_constraint,cbf_constraint,cp.norm(u) <= max_speed]

    constraints = [cbf_constraint, clf_constraint, u >= -max_speed, u <= max_speed, s >= 0]

    # QP: Minimize deviation from "go to goal" velocity
    v_goal = (goal - cur_pos)/max(dt, 1e-4)
    
    
    #Calculate detouring direction
    obs_vec = cur_pos - obs_center
    if np.linalg.norm(obs_vec) > 1e-6:
        detour_dir = np.array([obs_vec[1], -obs_vec[0]])
        detour_dir /= np.linalg.norm(detour_dir)
    else:
        detour_dir = np.array([1.0, 0.0])

    lateral_bias_weight = 1.0

    v_desired = v_goal + lateral_bias_weight * detour_dir
    

    objective = cp.Minimize(cp.sum_squares(u - v_desired) + slack_penalty * s)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    # Fallback
    if u.value is None:
        u_val = np.zeros(2)
    else:
        u_val = u.value

    return cur_pos + u_val * dt

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
    FINAL_XYZS = np.array([[0.0, 15.0, 1.0]])
    PERIOD = 10
    NUM_WP = control_freq_hz * PERIOD

    #Obstacle Parameters
    OBSTACLE_CENTER = np.array([0.0, 8.0, 1.0])
    OBSTACLE_BARRIER = 0.2  #"Safe" margin in meters (covers box and a bit more)
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
            #detour_points[detour_idx] += DETOUR_OFFSET  # Shift x to the right
            #print(detour_points[detour_idx]) #efficient way to obtain 0th element of that idx
        #waypoints = np.insert(waypoints, detour_idx, detour_points, axis=0)
        #print(f"Inserted detour at index {detour_idx}:", detour_points)
        #print(waypoints[:3, :])
    fig, ax = plt.subplots()

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
        edgecolor='k',
        facecolor='grey',
        alpha=0.5,
        label="Obstacle"
    )
    ax.add_patch(circ)

    safety_margin = 0.3
    danger_zone = patches.Circle(
    (obstacle_center[0], obstacle_center[1]),
    radius=OBSTACLE_BARRIER + safety_margin,
    linewidth=1.5,
    edgecolor='r',
    facecolor='none',  # no fill, just outline (or use 'r' with low alpha for a filled look)
    alpha=0.9,
    linestyle='--',
    label="CBF Danger Zone"
    )
    ax.add_patch(danger_zone)
    
    # Set plot limits a bit beyond start and goal
    margin = 6
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
            p.GEOM_SPHERE, radius=OBSTACLE_BARRIER, physicsClientId=PYB_CLIENT),
        basePosition=OBSTACLE_CENTER[:3],
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
        
        #if np.all(obs[0, :3] == FINAL_XYZS[0, :3]):
        #    terminated = True
        #    truncated = True

        #Hard coded arc offset to PID control
        '''
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=waypoints[wp_counters[j]],
                target_rpy=INIT_RPYS[j, :]
            )
        '''
        
        # CLF-CBF-QP control
        for j in range(num_drones):
            cur_pos = obs[j][:2] #x,y
            cur_pos3d = obs[j][:3]
            drone_traj.append(cur_pos3d.copy())
            cur_vel = obs[j][10:12] #vx,vy

            #goal = waypoints[wp_counters[j], :2]
            goal = FINAL_XYZS[0, :2]
            obs_center = OBSTACLE_CENTER[:2]
            obs_radius = OBSTACLE_BARRIER

            dt = 1.0/control_freq_hz
            #Obtain safe x,y position using CLF-CBF-QP
            safe_pos_2d = clf_cbf_qp_setpoint(cur_pos,
                                              cur_vel,
                                              goal,
                                              obs_center,
                                              obs_radius,
                                              dt=dt)
            
            #Maintain z position at original height
            safe_pos = np.array([safe_pos_2d[0], safe_pos_2d[1], INIT_XYZS[0,2]])

            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=safe_pos,
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
