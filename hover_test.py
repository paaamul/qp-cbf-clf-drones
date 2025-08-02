import gymnasium
import PyFlyt.gym_envs
import time

env = gymnasium.make("PyFlyt/QuadX-Ball-In-Cup-v4", render_mode="human")
obs, info = env.reset()

termination=False
truncation=False

target_fps = 20
dt = 1.0/target_fps


eps_end = 1000
for _ in range(eps_end):
    env.render()
    action = env.action_space.sample()
    obs, reward, termination, truncation, info = env.step(action)
    #time.sleep(dt)

    if termination or truncation:
        env.close()
        #obs, info = env.reset()
        #termination = False
        #truncation = False

