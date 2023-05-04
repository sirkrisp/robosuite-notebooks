import numpy as np
import robosuite as suite
import robosuite.macros

robosuite.macros.ENABLE_NUMBA = False

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

env.render()


#import bricks_env

#env = bricks_env.BricksEnv(robots="Panda", has_renderer=True)

#env.render()
