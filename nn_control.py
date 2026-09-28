import time

import numpy as np
import torch
import mujoco
import mujoco.viewer

from network import CartPoleNet


# --------------------------------------------------
# Load MuJoCo
# --------------------------------------------------

model = mujoco.MjModel.from_xml_path("cartpole.xml")
data = mujoco.MjData(model)


# --------------------------------------------------
# Load neural network
# --------------------------------------------------

net = CartPoleNet()
net.eval()


# --------------------------------------------------
# Starting state
# --------------------------------------------------

data.qpos[0] = 0.0       # cart position
data.qpos[1] = 0.05      # pole angle

data.qvel[0] = 0.0       # cart velocity
data.qvel[1] = 0.0       # pole angular velocity

mujoco.mj_forward(model, data)


# --------------------------------------------------
# NN → force
# --------------------------------------------------

def get_force():
    x = data.qpos[0]
    theta = data.qpos[1]

    x_dot = data.qvel[0]
    theta_dot = data.qvel[1]

    state = torch.tensor(
        [[x, theta, x_dot, theta_dot]],
        dtype=torch.float32
    )

    with torch.no_grad():
        force = net(state).item()

    return force


# --------------------------------------------------
# Control loop
# --------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Neural network controller running...")
    print("The network is currently UNTRAINED.")

    while viewer.is_running():

        start = time.time()

        # Get force from neural network
        force = get_force()

        # Send force to MuJoCo
        data.ctrl[0] = force

        # Advance physics
        mujoco.mj_step(model, data)

        # Update visualization
        viewer.sync()

        # Maintain approximately real-time simulation
        elapsed = time.time() - start
        time.sleep(
            max(0.0, model.opt.timestep - elapsed)
        )