import time
import numpy as np
import mujoco
import mujoco.viewer

PATH = "/Users/shashvatgarg/Desktop/Main/Shashvat_Garg/Projects/Neural Networks/Physics/Inverted Pendulum/cartpole.xml"


"""Control the cart-pole from Python. Only needs cartpole.xml in the same folder.

    python control.py            (macOS: mjpython control.py)
"""
"""Control the cart-pole from Python, with keyboard pushes you can apply live.

    python control.py            (macOS: mjpython Control.py)

KEYS (click the simulation window first so it has focus):
    LEFT / RIGHT arrow  ->  shove the CART left / right
    Z / X               ->  shove the POLE left / right
Mouse also works: double-click a body, then Ctrl + right-drag to pull on it.
"""
# ---- Load the model ----
model = mujoco.MjModel.from_xml_path(PATH)
data = mujoco.MjData(model)

# ---- Starting pose ----
data.qpos[0] = 0.0     # cart position (m)
data.qpos[1] = 0.05    # pole angle (rad), small tilt
mujoco.mj_forward(model, data)

# ---- Controller gains (one per state variable) ----
K_x, K_theta, K_x_dot, K_theta_dot = 3.11, 44.14, 5.60, 9.00
FORCE_LIMIT = 10.0     # newtons, matches ctrlrange in the XML

# ---- Disturbance settings (tweak these!) ----
CART_PUSH = 5.0        # newtons applied to the cart per key press
POLE_PUSH = 1.0        # newtons applied to the pole's middle per key press
PUSH_TIME = 0.1        # seconds each push lasts
# Rough limits for this policy: cart ~8+ N is fine, pole ~2 N knocks it over.

cart_id = model.body("cart").id
pole_id = model.body("pole").id

# Active pushes: body id -> (force, time when it ends). Written by the key
# callback, consumed by the main loop.
active_pushes = {}


def policy(x, theta, x_dot, theta_dot):
    """Takes the 4 state variables, returns a force on the cart (N)."""
    return (K_x * x
            + K_theta * theta
            + K_x_dot * x_dot
            + K_theta_dot * theta_dot)


def key_callback(keycode):
    """Called by the viewer whenever you press a key."""
    end = data.time + PUSH_TIME
    if keycode == 263:                 # left arrow
        active_pushes[cart_id] = (-CART_PUSH, end)
        print("push cart LEFT")
    elif keycode == 262:               # right arrow
        active_pushes[cart_id] = (+CART_PUSH, end)
        print("push cart RIGHT")
    elif keycode == ord("Z"):
        active_pushes[pole_id] = (-POLE_PUSH, end)
        print("push pole LEFT")
    elif keycode == ord("X"):
        active_pushes[pole_id] = (+POLE_PUSH, end)
        print("push pole RIGHT")


def apply_pushes():
    """Apply / release the external forces. These are disturbances from the
    'outside world': the policy does NOT know about them and they are not
    limited by FORCE_LIMIT."""
    for body_id, (force, end) in list(active_pushes.items()):
        if data.time < end:
            data.xfrc_applied[body_id, 0] = force      # force in +x direction
        else:
            data.xfrc_applied[body_id, 0] = 0.0
            del active_pushes[body_id]


# ---- Control loop ----
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    print(__doc__)
    while viewer.is_running():
        start = time.time()

        # 1. Read the state into named variables
        x         = data.qpos[0]   # cart position
        theta     = data.qpos[1]   # pole angle (0 = upright)
        x_dot     = data.qvel[0]   # cart velocity
        theta_dot = data.qvel[1]   # pole angular velocity

        # 2. Decide the force (your policy, unchanged)
        force = policy(x, theta, x_dot, theta_dot)
        if abs(x) > 2.2:
            print(
                f"x={x:.3f}, theta={theta:.3f}, "
                f"x_dot={x_dot:.3f}, theta_dot={theta_dot:.3f}, "
                f"force={force:.3f}"
            )

        # 3. Send it to the motor
        data.ctrl[0] = np.clip(force, -FORCE_LIMIT, FORCE_LIMIT)

        # 4. Apply any pushes you triggered from the keyboard
        apply_pushes()

        # 5. Advance physics and draw
        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))