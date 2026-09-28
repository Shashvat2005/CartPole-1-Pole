"""Single-link cart-pole in MuJoCo.

Run modes:
    python cartpole.py view      # interactive window (on macOS use: mjpython cartpole.py view)
    python cartpole.py video     # renders cartpole.mp4 (needs: pip install imageio imageio-ffmpeg)
    python cartpole.py test      # headless physics check, no graphics
"""
import sys
import numpy as np
import mujoco

XML = """
<mujoco model="cartpole">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>

  <worldbody>
    <light pos="0 -3 4" dir="0 0.6 -0.8"/>
    <geom name="floor" type="plane" size="6 1 0.1" pos="0 0 -0.5" rgba="0.85 0.85 0.85 1"/>

    <!-- rail the cart slides along -->
    <geom name="rail" type="capsule" fromto="-2.5 0 0 2.5 0 0" size="0.01" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>

    <!-- cart: slide joint along x -->
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2.4 2.4" damping="0.05"/>
      <geom name="cart_geom" type="box" size="0.2 0.1 0.08" rgba="0.2 0.4 0.8 1" mass="1"/>

      <!-- pole: hinge joint about y axis, starts pointing up (+z) -->
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.002"/>
        <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" rgba="0.9 0.3 0.2 1" mass="0.1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- force applied to the cart, clipped to +-10 N -->
    <motor name="push" joint="slider" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def make_model():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    return model, data


def get_state(data):
    """[cart position, pole angle, cart velocity, pole angular velocity]
    Pole angle is 0 when upright, positive tilts toward +x."""
    return np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])


def reset(model, data, angle_noise=0.05, rng=None):
    rng = rng or np.random.default_rng()
    mujoco.mj_resetData(model, data)
    data.qpos[1] = rng.uniform(-angle_noise, angle_noise)  # small initial tilt
    mujoco.mj_forward(model, data)


def controller(state):
    """Placeholder policy. Swap this for your neural net later.
    These gains come from an LQR design for this exact model
    (u = -K @ state). Note the x gain has the SAME sign as the angle gain:
    to bring a drifting cart back you first have to lean the pole toward home."""
    K = np.array([-3.11, -44.14, -5.60, -9.00])
    return float(-K @ state)


def run_test(seconds=5.0):
    model, data = make_model()
    reset(model, data, angle_noise=0.05, rng=np.random.default_rng(0))
    steps = int(seconds / model.opt.timestep)
    for i in range(steps):
        data.ctrl[0] = np.clip(controller(get_state(data)), -10, 10)
        mujoco.mj_step(model, data)
        if abs(data.qpos[1]) > 0.8:
            print(f"pole fell at t={data.time:.2f}s")
            return
    print(f"balanced for {seconds}s | final state: {np.round(get_state(data), 3)}")

    # compare: no control -> should fall
    reset(model, data, angle_noise=0.05, rng=np.random.default_rng(0))
    for i in range(steps):
        data.ctrl[0] = 0.0
        mujoco.mj_step(model, data)
        if abs(data.qpos[1]) > 0.8:
            print(f"with zero control the pole falls at t={data.time:.2f}s (as expected)")
            return


def run_viewer():
    import mujoco.viewer
    import time
    model, data = make_model()
    reset(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            data.ctrl[0] = np.clip(controller(get_state(data)), -10, 10)
            mujoco.mj_step(model, data)
            viewer.sync()
            # keep it real time
            time.sleep(max(0, model.opt.timestep - (time.time() - t0)))


def run_video(path="cartpole.mp4", seconds=6.0, fps=30):
    import imageio
    model, data = make_model()
    reset(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.3]
    cam.distance = 4.0
    cam.azimuth = 90      # look along the y-axis so the cart moves left-right
    cam.elevation = 0

    frames = []
    steps_per_frame = int(1 / (fps * model.opt.timestep))
    for _ in range(int(seconds * fps)):
        for _ in range(steps_per_frame):
            data.ctrl[0] = np.clip(controller(get_state(data)), -10, 10)
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    imageio.mimsave(path, frames, fps=fps)
    print("saved", path)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    {"test": run_test, "view": run_viewer, "video": run_video}[mode]()