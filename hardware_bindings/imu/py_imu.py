import gc
import importlib.util
import pathlib
from functools import lru_cache

import numpy as np
from tqdm import trange


@lru_cache(maxsize=None)
def _load_ext(name):
    path = pathlib.Path(__file__).parent / f"{name}.abi3.so"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run `cmake --build control/build` first")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




imu_nanobind = _load_ext("imu_nanobind")

class FakeIMU:
    def __init__(self):
        self.ang_vel = np.zeros(3)
        self.world_space_ang_vel = np.zeros(3)
        self.rotation_matrix = np.eye(3)
        self.quat_xyzw = np.array([0, 0, 0, 1], dtype=np.float32)  # xyzw format [x, y, z, w]
        self.quat_wxyz = np.array([1, 0, 0, 0], dtype=np.float32)  # wxyz format [w, x, y, z]
        self.transformed_quat_wxyz = np.array([1, 0, 0, 0], dtype=np.float32)
        self.transformed_ang_vel = np.zeros(3)
        self.transformed_gravity_vec = np.array([0, 0, -1], dtype=np.float32) # normalized
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32) # normalized
        self._counter = 1  # start non-zero so startup check passes
        self.timeStamp = 0.0

    @property
    def counter(self):
        # Auto-increment on each read so the runtime freshness check in
        # get_observation() always sees a new value (mirrors real IMU behaviour).
        self._counter += 1
        return self._counter

    def shutdown(self):
        pass


class IMU (imu_nanobind.IMU):
    """
    IMU wrapper with optional rotation offset for mounting compensation.

    Args:
        port_name: Serial port name (e.g., "/dev/ttyACM0"). Auto-detect if None.
        rotation_offset: 3x3 numpy rotation matrix to apply to all orientation data.
                        E.g., 180° about Z: np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    """
    def __init__(self, port_name=None, rotation_offset: np.ndarray = None):
        if port_name is not None:
            super().__init__(port_name)
        else:
            super().__init__()
            self.findPortNameByDescription() # use `lsusb` to get port name

        self.should_print = False

        # Set rotation offset in C++ if provided
        if rotation_offset is not None:
            self.rotation_offset = np.asarray(rotation_offset, dtype=np.float32)

        self.run()

    def shutdown(self):
        self.close()
        gc.collect()
        print("IMU closed")

if __name__ == "__main__":
    # Try the submodule-internal publisher first (works when this module is
    # imported as part of an installed/discoverable `hardware_bindings` package).
    # Fall back to `common.publisher` for direct-run cases
    # (`python hardware_bindings/imu/py_imu.py`). `common/` lives as a
    # submodule inside `hardware_bindings/`, so add that dir (not the
    # project root) to sys.path before importing.
    try:
        from hardware_bindings.common.publisher import DataPublisher
    except ImportError:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from common.publisher import DataPublisher

    import time
    import viser
    from viser import uplot
    from scipy.spatial.transform import Rotation as R

    imu = IMU()
    publisher = DataPublisher('udp://localhost:9870', encoding="msgpack", broadcast=False)

    server = viser.ViserServer(host="0.0.0.0", port=8080)
    print("Viser server running at http://localhost:8080")

    server.scene.add_frame("/world", axes_length=0.5, axes_radius=0.008)
    frames = {
        'quat':   server.scene.add_frame("/quat",   axes_length=0.30, axes_radius=0.006),
        'euler':  server.scene.add_frame("/euler",  axes_length=0.28, axes_radius=0.005),
        'matrix': server.scene.add_frame("/matrix", axes_length=0.26, axes_radius=0.004),
    }
    for name in frames:
        server.scene.add_label(f"/{name}/label", name.capitalize(), position=(0.35, 0, 0))

    gravity_spline = server.scene.add_spline_catmull_rom(
        "/matrix/gravity", positions=np.zeros((2, 3)), line_width=4.0, color=(255, 80, 255))
    gravity_direct_spline = server.scene.add_spline_catmull_rom(
        "/matrix/gravity_direct", positions=np.zeros((2, 3)), line_width=4.0, color=(200, 100, 100))

    plot_history, plot_idx = 200, [0]
    history = {k: np.zeros((plot_history, 3)) for k in ['acc', 'ang_vel', 'mag']}

    def make_xyz_plot(gui, title):
        gui.add_markdown(f"### {title}")
        init = (np.array([0.0]),) * 4
        series = (uplot.Series(label="t"), uplot.Series(label="X", stroke="red"),
                  uplot.Series(label="Y", stroke="green"), uplot.Series(label="Z", stroke="blue"))
        return gui.add_uplot(data=init, series=series)

    plots = {
        'acc':     make_xyz_plot(server.gui, "Acceleration"),
        'ang_vel': make_xyz_plot(server.gui, "Angular Velocity"),
        'mag':     make_xyz_plot(server.gui, "Magnetometer"),
    }

    def xyzw_to_wxyz(q):
        return np.array([q[3], q[0], q[1], q[2]])

    counter = 0
    try:
        for _ in trange(100000):
            while counter == imu.counter:
                time.sleep(1e-5)
            counter = imu.counter

            data ={
                "timeStamp": imu.timeStamp, "qos": imu.qos, "temperature": imu.temperature,
                "updateRate": imu.updateRate, "raw_acc": imu.raw_acc, "ang_vel": imu.ang_vel,
                "world_space_ang_vel": imu.world_space_ang_vel, "raw_mag": imu.raw_mag,
                "quat_xyzw": imu.quat_xyzw, "euler": imu.euler, "gravity_vec": imu.gravity_vec,
                "rotation_matrix": imu.rotation_matrix
            }
            publisher.publish({"sensor": data})
            # print(data)

            frames['quat'].wxyz   = xyzw_to_wxyz(imu.quat_xyzw)
            frames['euler'].wxyz  = xyzw_to_wxyz(R.from_euler('xyz', imu.euler, degrees=True).as_quat())
            frames['matrix'].wxyz = xyzw_to_wxyz(R.from_matrix(imu.rotation_matrix).as_quat())
            gravity_direct_spline.positions = np.array([[0, 0, 0], imu.direct_gravity_vec * 0.3])
            gravity_spline.positions        = np.array([[0, 0, 0], imu.gravity_vec * 0.3])

            idx = plot_idx[0] % plot_history
            history['acc'][idx] = imu.raw_acc
            history['ang_vel'][idx] = imu.ang_vel
            history['mag'][idx] = imu.raw_mag
            plot_idx[0] += 1
            if plot_idx[0] % 10 == 0:
                n = min(plot_idx[0], plot_history)
                roll = (idx + 1) if plot_idx[0] >= plot_history else 0
                t = np.arange(n, dtype=np.float64)
                for key in plots:
                    data = np.roll(history[key], -roll, axis=0)[:n]
                    plots[key].data = (t, data[:, 0], data[:, 1], data[:, 2])

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    imu.shutdown()

