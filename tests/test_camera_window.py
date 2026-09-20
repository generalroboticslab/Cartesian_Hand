"""The camera window, end to end, with no camera plugged in.

`cv2.VideoCapture` opens a video file the same way it opens `/dev/video*`, so a
generated clip exercises the whole path this adds: capture thread -> resize and
colour convert -> viser image handle. What it cannot cover is the device lookup
in `camera.find_device`, which needs hardware.

Run: python tests/test_camera_window.py
"""

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import viser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cartesian_hand import camera  # noqa: E402


def clip(path: str, frames: int = 60, size: tuple[int, int] = (640, 360)) -> None:
    """A short clip whose frames differ, so a frozen preview is detectable."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), 0, np.uint8)
        frame[:, :, 0] = i * 4          # blue ramp: BGR in, so R out
        writer.write(frame)
    writer.release()


def test_preview_is_small_rgb():
    frame = np.zeros((360, 640, 3), np.uint8)
    frame[:, :, 0] = 200                # blue in BGR
    out = camera.preview(frame)
    assert out.shape == (camera.PREVIEW_WIDTH * 360 // 640, camera.PREVIEW_WIDTH, 3)
    assert out[0, 0, 2] > 150 and out[0, 0, 0] < 50, "BGR was not converted to RGB"


def test_stream_feeds_a_viser_image_off_the_calling_thread():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/clip.avi"
        clip(path)

        server = viser.ViserServer(host="127.0.0.1", port=8099, verbose=False)
        view = server.gui.add_image(np.zeros((90, 160, 3), np.uint8), format="jpeg")
        blank = view._data

        seen = []
        # A file has no frame rate on the read side -- the whole clip decodes in
        # under a millisecond -- so the 15 Hz send limit would pass exactly one
        # frame before EOF. Raised for the test; a real camera paces itself.
        camera.PREVIEW_HZ = 1000
        stream = camera.CameraStream(path, lambda f: (seen.append(f),
                                                      setattr(view, "image", f)))
        # A fixed number of ticks, not "until frames arrive": the stall this
        # guards against is on this thread, so it has to keep ticking while the
        # camera thread works.
        caller = []
        for _ in range(100):
            caller.append(time.time())   # the "control loop": never blocked
            time.sleep(0.01)
        stream.close()
        server.stop()

        assert len(seen) >= 3, f"only {len(seen)} frames in 5 s"
        assert seen[0].shape[1] == camera.PREVIEW_WIDTH
        assert view._data not in (None, blank), "viser handle never got a frame"
        # 10 ms sleeps, so a read()/encode landing on this thread would show up
        # as a gap of a frame period or more.
        gaps = np.diff(caller)
        assert gaps.max() < 0.05, f"caller stalled for {gaps.max():.3f}s"


if __name__ == "__main__":
    test_preview_is_small_rgb()
    test_stream_feeds_a_viser_image_off_the_calling_thread()
    print("ok")
