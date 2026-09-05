"""A USB camera, decoded on its own thread, as RGB frames for the studio page.

    dev = find_device("OBSBOT Meet 2")      # or pass /dev/video4, or 0
    cam = CameraStream(dev, show)           # `show(frame)` per preview frame
    cam.close()

Imported only when `--camera` is passed, so opencv stays an optional dependency
of a package whose control loop does not need it.

**Nothing here runs on the control loop.** `cap.read()` blocks for a frame
period -- 33 ms at 30 fps, more than one 20 ms tick -- and the JPEG encode that
follows is another few milliseconds, so calling either inline would make every
servo write late. The thread does capture, resize, colour convert and encode
(viser encodes inside the `handle.image` setter, on whichever thread assigns
it), and hands viser's own server thread nothing but bytes to send. The control
loop never learns a camera exists.

Preview-sized, not sensor-sized: a 1280x720 JPEG is ~7 ms to encode and ~100 kB
on the wire 15 times a second, for a window a few inches wide. Resizing first
costs one INTER_AREA pass and makes the encode and the transfer ~10x cheaper.
Capture stays at full resolution because MJPG at 720p is what the camera is
willing to stream fast; the decimation is on what gets *sent*.

Adapted from `visual_servoing/camera_utils.py`, minus everything a live preview
does not use: no calibration, no undistortion maps, no intrinsics, no gray
buffer, no reusable frame buffers. What was worth carrying is the MJPG fourcc
(YUYV at 720p negotiates down to single-digit fps) and the capture-capability
check in device lookup (a UVC camera registers several `/dev/video*` nodes and
the metadata ones open fine, then never yield a frame).
"""

import subprocess
import threading
import time

import cv2
import numpy as np

# What the page gets, not what the sensor gives. See the module docstring.
PREVIEW_WIDTH = 480
PREVIEW_HZ = 15


def find_device(name: str | None = None) -> str | None:
    """The `/dev/video*` capture node of a camera, by v4l2 name or the first one.

    By name rather than by index because the index moves: unplugging any other
    UVC device renumbers the rest, so `/dev/video4` is a different camera on
    Tuesday. `None` takes whichever camera is plugged in, which is what the
    default `--camera auto` wants -- there is usually one.

    Returns `None` when nothing matches. The caller decides whether that is an
    error: an absent camera is normal under `auto` and is a typo under a name.
    """
    listing = subprocess.run(["v4l2-ctl", "--list-devices"],
                             capture_output=True, text=True).stdout
    for block in listing.strip().split("\n\n"):
        lines = block.splitlines()
        if not lines or (name and name.lower() not in lines[0].lower()):
            continue
        for node in (line.strip() for line in lines[1:]):
            formats = subprocess.run(
                ["v4l2-ctl", f"--device={node}", "--list-formats-ext"],
                capture_output=True, text=True).stdout
            if "Video Capture" in formats:
                return node
    return None


def open_device(device: str) -> str | int | None:
    """What to hand `cv2.VideoCapture`, from what a user typed at `--camera`.

    `auto` is the first camera plugged in, and `None` back means there is none
    -- no window, no failure, because running this hand with no camera attached
    is an ordinary session. A bare number is an index and a path is a path (a
    video file too, which is how this is tested with nothing plugged in).
    Anything else is a camera name, and a name that matches nothing raises:
    the user named a specific camera, so silence would look like a broken one.
    """
    if device == "auto":
        return find_device()
    if device.isdigit():
        return int(device)
    if "/" in device:
        return device
    node = find_device(device)
    if node is None:
        raise RuntimeError(f"no capture device for camera {device!r}; see "
                           f"`v4l2-ctl --list-devices`")
    return node


class CameraStream:
    """A capture thread that calls `on_frame(rgb)` at `PREVIEW_HZ`.

    `on_frame` runs on the capture thread, so what it does is charged to this
    thread and not to the caller's. Assigning a viser image handle is the
    intended body: viser serialises the frame there and its server thread sends
    it, both off the control loop.

    Failure is a printed line and a dead thread, never a raise: a camera that
    unplugs mid-session must not take the hand's control loop with it. The last
    frame simply stops updating.
    """

    def __init__(self, device: str | int, on_frame, width: int = 1280,
                 height: int = 720, fps: int = 30) -> None:
        self._device, self._on_frame = device, on_frame
        self._size, self._fps = (width, height), fps
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            print(f"camera {self._device}: could not open")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        # One frame deep, so a stall shows the present and not a queued past:
        # the driver otherwise buffers, and a preview that a person watches to
        # see what the hand is doing is worth nothing if it is a second behind.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        period, next_at = 1.0 / PREVIEW_HZ, 0.0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    # Keep reading: a USB camera drops frames and comes back,
                    # and a sleep here is what stops that from becoming a spin.
                    time.sleep(0.1)
                    continue
                now = time.monotonic()
                if now < next_at:
                    # Read every frame, send some. Skipping the *read* would
                    # leave the driver's queue to go stale instead.
                    continue
                next_at = now + period
                self._on_frame(preview(frame))
        finally:
            cap.release()

    def close(self) -> None:
        self._stop.set()


def preview(frame: np.ndarray) -> np.ndarray:
    """A BGR camera frame as a small RGB one. Aspect kept; RGB is what viser wants."""
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (PREVIEW_WIDTH, round(height * PREVIEW_WIDTH / width)),
                       interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
