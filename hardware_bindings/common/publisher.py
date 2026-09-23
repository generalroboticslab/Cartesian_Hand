import socket
import urllib.parse
import threading
import time
from typing import Any

import orjson
import msgpack
import numpy as np
import pynng


# -------
#  NNGPublisher / NNGSubscriber  (pynng Pub0/Sub0)
# -------

class NNGPublisher:
    """Publish dicts over pynng Pub0 (TCP).

    Publisher binds; subscribers connect. Supports any number of subscribers
    automatically. ``publish()`` is non-blocking: if no subscriber is connected
    or the send buffer is full, the message is dropped silently (TryAgain).

    When ``thread=True`` (default), ``publish()`` enqueues to a deque and a
    daemon thread drains it — the caller's hot loop is never blocked by I/O.

    Args:
        bind_url:  pynng URL to bind, e.g. ``"tcp://*:9870"``.
        encoding:  ``"msgpack"`` (default) or ``"json"``.
        thread:    Use background thread for non-blocking publish (default True).

    Example::

        pub = NNGPublisher("tcp://*:9870")
        pub.publish({"obs": 42})
        pub.close()
    """

    def __init__(self, bind_url: str, encoding: str = "msgpack", thread: bool = True):
        self._encode = (
            lambda d: msgpack.packb(_to_builtin(d), use_single_float=False, use_bin_type=True)
            if encoding == "msgpack"
            else orjson.dumps(_to_builtin(d))
        )
        # send_buffer_size=1: drop rather than queue when subscriber is slow.
        # Publisher binds; subscribers connect.
        self._sock = pynng.Pub0(send_buffer_size=1)
        self._sock.listen(bind_url)

        self._threaded = thread
        if thread:
            self._latest_data: dict[str, Any] | None = None
            self._has_data = False
            self._data_lock = threading.Lock()
            self._wake_event = threading.Event()
            self._stop = False
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self.publish = self._enqueue
        else:
            self.publish = self._send

    def _send(self, data: dict):
        try:
            self._sock.send(self._encode(data), block=False)
        except pynng.TryAgain:
            pass  # no subscriber or buffer full — drop silently

    def _enqueue(self, data: dict):
        with self._data_lock:
            self._latest_data = data
            self._has_data = True
        self._wake_event.set()

    def _loop(self):
        while not self._stop:
            self._wake_event.wait()
            if self._stop:
                break
            while True:
                with self._data_lock:
                    if not self._has_data:
                        self._wake_event.clear()
                        break
                    data = self._latest_data
                    self._has_data = False
                self._send(data)

    def close(self):
        """Stop background thread and close socket."""
        if self._threaded:
            self._stop = True
            self._wake_event.set()
            self._thread.join(timeout=1.0)
        self._sock.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class NNGSubscriber:
    """Subscribe to a pynng Pub0 publisher (TCP).

    Subscriber connects to the publisher's bind URL. Latest-value-wins: the
    background thread drains queued messages after each recv so ``self.data``
    always holds the newest packet. Call ``start()`` to begin receiving.

    Args:
        connect_url:      pynng URL to connect to, e.g. ``"tcp://192.168.1.10:9873"``.
        encoding:         ``"msgpack"`` (default) or ``"json"``.
        recv_timeout_ms:  Recv timeout in milliseconds (default 200).

    Example::

        sub = NNGSubscriber("tcp://192.168.1.10:9873")
        sub.start()
        # … later …
        if sub.data is not None:
            process(sub.data)
        sub.stop()
    """

    def __init__(self, connect_url: str, encoding: str = "msgpack",
                 recv_timeout_ms: int = 200):
        self._url = connect_url
        self._timeout = recv_timeout_ms
        self._decode = (
            (lambda b: msgpack.unpackb(b, raw=False))
            if encoding == "msgpack"
            else orjson.loads
        )
        # Public state — written by receiver thread, read by main thread (GIL-safe)
        self.data: dict | None = None
        self.data_id: int = -1
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Start background receive thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal receive thread to exit and wait."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self):
        """Background thread: owns the Sub0 socket for its entire lifetime.

        pynng sockets are NOT thread-safe — this thread creates, uses, and
        closes the socket entirely within itself (same pattern as
        camera_streaming_utils.py CameraStreamingClient).
        """
        sock = pynng.Sub0(recv_timeout=self._timeout, recv_buffer_size=1)
        sock.subscribe(b"")   # receive all messages

        # Retry dial until connected or stopped — publisher may not be up yet.
        while self._running:
            try:
                sock.dial(self._url, block=False)
                break
            except pynng.ConnectionRefused:
                time.sleep(0.5)
        else:
            sock.close()
            return

        while self._running:
            try:
                raw = sock.recv()
            except pynng.Closed:
                break
            except (pynng.Timeout, pynng.TryAgain):
                continue

            # Drain any additional queued messages — keep only latest
            while True:
                try:
                    raw = sock.recv(block=False)
                except (pynng.TryAgain, pynng.Timeout, pynng.Closed):
                    break

            try:
                self.data = self._decode(raw)
                self.data_id += 1
            except Exception:
                pass  # malformed packet — skip

        sock.close()

    # Alias for drop-in compatibility with DataReceiver call sites
    def receive_continuously(self):
        self.start()


# -------
#  DataPublisher
# -------

class DataPublisher:
    """Publish dicts over UDP/TCP sockets, optionally in a background thread.

    When ``thread=True`` (default), ``publish()`` only updates a latest-value
    slot and signals a daemon sender thread (latest-value-wins semantics).
    This keeps the caller's hot loop non-blocking without FIFO backlog.

    Supported URL schemes: ``udp://``, ``tcp://``, ``unix://``.

    Example::

        pub = DataPublisher('udp://localhost:9870')
        pub.publish({'sensor': 42, 'value': 3.14})
    """

    SUPPORTED_SCHEMES = {'unix', 'tcp', 'udp'}
    ENCODINGS = {
        'raw':     lambda data: data,
        'utf-8':   lambda data: data.encode('utf-8'),
        'msgpack': lambda data: msgpack.packb(data, use_single_float=False, use_bin_type=True),
        'json':    lambda data: orjson.dumps(data),
    }

    def __init__(
        self,
        target_url: str = 'udp://localhost:9870',
        encoding: str = 'msgpack',
        broadcast: bool = False,
        enable: bool = True,
        thread: bool = True,
        save_to_file: bool = False,
        save_file_path: str = None,
        **socket_kwargs,
    ):
        self.enable = enable
        self.url = urllib.parse.urlparse(target_url)
        if self.url.scheme not in self.SUPPORTED_SCHEMES:
            raise ValueError(f"Unsupported scheme: {target_url}")

        # Socket setup
        self._socket_family = self._get_socket_family()
        self.is_tcp = self.url.scheme == 'tcp'
        sock_type = socket.SOCK_STREAM if self.is_tcp else socket.SOCK_DGRAM
        self.socket = socket.socket(self._socket_family, sock_type, **socket_kwargs)
        self.hostname = '<broadcast>' if broadcast else self.url.hostname
        if broadcast and not self.is_tcp:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.connected = False  # TCP lazy-connect state

        # Encoding (skip convert_to_python_builtin_types for raw bytes)
        if encoding not in self.ENCODINGS:
            raise ValueError(f'Invalid encoding: {encoding}')
        self.encode = self.ENCODINGS[encoding]
        self._needs_conversion = encoding != 'raw'

        # Optional file recording — handle kept open for the lifetime
        self._save_file = None
        if save_to_file:
            if save_file_path is None:
                raise ValueError("save_file_path must be provided if save_to_file is True")
            self._save_file = open(save_file_path, 'ab')

        # Threading: publish() stores only the latest sample and signals sender thread.
        # This keeps caller non-blocking while avoiding FIFO backlog latency.
        self._threaded = thread
        if thread:
            self._latest_data: dict[str, Any] | None = None
            self._has_data = False
            self._data_lock = threading.Lock()
            self._wake_event = threading.Event()
            self._stop_flag = False
            self._thread = threading.Thread(target=self._publisher_loop, daemon=True)
            self._thread.start()
            self.publish = self._enqueue
        else:
            self.publish = self._send

    # -- Socket family detection ---------

    def _get_socket_family(self):
        if 'unix' in self.url.scheme:
            return socket.AF_UNIX
        return socket.AF_INET6 if ':' in (self.url.hostname or '') else socket.AF_INET

    # -- Sending --------------------------

    def _send(self, data: dict):
        """Encode *data* and send over the socket (+ optional file write)."""
        if not self.enable:
            return
        converted = convert_to_python_builtin_types(data) if self._needs_conversion else data
        encoded = self.encode(converted)
        try:
            if self.is_tcp:
                self._send_tcp(encoded)
            else:
                self.socket.sendto(encoded, (self.hostname, self.url.port))
            if self._save_file:
                self._save_file.write(encoded)
                self._save_file.flush()
        except Exception as e:
            print(f"\033[93m[Publisher] Failed to send data: {e}\033[0m")

    def _send_tcp(self, encoded: bytes):
        """Send over TCP with lazy connect and auto-reconnect."""
        try:
            if not self.connected:
                self.socket.connect((self.hostname, self.url.port))
                self.connected = True
            self.socket.sendall(encoded)
        except (BrokenPipeError, ConnectionResetError):
            print("\033[93m[Publisher] Connection lost. Reconnecting...\033[0m")
            try:
                self.socket.close()
                self.socket = socket.socket(self._socket_family, socket.SOCK_STREAM)
                self.socket.connect((self.hostname, self.url.port))
                self.connected = True
                self.socket.sendall(encoded)
                print("[Publisher] Reconnected successfully.")
            except Exception as err:
                self.connected = False
                print(f"\033[93m[Publisher] Reconnection failed: {err}\033[0m")

    # -- Threaded publishing --------------

    def _enqueue(self, data: dict):
        """Hot-path for threaded mode: overwrite stale data and signal sender."""
        with self._data_lock:
            self._latest_data = data
            self._has_data = True
        self._wake_event.set()

    def _publisher_loop(self):
        """Background thread: send latest data only (latest-value-wins)."""
        while not self._stop_flag:
            self._wake_event.wait()
            if self._stop_flag:
                break
            while True:
                with self._data_lock:
                    if not self._has_data:
                        self._wake_event.clear()
                        break
                    data = self._latest_data
                    self._has_data = False
                self._send(data)

    # -- Cleanup --------------------------

    def __del__(self):
        if self._threaded:
            self._stop()
        if self._save_file:
            self._save_file.close()

    def _stop(self):
        self._stop_flag = True
        self._wake_event.set()
        self._thread.join(timeout=1.0)
        self.socket.close()


# -------
#  DataReceiver
# -------

class DataReceiver:
    """Receive data published by ``DataPublisher`` over UDP or TCP.

    Call ``receive_continuously()`` to start a background thread that writes
    incoming data to ``self.data``.  The main thread reads ``self.data``
    directly — just an attribute access, no locks.

    Example::

        rx = DataReceiver(port=9870)
        rx.receive_continuously()
        # … later, in any thread …
        if rx.data is not None:
            process(rx.data)
        rx.stop()
    """

    DECODINGS = {
        'raw':     lambda data: data,
        'utf-8':   lambda data: data.decode('utf-8'),
        'msgpack': lambda data: msgpack.unpackb(data, raw=False),
        'json':    lambda data: orjson.loads(data),
    }

    def __init__(self, port=9870, decoding='msgpack', broadcast=False,
                 enable=True, protocol='udp'):
        self.protocol = protocol
        self.enable = enable
        self._thread = None
        self._running = True

        # Public state — written by receiver thread, read by main thread
        self.data = None
        self.data_id = -1
        self.address = None

        # Socket setup
        if protocol == 'tcp':
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('', port))
            self.socket.listen(1)
            self.client_socket = None
            self.client_address = None
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            if broadcast:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.bind(('<broadcast>' if broadcast else '', port))
            self.socket.settimeout(0.1)

        # Decoding
        decoding = decoding.lower()
        if decoding not in self.DECODINGS:
            raise ValueError(f"Invalid decoding: {decoding}")
        self._decode = self.DECODINGS[decoding]
        # TCP streams can coalesce messages; use a streaming unpacker
        self._unpacker = msgpack.Unpacker(raw=False) if (protocol == 'tcp' and decoding == 'msgpack') else None

    # -- Decoding -------------------------

    def decode(self, data: bytes):
        if self._unpacker:
            # TCP may coalesce multiple messages; keep the latest
            self._unpacker.feed(data)
            last = None
            for msg in self._unpacker:
                last = msg
            return last
        return self._decode(data)

    # -- Receiving ------------------------

    def _accept_tcp_connection(self):
        """Block until a TCP client connects (called once)."""
        if self.client_socket is None:
            self.client_socket, self.client_address = self.socket.accept()
            self.client_socket.setblocking(False)
            self.address = self.client_address
            print(f"\033[92m[Receiver] TCP connection from {self.client_address}\033[0m")

    def receive(self, timeout=0.1, buffer_size=65536):
        """Receive one message.  Returns ``(data, address)`` or ``(None, None)``."""
        try:
            if self.protocol == 'tcp':
                return self._receive_tcp(timeout, buffer_size)
            else:
                return self._receive_udp(buffer_size)
        except socket.timeout:
            return None, None

    def _receive_tcp(self, timeout, buffer_size):
        self._accept_tcp_connection()
        self.client_socket.settimeout(timeout)
        raw = self.client_socket.recv(buffer_size)
        if not raw:
            self.client_socket.close()
            self.client_socket = None
            return None, None
        self.data = self.decode(raw)
        self.data_id += 1
        return self.data, self.client_address

    def _receive_udp(self, buffer_size):
        raw, self.address = self.socket.recvfrom(buffer_size)
        self.data = self.decode(raw)
        self.data_id += 1
        return self.data, self.address

    def receive_continuously(self, timeout=0.1, buffer_size=65536):
        """Start a daemon thread that receives in a loop.

        Socket I/O blocks in C (releases the GIL), so this thread doesn't
        compete with a time-critical main loop while waiting for packets.
        """
        if self.protocol != 'tcp':
            self.socket.settimeout(timeout)

        def _loop():
            _recv = self.receive
            while self._running:
                if self.enable:
                    _recv(timeout=timeout, buffer_size=buffer_size)
                else:
                    time.sleep(timeout)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.protocol == 'tcp' and self.client_socket:
            self.client_socket.close()
        self.socket.close()


# -------
#  Utilities
# -------

def _to_builtin(nested_data):
    """Recursively convert numpy arrays / tensors to plain Python types (used by NNG classes)."""
    if isinstance(nested_data, dict):
        return {k: _to_builtin(v) for k, v in nested_data.items()}
    if hasattr(nested_data, 'tolist'):
        return nested_data.tolist()
    return nested_data


def convert_to_python_builtin_types(nested_data: dict) -> dict:
    """Recursively convert numpy arrays / tensors to plain Python types."""
    out = {}
    for key, value in nested_data.items():
        if isinstance(value, dict):
            out[key] = convert_to_python_builtin_types(value)
        elif hasattr(value, 'tolist'):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def unpack_data_from_file(file_path: str, decoding: str = 'msgpack') -> list:
    """Read all packed messages from a file."""
    with open(file_path, 'rb') as f:
        if decoding == 'msgpack':
            return list(msgpack.Unpacker(f, raw=False))
        raise ValueError(f"Unsupported decoding: {decoding}")


# -------
#  Self-test
# -------

if __name__ == "__main__":

    def test_data(i):
        return {
            "id": i,
            "sensor_id": np.random.randint(0, 10),
            "temperature": 25.5,
            "time": time.time(),
        }

    def test_tcp():
        print("\033[92mStarting publisher and receiver with TCP.\033[0m")

        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+b')
        path = tmp.name

        publisher = DataPublisher(
            target_url="tcp://localhost:9872", encoding="msgpack",
            broadcast=False, thread=True,
            save_to_file=True, save_file_path=path,
        )
        receiver = DataReceiver(port=9872, decoding="msgpack", protocol="tcp")
        receiver.receive_continuously()

        for i in range(10):
            publisher.publish(test_data(i))
            time.sleep(1e-3)
            print(f"[Receiver] {receiver.data_id}: {receiver.data}")
        receiver.stop()
        print("Stopped.")

        results = unpack_data_from_file(path)
        print(f"Data saved to {path}. Loaded {len(results)} messages.")
        print(f"All messages: {results}")

    def test_udp():
        print("\033[92mStarting publisher and receiver with UDP.\033[0m")

        publisher = DataPublisher(
            target_url="udp://localhost:9870", encoding="msgpack",
            broadcast=True, thread=True,
        )
        receivers = []
        for _ in range(2):
            rx = DataReceiver(port=9870, decoding="msgpack", broadcast=True)
            rx.receive_continuously()
            receivers.append(rx)

        for i in range(10):
            publisher.publish(test_data(i))
            time.sleep(1e-3)
            for k, rx in enumerate(receivers):
                print(f"receiver [{k}] {rx.data_id}, {rx.address}: {rx.data}")

        for rx in receivers:
            rx.stop()
        print("Publisher and Receiver have stopped.")

    test_tcp()
    # test_udp()
