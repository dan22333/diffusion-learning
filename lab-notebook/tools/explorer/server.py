#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from explorer.artifacts import PathPolicy, TableCache, content_type, preview
    from explorer.feedback import FeedbackStore, default_state_dir
    from explorer.model import ResearchModel, SearchIndex
else:
    from .artifacts import PathPolicy, TableCache, content_type, preview
    from .feedback import FeedbackStore, default_state_dir
    from .model import ResearchModel, SearchIndex


STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_IDLE_TIMEOUT = 5 * 60


class ServerLease:
    """An advisory, process-owned lease plus private connection metadata."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir.resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.state_dir / "explorer-server.lock"
        self.metadata_path = self.state_dir / "explorer-server.json"
        self.handle: int | None = None

    def acquire(self) -> bool:
        handle = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(handle).st_size == 0:
                    os.write(handle, b"\0")
                    os.lseek(handle, 0, os.SEEK_SET)
                msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            os.close(handle)
            return False
        self.handle = handle
        return True

    def write(self, metadata: dict[str, Any]) -> None:
        temporary = self.metadata_path.with_name(f"{self.metadata_path.name}.{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            encoded = (json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)
        os.replace(temporary, self.metadata_path)

    def read(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        required = {"pid", "root", "host", "port"}
        return payload if isinstance(payload, dict) and required <= payload.keys() else None

    def clear(self, *, pid: int | None = None) -> None:
        current = self.read()
        if current is None:
            self.metadata_path.unlink(missing_ok=True)
            return
        if pid is not None and current.get("pid") != pid:
            return
        self.metadata_path.unlink(missing_ok=True)

    def release(self) -> None:
        if self.handle is None:
            return
        if os.name == "nt":
            import msvcrt

            os.lseek(self.handle, 0, os.SEEK_SET)
            msvcrt.locking(self.handle, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle, fcntl.LOCK_UN)
        os.close(self.handle)
        self.handle = None


def explorer_url(metadata: dict[str, Any]) -> str:
    host = f"[{metadata['host']}]" if ":" in str(metadata["host"]) else metadata["host"]
    return f"http://{host}:{metadata['port']}/"


def api_url(metadata: dict[str, Any], path: str) -> str:
    host = f"[{metadata['host']}]" if ":" in str(metadata["host"]) else metadata["host"]
    return f"http://{host}:{metadata['port']}{path}"


def probe_runtime(metadata: dict[str, Any], timeout: float = 0.75) -> dict[str, Any] | None:
    try:
        request = urllib.request.Request(api_url(metadata, "/api/runtime"))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        if isinstance(payload, dict) and payload.get("root") == metadata.get("root"):
            return payload
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
        pass
    return None


def wait_for_runtime(lease: ServerLease, timeout: float = 3.0) -> tuple[dict[str, Any], dict[str, Any]] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        metadata = lease.read()
        if metadata is not None:
            runtime = probe_runtime(metadata)
            if runtime is not None:
                return metadata, runtime
        time.sleep(0.05)
    return None


class ExplorerState:
    def __init__(
        self,
        root: Path,
        notebook: str,
        experiments: str,
        state_dir: Path | None,
        poll_interval: float,
        editor_command: str | None,
        allow_system_open: bool,
    ) -> None:
        self.root = root.resolve()
        self.model = ResearchModel(self.root, notebook, experiments)
        self.state_dir = (state_dir or default_state_dir(self.root)).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.search = SearchIndex(self.state_dir / "explorer.sqlite3")
        self.policy = PathPolicy(self.root, self.model.notebook, experiments)
        self.tables = TableCache(self.state_dir / "explorer.sqlite3", self.policy)
        self.feedback = FeedbackStore(self.root, self.state_dir)
        self.poll_interval = max(0.25, poll_interval)
        self.editor_command = editor_command
        self.allow_system_open = allow_system_open
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.version = 0
        self.snapshot: dict[str, Any] = {}
        self.public_snapshot: dict[str, Any] = {}
        self.file_state: dict[str, tuple[int, int]] = {}
        self.stop_event = threading.Event()
        self.activity_lock = threading.Lock()
        self.last_activity = time.monotonic()
        self.browser_clients = 0
        self.closed = False
        self.refresh()

    def refresh(self) -> None:
        snapshot = self.model.build()
        self.search.rebuild(snapshot)
        with self.condition:
            self.version += 1
            snapshot["version"] = self.version
            feedback = self.feedback.list_resolved(snapshot)
            snapshot["feedback"] = feedback
            open_feedback = sum(item.get("status") == "open" for item in feedback)
            if open_feedback:
                attention = snapshot.get("dashboard", {}).get("attention", [])
                insert_at = next(
                    (index for index, item in enumerate(attention) if item.get("kind") in {"todo", "journal-gap"}),
                    len(attention),
                )
                attention.insert(
                    insert_at,
                    {
                        "kind": "feedback",
                        "tone": "warning",
                        "count": open_feedback,
                        "label": "Open explorer feedback",
                        "detail": "Addressable requests waiting in the feedback outbox.",
                        "route": "feedback",
                        "target": "",
                    },
                )
            snapshot["capabilities"] = {
                "system_open": self.allow_system_open,
                "editor": bool(self.editor_command),
                "state_dir": self.state_dir.as_posix(),
            }
            self.snapshot = snapshot
            # The browser never reads document text: blocks carry markdown and
            # plain forms, and feedback reattachment runs server-side.
            self.public_snapshot = {
                **snapshot,
                "documents": [
                    {key: value for key, value in document.items() if key != "text"}
                    for document in snapshot["documents"]
                ],
            }
            self.file_state = self.model.watched_state()
            self.condition.notify_all()

    # Refresh only after writes go quiet, but never starve longer than
    # MAX_PENDING_SECONDS while a live experiment streams files.
    QUIET_SECONDS = 0.4
    MAX_PENDING_SECONDS = 10.0

    def watcher(self) -> None:
        pending_since: float | None = None
        last_change: float | None = None
        delay = self.poll_interval
        while not self.stop_event.wait(delay):
            started = time.monotonic()
            try:
                current = self.model.watched_state()
                now = time.monotonic()
                if current != self.file_state:
                    pending_since = pending_since or now
                    last_change = now
                    self.file_state = current
                if pending_since is not None and last_change is not None:
                    quiet = now - last_change
                    waited = now - pending_since
                    if quiet >= self.QUIET_SECONDS or waited >= self.MAX_PENDING_SECONDS:
                        self.refresh()
                        pending_since = last_change = None
            except Exception as error:
                print(f"research-journal explorer watcher: {error}", file=sys.stderr)
            # Keep the watcher at or below a 50% duty cycle when scanning a
            # large artifact tree costs more than the configured interval.
            delay = max(self.poll_interval, time.monotonic() - started)

    def snapshot_copy(self) -> dict[str, Any]:
        with self.lock:
            return self.snapshot

    def public_snapshot_copy(self) -> dict[str, Any]:
        with self.lock:
            return self.public_snapshot

    def touch(self) -> None:
        with self.activity_lock:
            self.last_activity = time.monotonic()

    def browser_connected(self) -> None:
        with self.activity_lock:
            self.browser_clients += 1
            self.last_activity = time.monotonic()

    def browser_disconnected(self) -> None:
        with self.activity_lock:
            self.browser_clients = max(0, self.browser_clients - 1)
            self.last_activity = time.monotonic()

    def idle_for(self) -> tuple[int, float]:
        with self.activity_lock:
            return self.browser_clients, time.monotonic() - self.last_activity

    def request_stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()

    def open_target(self, raw_path: str, line: int, action: str) -> None:
        path = self.policy.resolve(raw_path)
        if action == "reveal":
            if not self.allow_system_open:
                raise PermissionError("system-open actions are disabled")
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path.as_posix()])
            elif os.name == "nt":
                subprocess.Popen(["explorer", "/select,", path.as_posix()])
            else:
                subprocess.Popen(["xdg-open", path.parent.as_posix()])
            return
        if action == "editor":
            if not self.editor_command:
                raise PermissionError("no editor command is configured")
            values = {"path": path.as_posix(), "line": str(max(1, line)), "root": self.root.as_posix()}
            command = [part.format(**values) for part in shlex.split(self.editor_command)]
            subprocess.Popen(command, cwd=self.root)
            return
        raise ValueError("unknown open action")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.request_stop()
        self.search.close()
        self.tables.close()


class ExplorerHandler(BaseHTTPRequestHandler):
    server_version = "ResearchJournalExplorer/1"
    protocol_version = "HTTP/1.1"

    @property
    def explorer(self) -> ExplorerState:
        return self.server.explorer  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):  # type: ignore[attr-defined]
            return
        super().log_message(format, *args)

    def _parsed(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urllib.parse.urlparse(self.path)
        return parsed.path, urllib.parse.parse_qs(parsed.query)

    def _trusted_host(self) -> bool:
        try:
            parsed = urllib.parse.urlsplit(f"//{self.headers.get('Host', '')}")
            return (
                parsed.hostname in {"127.0.0.1", "::1", "localhost"}
                and parsed.port == self.server.server_address[1]  # type: ignore[attr-defined]
            )
        except ValueError:
            return False

    def _trusted_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        try:
            parsed = urllib.parse.urlsplit(origin)
            return (
                parsed.scheme == "http"
                and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
                and parsed.port == self.server.server_address[1]  # type: ignore[attr-defined]
            )
        except ValueError:
            return False

    def _require_trusted_request(self, *, write: bool = False) -> bool:
        if not self._trusted_host() or (write and not self._trusted_origin()):
            self._error(403, "request must use the explorer's loopback origin")
            return False
        return True

    def _accepts_gzip(self) -> bool:
        accept = self.headers.get("Accept-Encoding", "")
        return any(token.split(";")[0].strip() == "gzip" for token in accept.split(","))

    def _compressible(self, data: bytes) -> bytes | None:
        if len(data) < 1024 or not self._accepts_gzip():
            return None
        return gzip.compress(data, compresslevel=5)

    def _json(self, payload: Any, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        compressed = self._compressible(encoded)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if compressed is not None:
            self.send_header("Content-Encoding", "gzip")
            encoded = compressed
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 128 * 1024:
            raise ValueError("request body is too large")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        return payload

    def _static(self, relative: str, *, head_only: bool = False) -> None:
        relative = relative or "index.html"
        candidate = (STATIC / relative).resolve()
        if STATIC.resolve() not in candidate.parents or not candidate.is_file():
            self._error(404, "not found")
            return
        data = candidate.read_bytes()
        compressed = self._compressible(data) if candidate.suffix in {".html", ".js", ".css", ".svg"} else None
        self.send_response(200)
        self.send_header("Content-Type", content_type(candidate))
        if compressed is not None:
            self.send_header("Content-Encoding", "gzip")
            data = compressed
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-src 'self'; object-src 'none'; base-uri 'none'")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def do_HEAD(self) -> None:
        if not self._require_trusted_request():
            return
        path, _query = self._parsed()
        if path == "/" or path.startswith("/static/"):
            relative = "index.html" if path == "/" else path.removeprefix("/static/")
            self._static(relative, head_only=True)
            return
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._require_trusted_request():
            return
        path, query = self._parsed()
        if path == "/" or path.startswith("/static/"):
            relative = "index.html" if path == "/" else path.removeprefix("/static/")
            self._static(relative)
            return
        if not path.startswith("/api/"):
            self._static("index.html")
            return
        try:
            if path == "/api/runtime":
                self._json(self.server.runtime_status())  # type: ignore[attr-defined]
            elif path == "/api/snapshot":
                self.explorer.touch()
                self._json(self.explorer.public_snapshot_copy())
            elif path == "/api/search":
                self.explorer.touch()
                self._json({"results": self.explorer.search.search((query.get("q") or [""])[0])})
            elif path == "/api/feedback":
                self.explorer.touch()
                self._json({"feedback": self.explorer.feedback.list_resolved(self.explorer.snapshot_copy())})
            elif path == "/api/artifact/preview":
                self.explorer.touch()
                self._json(
                    preview(
                        self.explorer.policy,
                        (query.get("path") or [""])[0],
                        int((query.get("offset") or ["0"])[0]),
                        int((query.get("limit") or [str(4 * 1024 * 1024)])[0]),
                    )
                )
            elif path == "/api/artifact/table":
                self.explorer.touch()
                raw_sort = (query.get("sort") or [""])[0]
                self._json(
                    self.explorer.tables.page(
                        (query.get("path") or [""])[0],
                        int((query.get("page") or ["0"])[0]),
                        int((query.get("page_size") or ["100"])[0]),
                        int(raw_sort) if raw_sort.isdigit() else None,
                        (query.get("direction") or ["asc"])[0],
                        (query.get("filter") or [""])[0],
                    )
                )
            elif path == "/api/artifact/content":
                self.explorer.touch()
                self._content((query.get("path") or [""])[0])
            elif path == "/api/events":
                self.explorer.touch()
                self._events(int((query.get("version") or ["0"])[0]))
            else:
                self._error(404, "unknown API route")
        except PermissionError as error:
            self._error(403, str(error))
        except FileNotFoundError as error:
            self._error(404, str(error))
        except (ValueError, json.JSONDecodeError) as error:
            self._error(400, str(error))
        except Exception as error:
            self._error(500, str(error))

    def _content(self, raw_path: str) -> None:
        path = self.explorer.policy.resolve(raw_path)
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            match = range_header[6:].split("-", 1)
            start = int(match[0] or 0)
            end = min(size - 1, int(match[1]) if match[1] else size - 1)
        length = max(0, end - start + 1)
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", content_type(path))
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Security-Policy", "sandbox")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _events(self, known_version: int) -> None:
        self.explorer.browser_connected()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            version = known_version
            while not self.explorer.stop_event.is_set():
                with self.explorer.condition:
                    if self.explorer.version <= version:
                        self.explorer.condition.wait(timeout=15)
                    current = self.explorer.version
                if current > version:
                    self.wfile.write(f"event: change\ndata: {json.dumps({'version': current})}\n\n".encode())
                    version = current
                else:
                    self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.explorer.browser_disconnected()

    def do_POST(self) -> None:
        if not self._require_trusted_request(write=True):
            return
        path, _query = self._parsed()
        self.explorer.touch()
        try:
            payload = self._body()
            if path == "/api/shutdown":
                self._json({"ok": True, "message": "explorer is stopping"})
                self.server.request_shutdown("requested by --stop")  # type: ignore[attr-defined]
            elif path == "/api/feedback":
                target = payload.get("target") or {}
                self.explorer.policy.resolve(str(target.get("path") or ""))
                record = self.explorer.feedback.create(payload)
                self.explorer.refresh()
                self._json(record, 201)
            elif path.startswith("/api/feedback/"):
                feedback_id = path.rsplit("/", 1)[-1]
                record = self.explorer.feedback.update(feedback_id, str(payload.get("status") or "open"), str(payload.get("note") or ""))
                self.explorer.refresh()
                self._json(record)
            elif path == "/api/open":
                self.explorer.open_target(
                    str(payload.get("path") or ""), int(payload.get("line") or 1), str(payload.get("action") or "reveal")
                )
                self._json({"ok": True})
            else:
                self._error(404, "unknown API route")
        except PermissionError as error:
            self._error(403, str(error))
        except KeyError as error:
            self._error(404, str(error))
        except (ValueError, json.JSONDecodeError) as error:
            self._error(400, str(error))
        except Exception as error:
            self._error(500, str(error))


class ExplorerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        explorer: ExplorerState,
        quiet: bool = False,
        idle_timeout: float = 0,
    ) -> None:
        super().__init__(address, ExplorerHandler)
        self.explorer = explorer
        self.quiet = quiet
        self.idle_timeout = max(0, idle_timeout)
        self.shutdown_lock = threading.Lock()
        self.shutdown_started = False
        self.shutdown_reason = ""

    def runtime_status(self) -> dict[str, Any]:
        clients, idle_seconds = self.explorer.idle_for()
        return {
            "pid": os.getpid(),
            "root": self.explorer.root.as_posix(),
            "active_browsers": clients,
            "idle_seconds": round(idle_seconds, 1),
            "idle_timeout": self.idle_timeout,
        }

    def request_shutdown(self, reason: str) -> None:
        with self.shutdown_lock:
            if self.shutdown_started:
                return
            self.shutdown_started = True
            self.shutdown_reason = reason
        self.explorer.request_stop()
        if not self.quiet:
            print(f"research-journal explorer stopping: {reason}", file=sys.stderr, flush=True)
        threading.Thread(target=self.shutdown, daemon=True, name="research-journal-shutdown").start()

    def idle_monitor(self) -> None:
        if self.idle_timeout <= 0:
            return
        interval = max(0.1, min(2.0, self.idle_timeout / 4))
        while not self.explorer.stop_event.wait(interval):
            clients, idle_seconds = self.explorer.idle_for()
            if clients == 0 and idle_seconds >= self.idle_timeout:
                self.request_shutdown(f"no browser connected for {self.idle_timeout:g} seconds")
                return


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Live, read-only research-journal explorer")
    result.add_argument("--project", default=".", help="project root")
    result.add_argument("--notebook", default="lab-notebook", help="notebook path relative to project")
    result.add_argument("--experiments", default="experiments/*", help="experiment-directory glob")
    result.add_argument("--host", default="127.0.0.1", help="listen host; non-loopback values are rejected")
    result.add_argument("--port", type=int, default=0, help="listen port; 0 chooses an available port")
    result.add_argument("--poll-interval", type=float, default=0.75, help="filesystem poll interval")
    result.add_argument("--state-dir", type=Path, help="derived cache and feedback location")
    result.add_argument("--open", action="store_true", help="open the explorer in the default browser")
    result.add_argument("--allow-system-open", action="store_true", help="enable reveal-in-file-manager actions")
    result.add_argument("--editor-command", help="editor command template with {path}, {line}, and {root}")
    result.add_argument("--quiet", action="store_true")
    result.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT,
        help=f"stop after this many seconds without a browser (default: {DEFAULT_IDLE_TIMEOUT}; 0 disables)",
    )
    controls = result.add_mutually_exclusive_group()
    controls.add_argument("--status", action="store_true", help="report the current project explorer")
    controls.add_argument("--stop", action="store_true", help="stop the current project explorer")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        print("research-journal explorer only binds to a loopback host", file=sys.stderr)
        return 2
    root = Path(args.project).resolve()
    notebook = root / args.notebook
    if not notebook.is_dir():
        print(f"notebook not found: {notebook}", file=sys.stderr)
        return 2
    state_dir = (args.state_dir or default_state_dir(root)).resolve()
    lease = ServerLease(state_dir)
    if args.status or args.stop:
        metadata = lease.read()
        runtime = probe_runtime(metadata) if metadata is not None else None
        if runtime is None:
            if lease.acquire():
                lease.clear()
                lease.release()
                print("research-journal explorer is not running")
                return 0 if args.stop else 1
            print("research-journal explorer is starting or not responding", file=sys.stderr)
            return 1
        if metadata.get("root") != root.as_posix():
            print(f"state directory is owned by another project: {metadata['root']}", file=sys.stderr)
            return 2
        if args.status:
            print(
                f"running pid={metadata['pid']} active_browsers={runtime['active_browsers']} "
                f"idle={runtime['idle_seconds']}s timeout={runtime['idle_timeout']}s\n{explorer_url(metadata)}"
            )
            return 0
        request = urllib.request.Request(
            api_url(metadata, "/api/shutdown"),
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=2):
                pass
        except urllib.error.URLError as error:
            print(f"could not stop research-journal explorer: {error}", file=sys.stderr)
            return 1
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and probe_runtime(metadata, timeout=0.2) is not None:
            time.sleep(0.05)
        if probe_runtime(metadata, timeout=0.2) is not None:
            print("research-journal explorer did not stop within 3 seconds", file=sys.stderr)
            return 1
        print("research-journal explorer stopped")
        return 0
    if not lease.acquire():
        existing = wait_for_runtime(lease)
        if existing is None:
            print("research-journal explorer is already starting, but did not become ready", file=sys.stderr)
            return 1
        metadata, runtime = existing
        if metadata.get("root") != root.as_posix():
            print(f"state directory is owned by another project: {metadata['root']}", file=sys.stderr)
            return 2
        missing_capability = (args.allow_system_open and not metadata.get("allow_system_open")) or (
            args.editor_command and args.editor_command != metadata.get("editor_command")
        )
        if missing_capability:
            print("existing explorer lacks requested open/editor capabilities; use --stop before relaunching", file=sys.stderr)
            return 2
        print(f"Reusing the project explorer (pid {metadata['pid']}).")
        print(explorer_url(metadata), flush=True)
        if args.open:
            webbrowser.open(explorer_url(metadata))
        return 0
    lease.clear()
    explorer = ExplorerState(
        root,
        args.notebook,
        args.experiments,
        state_dir,
        args.poll_interval,
        args.editor_command,
        args.allow_system_open,
    )
    server = ExplorerServer((args.host, args.port), explorer, args.quiet, args.idle_timeout)
    host, port = server.server_address[:2]
    metadata = {
        "pid": os.getpid(),
        "root": root.as_posix(),
        "host": host,
        "port": port,
        "idle_timeout": args.idle_timeout,
        "allow_system_open": args.allow_system_open,
        "editor_command": args.editor_command,
    }
    lease.write(metadata)
    url = explorer_url(metadata)
    print(url, flush=True)
    watcher = threading.Thread(target=explorer.watcher, daemon=True, name="research-journal-watcher")
    watcher.start()
    idle_monitor = threading.Thread(target=server.idle_monitor, daemon=True, name="research-journal-idle-monitor")
    idle_monitor.start()
    if args.open:
        webbrowser.open(url)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: server.request_shutdown("received SIGTERM"))
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        explorer.request_stop()
        watcher.join(timeout=2)
        idle_monitor.join(timeout=2)
        explorer.close()
        server.server_close()
        lease.clear(pid=os.getpid())
        lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
