"""Public web adapter for the unchanged FMCW generator; temporary downloads."""
from copy import deepcopy
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dataset'))
from fmcw_dataset.config import defaults, domain, radar_values, validate, bounds
from fmcw_dataset.core import json_safe, preview
from fmcw_dataset.storage import Cancelled, write_dataset
from fmcw_dataset.server import STATIC, presets

MAX_SCENES = 200
MAX_POINTS = 8192
MAX_BYTES = 32 * 1024**2
SESSION_SECONDS = 15 * 60
MAX_SESSIONS = 32
JOB_SECONDS = 60


def demo_config(raw, *, export=False):
    config = deepcopy(raw)
    if not isinstance(config, dict):
        raise ValueError('A configuration is required.')
    config['folder'] = ''  # A client-supplied path never selects server storage.
    if not isinstance(config.get('name'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', config['name']):
        raise ValueError('Use a file name with letters, digits, dots, hyphens or underscores (80 characters maximum).')
    config = validate(config)
    limit = domain(config)['max_points']
    objects = config['targets']['count'][1] + config['neighbors']['count'][1]
    if limit > MAX_POINTS or objects > 32:
        raise ValueError(f'Online demo: at most {MAX_POINTS:,} samples and 32 objects per scene. Use the local generator for larger jobs.')
    neighbors = config['neighbors']
    ramps = (neighbors['slope_factor'][1] if neighbors['mode'] == 'linked'
             else config['radar']['duration_us'][1] / neighbors['duration_us'][0])
    if ramps > 32:
        raise ValueError('Online demo: at most 32 interfering ramps per observation.')
    if export:
        if config['count'] > MAX_SCENES:
            raise ValueError(f'Online demo: at most {MAX_SCENES} scenes per file.')
        if config['count'] * limit * 24 > MAX_BYTES:
            raise ValueError('Online demo: at most 32 MiB of waveform data per file.')
        if config['count'] * limit * max(1, objects) > 25_000_000:
            raise ValueError('This online job is too large. Reduce the scenes, objects or samples.')
    return config


@dataclass
class Session:
    directory: Path
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)
    cancel: threading.Event = field(default_factory=threading.Event)
    job: dict = field(default_factory=lambda: dict(status='idle', done=0, total=0))
    worker: threading.Thread | None = None
    download_key: str | None = None
    file: Path | None = None
    readers: int = 0


class Demo:
    def __init__(self, origin, directory=None):
        parsed = urlparse(origin)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.path not in ('', '/'):
            raise ValueError('PUBLIC_ORIGIN must be the full site origin, without a path.')
        self.origin = f'{parsed.scheme}://{parsed.netloc}'
        self.host = parsed.netloc
        self.directory = Path(tempfile.mkdtemp(prefix='fmcw-demo-', dir=directory))
        self.lock = threading.RLock()
        self.sessions = {}
        self.generation_slot = threading.BoundedSemaphore(1)
        self.preview_slot = threading.BoundedSemaphore(1)
        self.closed = threading.Event()
        self.janitor = threading.Thread(target=self._clean_loop, daemon=True)
        self.janitor.start()

    def _clean_loop(self):
        while not self.closed.wait(60):
            self.cleanup()

    def cleanup(self):
        now = time.monotonic()
        with self.lock:
            for key, session in list(self.sessions.items()):
                with session.lock:
                    if now-session.last_used > SESSION_SECONDS and session.job['status'] != 'running' and not session.readers:
                        shutil.rmtree(session.directory, ignore_errors=True)
                        del self.sessions[key]

    def new_session(self):
        self.cleanup()
        with self.lock:
            if len(self.sessions) >= MAX_SESSIONS:
                raise ValueError('The demo is busy. Please try again later.')
            key = secrets.token_urlsafe(32)
            session = Session(self.directory / secrets.token_hex(16))
            session.directory.mkdir()
            self.sessions[key] = session
            return key

    def session(self, key):
        with self.lock:
            session = self.sessions.get(key)
            if session is None:
                raise ValueError('Your demo session has expired. Reload this page.')
            session.last_used = time.monotonic()
            return session

    def status(self, session):
        with session.lock:
            return deepcopy(session.job)

    def start(self, session, raw):
        config = demo_config(raw, export=True)
        with session.lock:
            if session.job['status'] == 'running':
                raise ValueError('Generation is already running in this session.')
            if session.readers:
                raise ValueError('Wait for your current download to finish.')
            if not self.generation_slot.acquire(blocking=False):
                raise ValueError('Another file is being generated. Please try again shortly.')
            try:
                for p in session.directory.iterdir():
                    if p.is_file():
                        p.unlink()
                session.file = None
                session.download_key = None
                session.cancel.clear()
                session.job = dict(status='running', done=0, total=config['count'], elapsed=0, speed=0, eta=None)
                def update(state):
                    with session.lock:
                        session.job.update(state)
                def work():
                    timer = threading.Timer(JOB_SECONDS, session.cancel.set)
                    timer.daemon = True
                    timer.start()
                    try:
                        result = write_dataset(config, folder=session.directory, progress=update, cancel=session.cancel)
                        path = Path(result.pop('path'))
                        # HDF5 includes metadata/chunk overhead beyond waveform estimates.
                        if result['bytes'] > 48 * 1024**2:
                            path.unlink()
                            raise ValueError('The generated file exceeds the online download limit. Reduce the scenes or samples.')
                        with session.lock:
                            session.file = path
                            session.download_key = secrets.token_urlsafe(32)
                            result.update(path=path.name, filename=path.name,
                                          download_url='/download/' + session.download_key)
                            session.job.update(status='complete', result=result)
                    except Cancelled:
                        update(dict(status='cancelled', message='Generation cancelled or time limit reached. Reduce the job and try again.'))
                    except Exception:
                        update(dict(status='error', message='Generation failed. Check the settings and try a smaller file.'))
                    finally:
                        timer.cancel()
                        self.generation_slot.release()
                session.worker = threading.Thread(target=work, daemon=True, name='demo-generation')
                session.worker.start()
            except Exception:
                session.job = dict(status='error', done=0, total=config['count'],
                                   message='Generation could not start. Please try again.')
                self.generation_slot.release()
                raise
        return self.status(session)

    def close(self):
        self.closed.set()
        with self.lock:
            sessions = list(self.sessions.values())
        for session in sessions:
            session.cancel.set()
        for session in sessions:
            if session.worker:
                session.worker.join(timeout=JOB_SECONDS+5)
        self.janitor.join(timeout=2)
        shutil.rmtree(self.directory, ignore_errors=True)


def html_page():
    page = (STATIC / 'index.html').read_text()
    page = page.replace('<span class="format">HDF5 · single chirp</span>', '<span class="format">Online demo · HDF5</span>')
    page = page.replace('<div class="row"><label for="folder">', '<div class="row" hidden><label for="folder">')
    page = page.replace('>Open folder</button>', '>Download HDF5</button>')
    page = page.replace('id="quit" type="button"', 'id="quit" type="button" hidden')
    page = page.replace('<fieldset class="dataset-settings">', '<p class="demo-note">Try the settings and preview signals, then generate and download an HDF5 file. Online limit: 200 scenes and 32 MiB of waveform data. Files expire after 15 minutes without activity. For larger datasets, use the <a href="https://github.com/walidhirouche/fmcw-mitigation-detection" target="_blank" rel="noopener noreferrer">local generator</a>.</p><fieldset class="dataset-settings">')
    return page


def javascript():
    code = (STATIC / 'app.js').read_text()
    code = code.replace("node.label=group==='principal'?'Main · 24,000 scenes':'External · 6,000 scenes';",
                        "node.label=group==='principal'?'Main radar presets · demo counts':'External radar presets · demo counts';")
    return code + '\n' + (Path(__file__).parent / 'web.js').read_text()


def create_server(origin, *, host='127.0.0.1', port=7860, directory=None):
    app = Demo(origin, directory)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def allowed(self):
            return self.headers.get('Host') == app.host

        def send(self, status, data, kind='application/json; charset=utf-8'):
            body = json.dumps(json_safe(data), ensure_ascii=False, allow_nan=False).encode() if kind.startswith('application/json') else data
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'self' https://huggingface.co; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def download(self, key):
            with app.lock:
                match = next((s for s in app.sessions.values() if s.download_key and secrets.compare_digest(s.download_key, key)), None)
                if match is None:
                    return self.send(404, dict(error='File not found or expired.'))
                with match.lock:
                    if match.job['status'] != 'complete' or not match.file or not match.file.is_file():
                        return self.send(404, dict(error='File not found or expired.'))
                    stream = match.file.open('rb')
                    size = match.file.stat().st_size
                    name = match.file.name
                    match.readers += 1
                    match.last_used = time.monotonic()
            try:
                with stream:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-hdf5')
                    self.send_header('Content-Disposition', f'attachment; filename="{name}"')
                    self.send_header('Content-Length', str(size))
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Referrer-Policy', 'no-referrer')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers()
                    shutil.copyfileobj(stream, self.wfile, length=65536)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            finally:
                with match.lock:
                    match.readers -= 1

        def do_GET(self):
            # A platform health check may use an internal hostname.
            if self.path == '/health':
                return self.send(200, dict(status='ok'))
            if not self.allowed():
                return self.send(403, dict(error='Host not allowed.'))
            path = urlparse(self.path).path
            if path.startswith('/download/'):
                return self.download(path[len('/download/'):])
            if path == '/api/status':
                try:
                    return self.send(200, app.status(app.session(self.headers.get('X-Local-Token'))))
                except ValueError as error:
                    return self.send(403, dict(error=str(error)))
            if path == '/':
                try:
                    key = app.new_session()
                    return self.send(200, html_page().replace('__LOCAL_TOKEN__', key).encode(), 'text/html; charset=utf-8')
                except ValueError as error:
                    return self.send(503, dict(error=str(error)))
            if path == '/app.js':
                return self.send(200, javascript().encode(), 'text/javascript; charset=utf-8')
            if path == '/style.css':
                extra = '\n.demo-note{margin:0 0 10px;font-size:12px;line-height:1.5;color:#59464a}.demo-note a{color:#9d2638}[hidden]{display:none!important}'
                return self.send(200, ((STATIC/'style.css').read_text()+extra).encode(), 'text/css; charset=utf-8')
            if path == '/favicon.svg':
                return self.send(200, (STATIC/'favicon.svg').read_bytes(), 'image/svg+xml')
            return self.send(404, dict(error='Not found.'))

        def do_POST(self):
            if not self.allowed() or self.headers.get('Origin') not in (None, app.origin):
                return self.send(403, dict(error='Origin not allowed.'))
            try:
                session = app.session(self.headers.get('X-Local-Token'))
            except ValueError as error:
                return self.send(403, dict(error=str(error)))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 128000:
                    raise ValueError('Request is empty or too large.')
                body = json.loads(self.rfile.read(size))
                path = urlparse(self.path).path
                if path == '/api/defaults':
                    choices = presets()
                    for choice in choices:
                        choice['config']['count'] = 100
                        choice['config']['folder'] = ''
                    config = deepcopy(choices[0]['config'])
                    return self.send(200, dict(config=config, domain=domain(config), radar=radar_values(config), presets=choices))
                if path == '/api/validate':
                    config = demo_config(body, export=True)
                    return self.send(200, dict(domain=domain(config), radar=radar_values(config), signal_bytes_max=config['count']*domain(config)['max_points']*24))
                if path == '/api/domain':
                    config = defaults()
                    config['radar'].update(body['radar'])
                    for key in ('duration_us','bandwidth_mhz','sample_rate_mhz','frequency_ghz'):
                        bounds(config['radar'][key], key, positive=True)
                    if config['radar']['reception'] not in ('symmetric', 'unilateral') or config['radar']['frequency_reference'] not in ('start','center','end'):
                        raise ValueError('Unknown radar mode.')
                    return self.send(200, dict(domain=domain(config), radar=radar_values(config)))
                if path == '/api/preview':
                    config = demo_config(body['config'])
                    index = body.get('index', 0)
                    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 2**31:
                        raise ValueError('Preview index is out of range.')
                    if not app.preview_slot.acquire(blocking=False):
                        raise ValueError('Another preview is being calculated. Try again shortly.')
                    try:
                        return self.send(200, preview(config, index, body.get('window', 'rectangular'), body.get('cfar')))
                    finally:
                        app.preview_slot.release()
                if path == '/api/generate':
                    return self.send(202, app.start(session, body))
                if path == '/api/cancel':
                    session.cancel.set()
                    return self.send(200, app.status(session))
                return self.send(404, dict(error='Unknown action.'))
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                return self.send(400, dict(error=str(error)))
            except Exception:
                return self.send(500, dict(error='The request could not be completed. Check the settings and try again.'))
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except Exception:
        app.close()
        raise
    server.app = app
    return server


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '7860'))
    origin = os.environ.get('PUBLIC_ORIGIN') or os.environ.get('RENDER_EXTERNAL_URL')
    if not origin and os.environ.get('SPACE_HOST'):
        origin = 'https://' + os.environ['SPACE_HOST']
    origin = origin or f'http://127.0.0.1:{port}'
    server = create_server(origin, host=os.environ.get('BIND_HOST', '127.0.0.1'), port=port)
    print(f'FMCW online demo: {origin}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.app.close()
