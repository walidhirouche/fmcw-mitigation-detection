"""Application web exclusivement locale, sans framework ni service externe."""
import argparse
import json
import mimetypes
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from .window import ApplicationWindow
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from .config import defaults, domain, radar_values, validate
from .core import json_safe, preview
from .storage import Cancelled, write_dataset

STATIC = Path(__file__).with_name('static')


def presets():
    """Saved training/evaluation configurations, shared with the public recipe."""
    result=[]
    root=Path(__file__).with_name('configs')
    for group in ('principal','external'):
        for path in sorted((root/group).glob('*.json')):
            config=validate(json.loads(path.read_text()))
            result.append(dict(id=f'{group}/{path.stem}',group=group,config=config))
    return result


class App:
    def __init__(self):
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.job = dict(status='idle',done=0,total=0)
        self.worker = None
        self.default_folder = str(Path(__file__).resolve().parents[1]/'datasets')

    def status(self):
        with self.lock:
            return dict(self.job)

    def start(self, config):
        with self.lock:
            if self.job['status'] == 'running':
                raise ValueError('Generation is already running.')
            self.cancel.clear()
            self.job = dict(status='running',done=0,total=config['count'],elapsed=0,speed=0,eta=None)
            def update(state):
                with self.lock:
                    self.job.update(state)
            def work():
                try:
                    result = write_dataset(config,progress=update,cancel=self.cancel)
                    update(dict(status='complete',result=result))
                except Cancelled as exc:
                    update(dict(status='cancelled',message=str(exc)))
                except Exception as exc:
                    update(dict(status='error',message=str(exc)))
            self.worker = threading.Thread(target=work,name='generation-hdf5')
            self.worker.start()


def create_server(port=0):
    app = App()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def send(self,status,data,kind='application/json; charset=utf-8'):
            body = json.dumps(json_safe(data),ensure_ascii=False,allow_nan=False).encode() if kind.startswith('application/json') else data
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):
                pass

        def allowed(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

        def do_GET(self):
            if not self.allowed():
                return self.send(403,{'error':'Host not allowed.'})
            path = urlparse(self.path).path
            if path == '/api/status':
                if self.headers.get('X-Local-Token') != app.token:
                    return self.send(403,{'error':'Local session required.'})
                return self.send(200,app.status())
            files = {'/':'index.html','/app.js':'app.js','/style.css':'style.css','/favicon.svg':'favicon.svg'}
            if path not in files:
                return self.send(404,{'error':'Not found.'})
            file = STATIC/files[path]
            body = file.read_bytes()
            if path == '/':
                body = body.replace(b'__LOCAL_TOKEN__',app.token.encode())
            kind = mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
            return self.send(200,body,kind+'; charset=utf-8')

        def do_POST(self):
            if not self.allowed() or self.headers.get('X-Local-Token') != app.token:
                return self.send(403,{'error':'Local session required.'})
            origin = self.headers.get('Origin')
            if origin and origin != f'http://127.0.0.1:{self.server.server_port}':
                return self.send(403,{'error':'Origin not allowed.'})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 128000:
                    raise ValueError('Request is empty or too large.')
                body = json.loads(self.rfile.read(length))
                path = urlparse(self.path).path
                if path == '/api/defaults':
                    c = defaults()
                    c['folder'] = app.default_folder
                    return self.send(200,dict(config=c,domain=domain(c),radar=radar_values(c),presets=presets()))
                if path == '/api/validate':
                    c = validate(body)
                    d = domain(c)
                    return self.send(200,dict(domain=d,radar=radar_values(c),signal_bytes_max=c['count']*d['max_points']*24))
                if path == '/api/domain':
                    # La validation complète reste obligatoire avant toute synthèse.
                    c = defaults()
                    c['radar'].update(body['radar'])
                    for key in ('duration_us','bandwidth_mhz','sample_rate_mhz','frequency_ghz'):
                        from .config import bounds
                        bounds(c['radar'][key],key,positive=True)
                    return self.send(200,dict(domain=domain(c),radar=radar_values(c)))
                if path == '/api/preview':
                    c = validate(body['config'])
                    index = int(body.get('index',0))
                    if not 0 <= index < 2**31:
                        raise ValueError('Preview index is out of range.')
                    if domain(c)['max_points'] > 100000:
                        raise ValueError('Preview limit: 100,000 samples. HDF5 export supports up to 2,000,000.')
                    if domain(c)['max_points']*(c['targets']['count'][1]+c['neighbors']['count'][1]) > 20_000_000:
                        raise ValueError('Preview too large: reduce objects or samples. HDF5 export is still available.')
                    return self.send(200,preview(c,index,body.get('window','rectangular'),body.get('cfar')))
                if path == '/api/generate':
                    app.start(validate(body))
                    return self.send(202,app.status())
                if path == '/api/cancel':
                    app.cancel.set()
                    return self.send(200,app.status())
                if path == '/api/folders':
                    folder = Path(body.get('path') or Path.home()).expanduser().resolve()
                    if not folder.is_dir():
                        raise ValueError('Folder not found.')
                    entries=[]
                    for item in sorted(folder.iterdir(),key=lambda p:p.name.casefold()):
                        try:
                            if item.is_dir() and not item.name.startswith('.'):
                                entries.append(dict(name=item.name,path=str(item)))
                        except OSError:
                            continue
                    return self.send(200,dict(path=str(folder),parent=str(folder.parent),entries=entries))
                if path == '/api/open-folder':
                    result = app.status().get('result')
                    if not result:
                        raise ValueError('No completed export.')
                    folder = str(Path(result['path']).parent)
                    cmd = ['open',folder] if sys.platform=='darwin' else (['explorer',folder] if sys.platform=='win32' else ['xdg-open',folder])
                    subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    return self.send(200,dict(ok=True))
                if path == '/api/quit':
                    app.cancel.set()
                    self.send(200,dict(ok=True))
                    threading.Thread(target=self.server.shutdown,daemon=True).start()
                    return
                return self.send(404,{'error':'Unknown action.'})
            except (ValueError,KeyError,TypeError,OverflowError,OSError) as exc:
                self.send(400,{'error':str(exc)})
            except Exception as exc:
                self.send(500,{'error':f'Error: {exc}'})
    server = ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.app = app
    return server


def main():
    parser = argparse.ArgumentParser(description='FMCW dataset generator — local application')
    parser.add_argument('--port',type=int,default=0,help='0: choose a free local port')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--no-browser',action='store_true',help='Server only, without a window')
    mode.add_argument('--browser',action='store_true',help='Open the default browser instead of a dedicated window')
    args = parser.parse_args()
    server = create_server(args.port)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'FMCW Dataset Generator: {url}',flush=True)
    window = None
    serve_thread = threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.2},name='serveur-local')
    serve_thread.start()
    try:
        if args.browser:
            webbrowser.open(url)
        elif not args.no_browser:
            window = ApplicationWindow(url)
        while serve_thread.is_alive():
            serve_thread.join(timeout=.2)
            if window is not None and window.process.poll() is not None:
                server.shutdown()
                if window.process.returncode:
                    print('Chromium closed with an error. Retry with --browser.',file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        serve_thread.join()
        server.app.cancel.set()
        if server.app.worker:
            server.app.worker.join()
        if window is not None:
            window.close()
        server.server_close()


if __name__ == '__main__':
    main()
