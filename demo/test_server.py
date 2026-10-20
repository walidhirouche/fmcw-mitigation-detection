"""Exercise real HTTP handlers in memory and export real, small HDF5 files."""
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import server as demo


class MemoryServer:
    def __init__(self, address, handler):
        self.handler = handler


class Connection:
    def __init__(self, request):
        self.input = io.BytesIO(request)
        self.output = bytearray()

    def makefile(self, *args, **kwargs):
        return self.input

    def sendall(self, data):
        self.output.extend(data)

    def settimeout(self, value):
        pass


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        with patch.object(demo, 'ThreadingHTTPServer', MemoryServer):
            self.server = demo.create_server('https://demo.test', directory=self.temp.name)
        self.app = self.server.app
        self.key = self.new_visitor()

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    def request(self, path, data=None, *, token=None, headers=None):
        head = dict(Host='demo.test', **{'X-Local-Token': token or self.key})
        head.update(headers or {})
        body = json.dumps(data).encode() if data is not None else b''
        method = 'POST' if data is not None else 'GET'
        head['Content-Length'] = str(len(body))
        wire = f'{method} {path} HTTP/1.0\r\n'
        wire += ''.join(f'{key}: {value}\r\n' for key, value in head.items()) + '\r\n'
        connection = Connection(wire.encode() + body)
        self.server.handler(connection, ('127.0.0.1', 1234), self.server)
        header, body = bytes(connection.output).split(b'\r\n\r\n', 1)
        lines = header.decode().split('\r\n')
        fields = dict(line.split(': ', 1) for line in lines[1:])
        return int(lines[0].split()[1]), fields, body

    def new_visitor(self):
        # request() is also used before setUp has assigned self.key.
        if not hasattr(self, 'key'):
            self.key = 'none'
        code, _, html = self.request('/')
        self.assertEqual(code, 200)
        return re.search(rb'name="local-token" content="([^"]+)"', html)[1].decode()

    def wait(self, key=None):
        session = self.app.session(key or self.key)
        session.worker.join(timeout=10)
        self.assertFalse(session.worker.is_alive())
        return self.app.status(session)

    def config(self):
        config = demo.defaults()
        config['count'] = 2
        return config

    def test_download_matches_local_generator_and_visitors_are_separate(self):
        other = self.new_visitor()
        config = self.config()
        config['folder'] = '/a/client/path/that/must/not/be/used'
        self.assertEqual(self.request('/api/generate', config)[0], 202)
        job = self.wait()
        self.assertEqual(job['status'], 'complete', job)
        self.assertEqual(self.app.status(self.app.session(other))['status'], 'idle')
        code, headers, data = self.request(job['result']['download_url'], token=other)
        self.assertEqual(code, 200)  # An unguessable download URL is a shareable capability.
        self.assertIn('attachment; filename="dataset.h5"', headers['Content-Disposition'])
        self.assertEqual(len(data), job['result']['bytes'])
        self.assertNotIn(self.temp.name, json.dumps(job))
        local = demo.write_dataset(demo.demo_config(config), folder=Path(self.temp.name)/'local')
        with h5py.File(io.BytesIO(data)) as actual, h5py.File(local['path']) as expected:
            self.assertEqual(actual.attrs['complete'], True)
            self.assertEqual(actual['config'][()], expected['config'][()])
            for group in ('s', 'sn', 'y'):
                self.assertEqual(set(actual[group]), set(expected[group]))
                for length in actual[group]:
                    np.testing.assert_array_equal(actual[group][length][:], expected[group][length][:])
            for dataset in ('label', 'voisins', 'info'):
                for field in actual[dataset].dtype.names:
                    np.testing.assert_array_equal(actual[dataset][field], expected[dataset][field])
        # A second export replaces only this visitor's previous file/link.
        self.assertEqual(self.request('/api/generate', self.config())[0], 202)
        self.assertEqual(self.wait()['status'], 'complete')
        self.assertEqual(self.request(job['result']['download_url'])[0], 404)

    def test_host_origin_token_and_local_actions(self):
        self.assertEqual(self.request('/api/status', token='invalid')[0], 403)
        self.assertEqual(self.request('/', headers={'Host': 'elsewhere.test'})[0], 403)
        self.assertEqual(self.request('/health', headers={'Host': 'localhost'})[0], 200)
        self.assertEqual(self.request('/api/generate', self.config(), headers={'Origin': 'https://elsewhere.test'})[0], 403)
        for action in ('folders', 'open-folder', 'quit'):
            self.assertEqual(self.request('/api/'+action, {})[0], 404)
        for path in ('/../../README.md', '/dataset/partitions.json', '/download/guessed'):
            self.assertEqual(self.request(path)[0], 404)

    def test_limits_and_all_presets(self):
        code, _, data = self.request('/api/defaults', {})
        self.assertEqual(code, 200)
        presets = json.loads(data)['presets']
        self.assertEqual(len(presets), 9)
        for preset in presets:
            self.assertEqual(preset['config']['count'], 100)
            demo.demo_config(preset['config'], export=True)
        for key, value in [('count', 201), ('name', '../escape'), ('name', 'a\r\nheader')]:
            config = self.config()
            config[key] = value
            self.assertEqual(self.request('/api/generate', config)[0], 400)
        config = self.config()
        config['radar']['sample_rate_mhz'] = [1000, 1000]
        self.assertEqual(self.request('/api/generate', config)[0], 400)
        config = self.config()
        config['neighbors']['slope_factor'] = [1, 100]
        self.assertEqual(self.request('/api/generate', config)[0], 400)
        # Busy requests do not start an extra worker.
        self.app.generation_slot.acquire()
        try:
            self.assertEqual(self.request('/api/generate', self.config())[0], 400)
        finally:
            self.app.generation_slot.release()

    def test_preview_and_static_files(self):
        code, _, data = self.request('/api/preview', dict(config=self.config(), index=0))
        self.assertEqual(code, 200, data[:200])
        self.assertIn('cfar', json.loads(data))
        for path in ('/app.js', '/style.css'):
            code, headers, content = self.request(path)
            self.assertEqual(code, 200)
            self.assertTrue(content)
            self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')

    def test_cancel_and_error_release_generation_slot(self):
        entered = threading.Event()
        def cancellable(*args, cancel, **kwargs):
            entered.set()
            cancel.wait(3)
            raise demo.Cancelled('cancelled')
        with patch.object(demo, 'write_dataset', cancellable):
            self.assertEqual(self.request('/api/generate', self.config())[0], 202)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.request('/api/generate', self.config())[0], 400)
            self.assertEqual(self.request('/api/cancel', {})[0], 200)
            self.assertEqual(self.wait()['status'], 'cancelled')
        with patch.object(demo, 'write_dataset', side_effect=RuntimeError('private details')):
            self.assertEqual(self.request('/api/generate', self.config())[0], 202)
            job = self.wait()
            self.assertEqual(job['status'], 'error')
            self.assertNotIn('private details', json.dumps(job))
        self.assertEqual(self.request('/api/generate', self.config())[0], 202)
        self.assertEqual(self.wait()['status'], 'complete')

    def test_expiry_removes_files_and_access(self):
        self.request('/api/generate', self.config())
        job = self.wait()
        session = self.app.session(self.key)
        session.last_used = time.monotonic()-demo.SESSION_SECONDS-1
        self.app.cleanup()
        self.assertFalse(session.directory.exists())
        self.assertEqual(self.request('/api/status')[0], 403)
        self.assertEqual(self.request(job['result']['download_url'])[0], 404)


if __name__ == '__main__':
    unittest.main()
