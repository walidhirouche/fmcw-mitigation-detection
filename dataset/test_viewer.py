"""Saved scene browsing must use the exported arrays, including float32 rounding."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from fmcw_dataset.config import defaults
from fmcw_dataset.core import representation
from fmcw_dataset.storage import read_scene, write_dataset
from fmcw_dataset.viewer import saved_preview
from fmcw_dataset import server as local


class MemoryServer:
    def __init__(self,address,handler):
        self.handler=handler
        self.server_port=8765


class Connection:
    def __init__(self,request):
        self.input=io.BytesIO(request)
        self.output=bytearray()

    def makefile(self,*args,**kwargs):return self.input
    def sendall(self,data):self.output.extend(data)


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        config=defaults()
        config.update(count=3,folder=self.temp.name)
        config['radar']['duration_us']=[12.5,14]
        self.path=Path(write_dataset(config)['path'])

    def tearDown(self):self.temp.cleanup()

    def test_reads_each_saved_array_without_regeneration(self):
        # Alter a saved sample: regenerating from seed/config would miss this value.
        with h5py.File(self.path,'r+') as file:
            record=file['info'][1]
            file['y'][f'n{record["N"]}'][record['row'],0]=1.25+2.5j
        with patch('fmcw_dataset.core.generate_scene',side_effect=AssertionError('Must not regenerate')):
            for index in range(3):
                scene=read_scene(self.path,index)
                payload=saved_preview(self.path,index,'hann')
                rep=representation(scene,2,'hann')
                self.assertEqual(payload['dataset']['count'],3)
                self.assertEqual(payload['index'],index)
                self.assertEqual(payload['targets'],scene.targets)
                for signal in ('s','sn','y'):
                    np.testing.assert_array_equal(payload['temporal'][signal]['real'],getattr(scene,signal).real)
                    np.testing.assert_array_equal(payload['temporal'][signal]['imag'],getattr(scene,signal).imag)
                    np.testing.assert_array_equal(payload['spectrum'][signal]['real'],rep[signal].real)
                    np.testing.assert_array_equal(payload['spectrum'][signal]['imag'],rep[signal].imag)

    def test_bad_indices_incomplete_file_and_large_scene(self):
        for index in (-1,3,True,0.5,'0'):
            with self.subTest(index=index),self.assertRaises(ValueError):saved_preview(self.path,index)
        with h5py.File(self.path,'r+') as file:file.attrs['complete']=False
        with self.assertRaisesRegex(ValueError,'incomplete'):saved_preview(self.path)
        with h5py.File(self.path,'r+') as file:
            file.attrs['complete']=True
            record=file['info'][0];record['N']=100001;file['info'][0]=record
        with self.assertRaisesRegex(ValueError,'100,000'):saved_preview(self.path)

    def test_empty_scene_and_local_http_listing(self):
        config=defaults();config.update(name='empty',count=1,folder=self.temp.name)
        config['profile']=[0,0,0,100]
        empty=Path(write_dataset(config)['path'])
        self.assertEqual(saved_preview(empty)['targets'],[])
        with patch.object(local,'ThreadingHTTPServer',MemoryServer):server=local.create_server()
        def request(path,data,token=None):
            body=json.dumps(data).encode()
            head=f'POST /api/{path} HTTP/1.0\r\nHost: 127.0.0.1:8765\r\nX-Local-Token: {token or server.app.token}\r\nContent-Length: {len(body)}\r\n\r\n'
            connection=Connection(head.encode()+body)
            server.handler(connection,('127.0.0.1',1234),server)
            header,data=bytes(connection.output).split(b'\r\n\r\n',1)
            return int(header.split()[1]),json.loads(data)
        code,listing=request('viewer/files',dict(folder=self.temp.name))
        self.assertEqual(code,200)
        self.assertEqual({f['name'] for f in listing['files']},{'dataset.h5','empty.h5'})
        self.assertEqual(request('viewer/scene',dict(file=str(self.path),index=2))[0],200)
        self.assertEqual(request('viewer/scene',dict(file=str(self.path),index=3))[0],400)
        self.assertEqual(request('viewer/scene',dict(file=str(self.path)),token='wrong')[0],403)


if __name__=='__main__':unittest.main()
