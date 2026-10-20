# Online generator demo

The same red interface, waveform generator, preview and HDF5 writer, with
temporary downloads instead of local folder access. Scientific generation
code is unchanged. No training or model inference runs in this demo.

Visitors open the HTTPS URL, choose settings, preview, generate, and download
the HDF5 file. A download starts when generation finishes. The Download HDF5
button remains available; supported browsers also offer a save-as dialog on
that click. Other browsers follow their normal download settings.

For a CPU demonstration, the limits are 200 scenes, 8,192 samples per scene,
32 objects, 32 interfering ramps per observation, 32 MiB of waveform data and
60 seconds per generation. The complete HDF5 file may include metadata overhead
and is limited to 48 MiB. The nine presets use 100 scenes for the online demo;
the original local generation presets retain their original counts and seeds.
One export and one preview can run at a time. Visitors have separate sessions;
idle sessions and downloads expire after 15 minutes. Reloading opens a new
session. The local generator supports larger jobs.

## Local check

From the repository root:

```bash
python -m pip install -r dataset/requirements.txt
python demo/server.py
```

Open `http://127.0.0.1:7860`.

## Hosting behind HTTPS

For a Render deployment, choose **New > Web Service**, connect this GitHub
repository, leave the root directory empty, and use these settings:

| Field | Value |
| --- | --- |
| Language | Python 3 |
| Build command | `python -m pip install -r dataset/requirements.txt` |
| Start command | `python demo/server.py` |
| Instance | Free |
| Environment | `BIND_HOST=0.0.0.0`, `PYTHON_VERSION=3.12.12` |
| Health check | `/health` |

The public origin is read automatically from `RENDER_EXTERNAL_URL` and the
port from `PORT`. The same settings are in `demo/render.yaml` if you prefer
a Blueprint with that path. These files must be pushed to GitHub first.
The URL is assigned by Render after deployment; there is no live demo yet.

The [Render free tier](https://render.com/docs/free) sleeps after 15 minutes
without traffic. Waking it can take about a minute. Restarts or sleep delete
the temporary files, so visitors should download their exports immediately.
Monthly usage limits apply. Without a payment method, exceeding included
bandwidth suspends free services instead of billing for extra bandwidth.

### Other hosts

Set `PUBLIC_ORIGIN` to the exact public origin, such as `https://demo.example.org`,
and `BIND_HOST=0.0.0.0`. Use one server process; the session state is held in memory.
HTTPS is supplied by the hosting platform or reverse proxy.

```bash
docker build -f demo/Dockerfile -t fmcw-demo .
docker run --rm -p 7860:7860 -e PUBLIC_ORIGIN=https://demo.example.org fmcw-demo
```

On a Hugging Face Docker Space, `SPACE_HOST` supplies the public hostname.
No GPU is needed. Check the host's current account requirements and pricing
before creating a service. Nothing in this folder deploys or starts billing.

## Checks

```bash
python -m unittest discover -s demo -p test_server.py -v
```

These tests run the real HTTP handlers in memory, generate small HDF5 files,
and compare signals and annotations against the local generator. They also
check session separation, request validation, cancellation, expiration and
download headers. They do not replace a browser check on the deployed host.
