"""Fenêtre Chromium en mode application, sans utiliser le navigateur par défaut."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def chromium_executable():
    for command in ('chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable', 'microsoft-edge', 'msedge'):
        executable = shutil.which(command)
        if executable:
            return executable
    candidates = []
    if sys.platform == 'darwin':
        candidates = ['/Applications/Chromium.app/Contents/MacOS/Chromium',
                      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']
    elif sys.platform == 'win32':
        for variable in ('LOCALAPPDATA', 'PROGRAMFILES', 'PROGRAMFILES(X86)'):
            base = os.environ.get(variable)
            if base:
                candidates.extend([str(Path(base)/'Google/Chrome/Application/chrome.exe'),
                                   str(Path(base)/'Microsoft/Edge/Application/msedge.exe')])
    return next((p for p in candidates if Path(p).is_file()), None)


class ApplicationWindow:
    """Profil temporaire propre à cette instance, fermé/nettoyé avec l'application."""
    def __init__(self, url):
        executable = chromium_executable()
        if not executable:
            raise RuntimeError('Install Chromium/Chrome/Edge, or use --browser.')
        self.profile = tempfile.TemporaryDirectory(prefix='fmcw-window-',ignore_cleanup_errors=True)
        self.process = None
        try:
            self.process = subprocess.Popen([
                executable, f'--app={url}', f'--user-data-dir={self.profile.name}',
                '--no-first-run', '--no-default-browser-check', '--disable-background-mode',
                '--window-size=1120,1000', '--class=fmcw-dataset',
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            self.profile.cleanup()
            raise

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.profile.cleanup()
