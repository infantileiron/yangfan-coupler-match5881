"""Point the test suite at a throwaway SQLite database before any src import."""
import os
import tempfile
from pathlib import Path

_SCRATCH = tempfile.mkdtemp(prefix='coupler-tests-')
os.environ.setdefault('DATABASE_URL', 'sqlite:///' + str(Path(_SCRATCH) / 'tests.db'))
os.environ.setdefault('REDIS_URL', 'redis://127.0.0.1:6399/0')
os.environ.setdefault('SEED_DEMO', 'false')
