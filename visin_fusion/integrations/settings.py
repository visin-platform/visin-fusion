"""Caller-owned Visin configuration; the installed package never stores credentials."""

import os
from pathlib import Path


def env_file():
    """An explicit external file, or the current application's .env if present."""
    explicit = os.environ.get('VISIN_ENV_FILE')
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'VISIN_ENV_FILE does not point to a file: {path}')
        return path
    path = Path.cwd() / '.env'
    return path if path.is_file() else None


def _token_in_file(path):
    """Recognize a file-based key even without the optional python-dotenv extra."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line.startswith('export '):
                line = line[len('export '):]
            name, separator, value = line.partition('=')
            if separator and name.strip() == 'VISIN_TOKEN':
                value = value.strip()
                if value and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                return bool(value.split(' #', 1)[0].strip())
        return False
    return bool(dotenv_values(path).get('VISIN_TOKEN'))


def pipeline_key_present():
    """Detect reporting configuration before importing the optional Visin extra."""
    path = env_file()
    return bool(os.environ.get('VISIN_TOKEN')) or (path is not None and _token_in_file(path))


def load_environment():
    """Load a caller-owned file without overriding variables already exported."""
    path = env_file()
    if path is not None:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
