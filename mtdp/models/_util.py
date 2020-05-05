import hashlib
import os
import re
import shutil
import sys
import tempfile

import torch
from torch.utils.model_zoo import _download_url_to_file

try:
    from requests.utils import urlparse
    from requests import get as urlopen
    requests_available = True
except ImportError:
    requests_available = False
    from urllib.request import urlopen
    from urllib.parse import urlparse
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # defined below


def _remove_prefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s


def clean_state_dict(state_dict, prefix, filter=None):
    if filter is None:
        filter = lambda *args: True
    return {_remove_prefix(k, prefix): v for k, v in state_dict.items() if filter(k)}


def load_dox_url(url, filename, model_dir=None, map_location=None, progress=True):
    r"""Adapt to fit format file of mtdp pre-trained models
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        sys.stderr.flush()
        _download_url_to_file(url, cached_file, None, progress=progress)
    return torch.load(cached_file, map_location=map_location)