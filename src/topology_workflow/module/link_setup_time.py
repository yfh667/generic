"""Compatibility shim for link setup time helpers.

The reusable implementation now lives in :mod:`src.link_setup_time.module`.
This module remains so existing topology workflow imports continue to work.
"""

from src.link_setup_time.module.core import *  # noqa: F401,F403
