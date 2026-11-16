"""Verda provisioner package.

Exports the exact surface SkyPilot's dispatch looks for, mirroring
``sky/provision/quantacloud/__init__.py``.
"""

from sky.provision.verda.config import bootstrap_instances
from sky.provision.verda.instance import cleanup_ports
from sky.provision.verda.instance import get_cluster_info
from sky.provision.verda.instance import open_ports
from sky.provision.verda.instance import query_instances
from sky.provision.verda.instance import run_instances
from sky.provision.verda.instance import stop_instances
from sky.provision.verda.instance import terminate_instances
from sky.provision.verda.instance import wait_instances

__all__ = [
    'bootstrap_instances',
    'cleanup_ports',
    'get_cluster_info',
    'open_ports',
    'query_instances',
    'run_instances',
    'stop_instances',
    'terminate_instances',
    'wait_instances',
]
