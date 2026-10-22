"""MassedCompute provisioner package.

Exports the exact surface SkyPilot's dispatch looks for, mirroring
``sky/provision/quantacloud/__init__.py``.
"""

from sky.provision.massed.config import bootstrap_instances
from sky.provision.massed.instance import cleanup_ports
from sky.provision.massed.instance import get_cluster_info
from sky.provision.massed.instance import open_ports
from sky.provision.massed.instance import query_instances
from sky.provision.massed.instance import run_instances
from sky.provision.massed.instance import stop_instances
from sky.provision.massed.instance import terminate_instances
from sky.provision.massed.instance import wait_instances

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