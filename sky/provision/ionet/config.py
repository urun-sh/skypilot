"""io.net configuration bootstrapping."""

from sky.provision import common


def bootstrap_instances(
        region: str, cluster_name: str,
        config: common.ProvisionConfig) -> common.ProvisionConfig:
    """Nothing to bootstrap.

    io.net has no VPC, subnet, security group or firewall to prepare: a
    deployment boots as-is, with SSH on port 22 and any extra ports
    opened through the deploy body's ``network_services`` map (the
    provider configures them during the deploy itself; 'general' ships
    CUDA + NVIDIA drivers already).
    """
    del region, cluster_name  # unused
    return config