"""Ornn configuration bootstrapping."""

from sky.provision import common


def bootstrap_instances(
        region: str, cluster_name: str,
        config: common.ProvisionConfig) -> common.ProvisionConfig:
    """Nothing to bootstrap.

    Ornn spot VMs auto-launch from a filled bid with every active account
    SSH key attached; there is no VPC, subnet, security group or firewall to
    prepare, and the bid API takes no region (``gpuSlug`` + ``gpus`` only).
    """
    del region, cluster_name  # unused
    return config
