"""Packet.ai configuration bootstrapping."""

from sky.provision import common


def bootstrap_instances(
        region: str, cluster_name: str,
        config: common.ProvisionConfig) -> common.ProvisionConfig:
    """Nothing to bootstrap.

    Packet.ai has no VPC, subnet, security group or firewall to prepare: a
    pod boots as-is with a public IP and direct SSH on a per-instance port
    (the stock image ships NVIDIA drivers + Docker already, measured CUDA
    13.0 / driver 580.126.09). Port exposure (when needed) rides the
    provider's NodePort surface (POST /api/v1/services) after the box is
    up, not at bootstrap.
    """
    del region, cluster_name  # unused
    return config
