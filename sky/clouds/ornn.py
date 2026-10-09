"""Ornn cloud for SkyPilot.

Ornn (https://compute.ornn.com) rents GPU VMs by the hour on a **spot
market**: per-GPU-type order books, 1-8 GPUs per bid on ONE node, prepaid
per hour with part-hour refunds to credits. There is NO REST API key — the
automation surface is the hosted MCP server (JSON-RPC over HTTP, OAuth
credential in ``ORNN_MCP_CREDENTIALS``).

Instance types are stable **tokens** ``ornn:<gpu-slug>:<gpus>`` (e.g.
``ornn:nvidia_rtx_pro_6000:1``): order/reservation ids are ephemeral, so the
catalog rows and the provisioner communicate through the slug+count pair
and the provisioner re-resolves the live book at launch (the QuantaCloud
pattern). ``Region`` is the single constant ``ornn`` (the bid API takes no
region — ``gpuSlug`` + ``gpus`` only); there is no zone below it.

Modeled on ``sky/clouds/latitude.py`` (registered in-tree via
``@registry.CLOUD_REGISTRY.register``). Deliberate differences, each grounded
in the Ornn MCP receipts (ENG-541, 2026-10-08):

* **STOP is unsupported.** The only teardown is ``ornn_spot_cancel`` (it
  ENDS the reservation and destroys the VM + its local disk) — a stop that
  silently destroys is refused, not implemented.
* **Spot is unsupported** in the SkyPilot sense: Ornn's market fills are
  prepaid-hourly isolated VMs, not an interruptible/preemptible tier.
* **SSH keys are account-level**: spot VMs auto-attach EVERY active account
  key at launch; the provisioner ensures the deployment's key exists as an
  account key (matched by material) before bidding.
* **The image is not selectable**: a filled bid auto-starts a single
  isolated VM from the org's first custom image (or Ornn's default). The
  bootstrap ladder installs the pinned CUDA runtime itself, exactly like
  the Spheron/Vast/Latitude lanes.
"""

from __future__ import annotations

import os
import typing
from typing import Any, Dict, List, Optional, Tuple, Union

from sky import clouds
from sky.catalog import ornn_catalog
from sky.utils import registry
from sky.utils import resources_utils

if typing.TYPE_CHECKING:
    from sky import resources as resources_lib

# The credential the provisioner needs on the machine running SkyPilot (the
# dev-machine path; the controller pod materializes the env form).
_CREDENTIAL_FILES = ["mcp-credentials"]

# The region token is a constant: the bid API takes no region.
REGION = "ornn"


@registry.CLOUD_REGISTRY.register
class Ornn(clouds.Cloud):
    """Ornn spot-market GPU cloud."""

    # `_REPR` IS THE CATALOG NAME, not a display string. `Cloud.validate_region_zone`
    # passes `clouds=cls._REPR.lower()`, which SkyPilot turns into
    # `sky.catalog.{name}_catalog`. Inheriting the ABC's placeholder gives
    # `'<cloud>'` and every launch dies with
    #     ValueError: Cannot find module "sky.catalog.<cloud>_catalog"
    # — the bug that killed every Shadeform launch (see the skypilot-controller
    # Dockerfile's fork-pin comment).
    _REPR = "Ornn"

    _MAX_CLUSTER_NAME_LEN_LIMIT = 120

    _CLOUD_UNSUPPORTED_FEATURES = {
        clouds.CloudImplementationFeatures.STOP:
            ("ornn_spot_cancel is the only teardown and it destroys the VM "
             "and its local disk — a stop would silently destroy, so it is "
             "refused rather than implemented."),
        clouds.CloudImplementationFeatures.SPOT_INSTANCE:
            ("Ornn spot bids are prepaid-hourly isolated VMs, not an "
             "interruptible/preemptible tier."),
        clouds.CloudImplementationFeatures.MULTI_NODE:
            ("A bid rents 1-8 GPUs on ONE node; multi-node clusters are not "
             "supported."),
        clouds.CloudImplementationFeatures.CUSTOM_DISK_TIER:
            ("Spot VMs ship the node's fixed drives; disk tier is not "
             "selectable."),
        clouds.CloudImplementationFeatures.CUSTOM_NETWORK_TIER:
            ("Network tier is not selectable on Ornn."),
        clouds.CloudImplementationFeatures.STORAGE_MOUNTING:
            ("Object storage mounting is not supported on Ornn spot VMs."),
        clouds.CloudImplementationFeatures.HOST_CONTROLLERS:
            ("Host controllers are not supported on Ornn."),
        clouds.CloudImplementationFeatures.HIGH_AVAILABILITY_CONTROLLERS:
            ("High availability controllers are not supported on Ornn."),
        clouds.CloudImplementationFeatures.CLONE_DISK_FROM_CLUSTER:
            ("Disk cloning is not supported on Ornn (teardown destroys the "
             "local disk)."),
        clouds.CloudImplementationFeatures.IMAGE_ID:
            ("A filled bid auto-starts from the org's first custom image (or "
             "Ornn's default); an arbitrary image id cannot be requested on "
             "the spot surface."),
        clouds.CloudImplementationFeatures.DOCKER_IMAGE:
            ("Spot VMs boot the OS image directly; there is no docker image "
             "selection on Ornn."),
        clouds.CloudImplementationFeatures.CUSTOM_MULTI_NETWORK:
            ("Custom multi-network is not supported on Ornn."),
        clouds.CloudImplementationFeatures.LOCAL_DISK:
            ("Local disk is not selectable on Ornn."),
    }

    PROVISIONER_VERSION = clouds.ProvisionerVersion.SKYPILOT
    STATUS_VERSION = clouds.StatusVersion.SKYPILOT
    OPEN_PORTS_VERSION = clouds.OpenPortsVersion.LAUNCH_ONLY

    @classmethod
    def _unsupported_features_for_resources(
        cls,
        resources: "resources_lib.Resources",
        region: Optional[str] = None,
    ) -> Dict[clouds.CloudImplementationFeatures, str]:
        del resources, region
        return cls._CLOUD_UNSUPPORTED_FEATURES

    @classmethod
    def _max_cluster_name_length(cls) -> Optional[int]:
        return cls._MAX_CLUSTER_NAME_LEN_LIMIT

    def __repr__(self):
        return "Ornn"

    # -- catalog-backed lookups -------------------------------------------

    @classmethod
    def regions_with_offering(
        cls,
        instance_type: str,
        accelerators: Optional[Dict[str, int]],
        use_spot: bool,
        region: Optional[str],
        zone: Optional[str],
        resources: Optional["resources_lib.Resources"] = None,
    ) -> List[clouds.Region]:
        del accelerators, resources
        assert zone is None, "Ornn does not support zones."
        regions = ornn_catalog.get_region_zones_for_instance_type(
            instance_type, use_spot)
        if region is not None:
            regions = [r for r in regions if r.name == region]
        return regions

    @classmethod
    def zones_provision_loop(
        cls,
        *,
        region: str,
        num_nodes: int,
        instance_type: str,
        accelerators: Optional[Dict[str, int]] = None,
        use_spot: bool = False,
    ) -> typing.Iterator[None]:
        # One yield PER region that has an offering (the base contract): a
        # single yield for "any region" turned a first-region capacity
        # failure into the end of provisioning (urun-sh/skypilot#5/#6).
        del num_nodes
        regions = cls.regions_with_offering(instance_type,
                                            accelerators,
                                            use_spot,
                                            zone=None,
                                            region=region)
        for r in regions:
            assert r.zones is None, r
            yield r.zones

    @classmethod
    def get_vcpus_mem_from_instance_type(
            cls, instance_type: str) -> Tuple[Optional[float], Optional[float]]:
        return ornn_catalog.get_vcpus_mem_from_instance_type(instance_type)

    @classmethod
    def get_accelerators_from_instance_type(
            cls, instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
        return ornn_catalog.get_accelerators_from_instance_type(instance_type)

    @classmethod
    def get_default_instance_type(
        cls,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[Any] = None,
        local_disk: Optional[Any] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None,
    ) -> Optional[str]:
        del disk_tier, local_disk
        return ornn_catalog.get_default_instance_type(
            cpus=cpus,
            memory=memory,
            region=region,
            zone=zone,
            use_spot=use_spot,
            max_hourly_cost=max_hourly_cost,
        )

    def instance_type_exists(self, instance_type: str) -> bool:
        return ornn_catalog.instance_type_exists(instance_type)

    def instance_type_to_hourly_cost(
        self,
        instance_type: str,
        use_spot: bool = False,
        region: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> float:
        return ornn_catalog.get_hourly_cost(instance_type,
                                            use_spot=use_spot,
                                            region=region,
                                            zone=zone)

    def accelerators_to_hourly_cost(
        self,
        accelerators: Dict[str, int],
        use_spot: bool = False,
        region: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> float:
        del accelerators, use_spot, region, zone
        # A bid's price is per GPU-hour and the catalog row already carries
        # ask × count (the whole-box hourly); there is no separate
        # per-GPU line item to add.
        return 0.0

    def get_egress_cost(self, num_gigabytes: float) -> float:
        del num_gigabytes
        # Ornn publishes no egress line item on the spot surface; revisit
        # if overage billing ever appears.
        return 0.0

    @classmethod
    def get_zone_shell_cmd(cls) -> Optional[str]:
        return None

    # -- identity / credentials -------------------------------------------

    @classmethod
    def get_user_identities(cls) -> Optional[List[List[str]]]:
        return None

    @classmethod
    def get_current_user_identity(cls) -> Optional[List[str]]:
        return None

    @classmethod
    def get_current_user_identity_str(cls) -> Optional[str]:
        return None

    def get_credential_file_mounts(self) -> Dict[str, str]:
        return {
            f"~/.ornn/{filename}": f"~/.ornn/{filename}"
            for filename in _CREDENTIAL_FILES
        }

    @classmethod
    def _check_compute_credentials(cls) -> Tuple[bool, Optional[str]]:
        """Verify the MCP OAuth credential answers identity_whoami."""
        # pylint: disable=import-outside-toplevel
        from sky.adaptors import ornn as api

        try:
            credentials = api.OrnnCredentials.from_env()
        except api.OrnnAuthError as exc:
            return False, str(exc)
        try:
            api.OrnnClient(credentials).whoami()
        except api.OrnnError as exc:
            return False, str(exc)
        return True, None

    @classmethod
    def check_credentials(
            cls, cloud_capability: clouds.CloudCapability
    ) -> Tuple[bool, Optional[str]]:
        """Check Ornn credentials for the requested capability.

        MUST accept ``cloud_capability``: ``sky check`` calls this with the
        capability positionally, so a no-arg override raises TypeError, the
        cloud is reported DISABLED, and every launch fails with "Task
        requires ornn which is not enabled" (the urun-sh/skypilot#4 class).
        """
        if cloud_capability == clouds.CloudCapability.COMPUTE:
            return cls._check_compute_credentials()
        return False, f"Ornn does not support {cloud_capability.value}."

    # -- provisioning ------------------------------------------------------

    def make_deploy_resources_variables(
        self,
        resources: "resources_lib.Resources",
        cluster_name: resources_utils.ClusterName,
        region: clouds.Region,
        zones: Optional[List[clouds.Zone]],
        num_nodes: int,
        dryrun: bool = False,
        volume_mounts: Optional[List[Any]] = None,
    ) -> Dict[str, Optional[str]]:
        del cluster_name, dryrun, volume_mounts, num_nodes
        assert zones is None, "Ornn does not support zones."
        resources = resources.assert_launchable()
        acc_dict = self.get_accelerators_from_instance_type(
            resources.instance_type)
        custom_resources = resources_utils.make_ray_custom_resources_str(
            acc_dict)
        # InstanceType is the stable ``ornn:<gpu-slug>:<gpus>`` token; the
        # provisioner re-parses it (last colon) and re-resolves the live
        # book at launch — order ids are ephemeral and never catalogued.
        return {
            "instance_type": resources.instance_type,
            "custom_resources": custom_resources,
            "region": region.name,
            "use_spot": resources.use_spot,
            "ornn_gpu_slug": ornn_catalog.gpu_slug_from_instance_type(
                resources.instance_type),
            "ornn_gpus": ornn_catalog.gpu_count_from_instance_type(
                resources.instance_type),
        }

    def _get_feasible_launchable_resources(
        self, resources: "resources_lib.Resources"
    ) -> resources_utils.FeasibleResources:
        if resources.instance_type is not None:
            assert resources.is_launchable(), resources
            resources = resources.copy(accelerators=None)
            return resources_utils.FeasibleResources([resources], [], None)

        def _make(instance_list):
            resource_list = []
            for instance_type in instance_list:
                r = resources.copy(
                    cloud=Ornn(),
                    instance_type=instance_type,
                    accelerators=None,
                    cpus=None,
                    memory=None,
                )
                resource_list.append(r)
            return resource_list

        accelerators = resources.accelerators
        if accelerators is None:
            default_instance_type = Ornn.get_default_instance_type(
                cpus=resources.cpus,
                memory=resources.memory,
                region=resources.region,
                zone=resources.zone,
                use_spot=resources.use_spot,
                max_hourly_cost=resources.max_hourly_cost,
            )
            if default_instance_type is None:
                return resources_utils.FeasibleResources([], [], None)
            return resources_utils.FeasibleResources(
                _make([default_instance_type]), [], None)

        assert len(accelerators) == 1, resources
        acc, acc_count = list(accelerators.items())[0]
        (instance_list,
         fuzzy_candidate_list) = ornn_catalog.get_instance_type_for_accelerator(
             acc,
             acc_count,
             use_spot=resources.use_spot,
             cpus=resources.cpus,
             memory=resources.memory,
             region=resources.region,
             zone=resources.zone,
         )
        if instance_list is None:
            return resources_utils.FeasibleResources([], fuzzy_candidate_list,
                                                     None)
        return resources_utils.FeasibleResources(_make(instance_list),
                                                 fuzzy_candidate_list, None)

    @classmethod
    def query_status(
        cls,
        name: str,
        tag_filters: Dict[str, str],
        region: Optional[str],
        zone: Optional[str],
        **kwargs,
    ) -> List[Any]:
        # STATUS_VERSION is SKYPILOT, so the provisioner's query_instances is
        # the authority and this path is not used.
        raise NotImplementedError(
            "Ornn uses StatusVersion.SKYPILOT; status comes from "
            "sky.provision.ornn.query_instances.")

    @classmethod
    def get_image_size(cls, image_id: str, region: Optional[str]) -> float:
        del image_id, region
        # The VM image is provider-side and its size is not exposed; 0.0
        # lets every image through (same policy as the Vast/Latitude
        # clouds).
        return 0.0
