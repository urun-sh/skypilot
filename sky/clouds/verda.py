"""Verda cloud for SkyPilot.

Verda (https://verda.com — the rebranded DataCrunch cloud) rents on-demand
NVIDIA GPU VMs against a project balance. Instances deploy from a
Verda instance-type code (``1H100.80S.22V`` = {GPU count}{MODEL}.{GPU
mem}S.{vCPUs}V) in a location (``FIN-01`` "Finland 1"), reached over
direct SSH.

Modeled on ``sky/clouds/quantacloud.py``. Deliberate differences, each
grounded in the Verda API (OpenAPI spec reviewed 2026-11-16):

* **Spot IS supported.** Verda has a genuine spot tier: per-location
  ``spot_price`` in ``location_prices`` and an ``is_spot`` deploy flag
  (unlike quantacloud, SpotPrice is emitted and honored).
* **STOP is unsupported.** The API's stop-shaped actions
  (``shutdown``/``hibernate``/``discontinue``) park a billing box in an
  unknown state; the lane terminates instead of pretending stop is safe.
* **Credentials are OAuth2 client credentials** (VERDA_CLIENT_ID +
  VERDA_CLIENT_SECRET, or ``~/.verda/credentials``), not an API key.
* **Region is the location NAME** (``"Finland 1"``): the catalog's
  convention; the provisioner maps the name back to the location code
  at launch via ``GET /v1/locations``.
"""

from __future__ import annotations

import os
import typing
from typing import Any, Dict, List, Optional, Tuple, Union

from sky import clouds
from sky.catalog import verda_catalog
from sky.utils import registry
from sky.utils import resources_utils

if typing.TYPE_CHECKING:
    from sky import resources as resources_lib

# The credential the provisioner needs on the machine running SkyPilot.
_CREDENTIAL_FILES = ['credentials']


@registry.CLOUD_REGISTRY.register
class Verda(clouds.Cloud):
    """Verda on-demand GPU VM cloud."""

    # `_REPR` IS THE CATALOG NAME, not a display string (see the
    # quantacloud twin for the failure mode it prevents).
    _REPR = 'Verda'

    _MAX_CLUSTER_NAME_LEN_LIMIT = 120

    _CLOUD_UNSUPPORTED_FEATURES = {
        clouds.CloudImplementationFeatures.STOP:
            ('The API stop-shaped actions (shutdown/hibernate/discontinue) '
             'park a billing box in an unknown state; stop is refused '
             'rather than silently destroy or keep billing.'),
        clouds.CloudImplementationFeatures.MULTI_NODE:
            ('Multi-node clusters are not supported on Verda.'),
        clouds.CloudImplementationFeatures.CUSTOM_DISK_TIER:
            ('Instances ship fixed included storage; disk tier is not '
             'selectable.'),
        clouds.CloudImplementationFeatures.CUSTOM_NETWORK_TIER:
            ('Network tier is not selectable on Verda.'),
        clouds.CloudImplementationFeatures.STORAGE_MOUNTING:
            ('Object storage mounting is not supported on Verda.'),
        clouds.CloudImplementationFeatures.HOST_CONTROLLERS:
            ('Host controllers are not supported on Verda.'),
        clouds.CloudImplementationFeatures.HIGH_AVAILABILITY_CONTROLLERS:
            ('High availability controllers are not supported on Verda.'),
        clouds.CloudImplementationFeatures.CLONE_DISK_FROM_CLUSTER:
            ('Disk cloning is not supported on Verda.'),
        clouds.CloudImplementationFeatures.IMAGE_ID:
            ('The lane deploys the fixed CUDA image; an arbitrary image '
             'id is not selectable.'),
        clouds.CloudImplementationFeatures.DOCKER_IMAGE:
            ('The stock image boots the OS directly; docker image '
             'selection is not supported on Verda.'),
        clouds.CloudImplementationFeatures.CUSTOM_MULTI_NETWORK:
            ('Custom multi-network is not supported on Verda.'),
        clouds.CloudImplementationFeatures.LOCAL_DISK:
            ('Local disk is not selectable on Verda.'),
    }

    PROVISIONER_VERSION = clouds.ProvisionerVersion.SKYPILOT
    STATUS_VERSION = clouds.StatusVersion.SKYPILOT
    OPEN_PORTS_VERSION = clouds.OpenPortsVersion.LAUNCH_ONLY

    @classmethod
    def _unsupported_features_for_resources(
        cls,
        resources: 'resources_lib.Resources',
        region: Optional[str] = None,
    ) -> Dict[clouds.CloudImplementationFeatures, str]:
        del resources, region
        return cls._CLOUD_UNSUPPORTED_FEATURES

    @classmethod
    def _max_cluster_name_length(cls) -> Optional[int]:
        return cls._MAX_CLUSTER_NAME_LEN_LIMIT

    def __repr__(self):
        return 'Verda'

    # -- catalog-backed lookups -------------------------------------------

    @classmethod
    def regions_with_offering(
        cls,
        instance_type: str,
        accelerators: Optional[Dict[str, int]],
        use_spot: bool,
        region: Optional[str],
        zone: Optional[str],
        resources: Optional['resources_lib.Resources'] = None,
    ) -> List[clouds.Region]:
        del accelerators, resources
        assert zone is None, 'Verda does not support zones.'
        regions = verda_catalog.get_region_zones_for_instance_type(
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
        # One yield PER region that has an offering (the base contract):
        # a single yield for "any region" turns a first-region capacity
        # failure into the end of provisioning (the Spheron lesson).
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
        return verda_catalog.get_vcpus_mem_from_instance_type(instance_type)

    @classmethod
    def get_accelerators_from_instance_type(
            cls, instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
        return verda_catalog.get_accelerators_from_instance_type(instance_type)

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
        return verda_catalog.get_default_instance_type(
            cpus=cpus,
            memory=memory,
            region=region,
            zone=zone,
            use_spot=use_spot,
            max_hourly_cost=max_hourly_cost,
        )

    def instance_type_exists(self, instance_type: str) -> bool:
        return verda_catalog.instance_type_exists(instance_type)

    def instance_type_to_hourly_cost(
        self,
        instance_type: str,
        use_spot: bool = False,
        region: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> float:
        return verda_catalog.get_hourly_cost(instance_type,
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
        # An instance's hourly price is for the whole box (GPUs included);
        # there is no separate per-GPU line item to add.
        return 0.0

    def get_egress_cost(self, num_gigabytes: float) -> float:
        del num_gigabytes
        # The spec publishes no egress charge line items; revisit if
        # overage billing ever appears.
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
            f'~/.verda/{filename}': f'~/.verda/{filename}'
            for filename in _CREDENTIAL_FILES
        }

    @classmethod
    def _check_compute_credentials(cls) -> Tuple[bool, Optional[str]]:
        """Verify we can talk to Verda (GET /v1/balance)."""
        # pylint: disable=import-outside-toplevel
        from sky.provision.verda import utils

        creds = utils.resolve_credentials()
        if creds is None:
            return False, ('Neither VERDA_CLIENT_ID/VERDA_CLIENT_SECRET nor '
                           '~/.verda/credentials is configured. Create OAuth2 '
                           'client credentials in the Verda console.')
        client_id, client_secret = creds
        try:
            utils.VerdaClient(client_id, client_secret).get_balance()
        except utils.VerdaError as exc:
            return False, str(exc)
        return True, None

    @classmethod
    def check_credentials(
            cls, cloud_capability: clouds.CloudCapability
    ) -> Tuple[bool, Optional[str]]:
        """Check Verda credentials for the requested capability.

        MUST accept ``cloud_capability``: ``sky check`` calls this with
        the capability positionally (the urun-sh/skypilot#4 lesson).
        """
        if cloud_capability == clouds.CloudCapability.COMPUTE:
            return cls._check_compute_credentials()
        return False, f'Verda does not support {cloud_capability.value}.'

    # -- provisioning ------------------------------------------------------

    def make_deploy_resources_variables(
        self,
        resources: 'resources_lib.Resources',
        cluster_name: resources_utils.ClusterName,
        region: clouds.Region,
        zones: Optional[List[clouds.Zone]],
        num_nodes: int,
        dryrun: bool = False,
        volume_mounts: Optional[List[Any]] = None,
    ) -> Dict[str, Optional[str]]:
        del cluster_name, dryrun, volume_mounts, num_nodes
        assert zones is None, 'Verda does not support zones.'
        resources = resources.assert_launchable()
        acc_dict = self.get_accelerators_from_instance_type(
            resources.instance_type)
        custom_resources = resources_utils.make_ray_custom_resources_str(
            acc_dict)
        # InstanceType is the Verda code (e.g. 1H100.80S.22V); Region is
        # the location NAME — the provisioner maps it to a location code
        # via GET /v1/locations at launch.
        return {
            'instance_type': resources.instance_type,
            'custom_resources': custom_resources,
            'region': region.name,
            'use_spot': resources.use_spot,
        }

    def _get_feasible_launchable_resources(
        self, resources: 'resources_lib.Resources'
    ) -> resources_utils.FeasibleResources:
        if resources.instance_type is not None:
            assert resources.is_launchable(), resources
            resources = resources.copy(accelerators=None)
            return resources_utils.FeasibleResources([resources], [], None)

        def _make(instance_list):
            resource_list = []
            for instance_type in instance_list:
                r = resources.copy(
                    cloud=Verda(),
                    instance_type=instance_type,
                    accelerators=None,
                    cpus=None,
                    memory=None,
                )
                resource_list.append(r)
            return resource_list

        accelerators = resources.accelerators
        if accelerators is None:
            default_instance_type = Verda.get_default_instance_type(
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
        (instance_list, fuzzy_candidate_list
        ) = verda_catalog.get_instance_type_for_accelerator(
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
        # STATUS_VERSION is SKYPILOT, so the provisioner's query_instances
        # is the authority and this path is not used.
        raise NotImplementedError(
            'Verda uses StatusVersion.SKYPILOT; status comes from '
            'sky.provision.verda.query_instances.')

    @classmethod
    def get_image_size(cls, image_id: str, region: Optional[str]) -> float:
        del image_id, region
        # The OS image is provider-side and its size is not exposed; 0.0
        # lets every image through (the same policy as the lane twins).
        return 0.0
