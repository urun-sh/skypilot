import pytest

from sky.clouds.cloud import Cloud


@pytest.mark.parametrize(("specific_reservations", "expected"), [({"a"}, {
    "a": 0
}), ((set(), {}))])
def test_cloud_get_reservations_available_resources(specific_reservations,
                                                    expected):

    available_resources = Cloud().get_reservations_available_resources(
        "instance_type", "region", "zone", specific_reservations)
    assert available_resources == expected


def test_every_cloud_sets_its_own_repr():
    """`Cloud._REPR` is the cloud's catalog name, not a display string.

    `Cloud.validate_region_zone` and `Cloud.is_image_tag_valid` both pass
    `cls._REPR.lower()` to the catalog as the `clouds=` argument. A subclass
    that does not set `_REPR` therefore inherits the base placeholder
    `'<Cloud>'`, which lowercases to `'<cloud>'` and fails inside the catalog
    loader:

        ValueError: Cannot find module "sky.catalog.<cloud>_catalog"
                    for cloud "<cloud>"

    Shadeform shipped in exactly that state. It defines its own `__repr__`, so
    `repr()` and `str()` both read "Shadeform" everywhere a human looks --
    which is why the omission survived review, and why asserting on `repr()`
    would not have caught it. This asserts on `_REPR` itself.
    """
    from sky.utils import registry

    offenders = []
    for name, cloud_cls in registry.CLOUD_REGISTRY.items():
        if not isinstance(cloud_cls, type):
            cloud_cls = type(cloud_cls)
        if cloud_cls.__name__ == 'DummyCloud':
            continue
        if cloud_cls._REPR == Cloud._REPR:
            offenders.append(f'{name} ({cloud_cls.__name__})')
    assert not offenders, (
        'these clouds inherit the placeholder Cloud._REPR, so '
        'validate_region_zone and is_image_tag_valid will look up '
        f'sky.catalog.<cloud>_catalog for them: {offenders}')
