"""Vast cloud adaptor."""

import functools

_vast_sdk = None






def _patch_sdk_regions():
    """Tolerate vastai-sdk 0.2.5's incomplete ISO 3166-1 reverse map.

    Measured live 2026-09-16: the SDK's `queryFormatter` does
    `_regions_rev[country]` and raises `KeyError('IE')` on Ireland-hosted
    offers, crashing the entire catalog fetch (`fetch_vast.py` → 0 GPUs).
    The SDK ships 182 codes; only `IE` is absent from the marketplace's
    observed set (verified against the live `/bundles` API, 64 offers).

    `_regions_rev` maps ISO alpha-2 → GEOREGION TOKEN (AF/AS/EU/LC/NA/OC),
    NOT country name — verified: US→NA, DE→EU, JP→AS, AU→OC. So IE→EU.

    VERSION-GUARDED, NOT SILENT: this touches a private global in a
    deprecated SDK (renamed to `vastai`). If the SDK is upgraded and no
    longer has `_regions_rev` or `queryFormatter`, the catalog would
    silently regress to the crash — so we RAISE, not no-op. When a newer
    SDK covers `IE` natively, the `setdefault` is a harmless no-op and
    this shim can be deleted.
    """
    from vastai import vastai_sdk
    # Fail loudly if the SDK internals we patch have moved (version drift)
    if not hasattr(vastai_sdk, '_regions_rev'):
        raise RuntimeError(
            'vastai-sdk internal _regions_rev not found — the SDK has '
            'changed shape. The IE georegion shim needs review. Expected '
            'vastai-sdk 0.2.x (pinned 0.2.5 in skypilot-controller).')
    vastai_sdk._regions_rev.setdefault('IE', 'EU')


def import_package(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _vast_sdk

        if _vast_sdk is None:
            try:
                import vastai_sdk as _vast  # pylint: disable=import-outside-toplevel
                _vast_sdk = _vast.VastAI()
                _patch_sdk_regions()
            except ImportError as e:
                raise ImportError(f'Fail to import dependencies for vast: {e}\n'
                                  'Try pip install "skypilot[vast]"') from None
        return func(*args, **kwargs)

    return wrapper


@import_package
def vast():
    """Return the vast package."""
    return _vast_sdk
