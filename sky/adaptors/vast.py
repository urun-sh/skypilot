"""Vast cloud adaptor."""

import functools

_vast_sdk = None

# ISO 3166-1 alpha-2 → georegion token for codes vastai-sdk 0.2.5's
# `_regions_rev` omits. The SDK ships 182 of the 249 officially assigned
# codes; the live marketplace is not limited to that set — Ireland (IE)
# offers appeared 2026-09-15, Macau (MO) on 2026-09-16, each crashing the
# whole catalog fetch with KeyError inside `queryFormatter`. Rather than
# patching one country at a time, this table completes the map: after the
# patch, every officially assigned alpha-2 code resolves.
#
# Tokens are the SDK's own (AF/AS/EU/LC/NA/OC), assigned per the UN M49
# geoscheme; territories with no M49 continent (AQ, BV, GS, HM, TF, UM)
# fall to Oceania by convention. `setdefault` keeps each entry a no-op
# once a newer SDK covers the code natively.
_MISSING_GEOREGIONS = {
    'AF': 'AS',  # Afghanistan
    'AI': 'LC',  # Anguilla
    'AQ': 'OC',  # Antarctica
    'AS': 'OC',  # American Samoa
    'AW': 'LC',  # Aruba
    'AX': 'EU',  # Åland Islands
    'BL': 'LC',  # Saint Barthélemy
    'BM': 'NA',  # Bermuda
    'BQ': 'LC',  # Caribbean Netherlands
    'BV': 'OC',  # Bouvet Island
    'CC': 'AS',  # Cocos (Keeling) Islands
    'CI': 'AF',  # Côte d'Ivoire
    'CK': 'OC',  # Cook Islands
    'CW': 'LC',  # Curaçao
    'CX': 'AS',  # Christmas Island
    'DM': 'LC',  # Dominica
    'EH': 'AF',  # Western Sahara
    'FK': 'LC',  # Falkland Islands
    'FO': 'EU',  # Faroe Islands
    'GD': 'LC',  # Grenada
    'GF': 'LC',  # French Guiana
    'GG': 'EU',  # Guernsey
    'GI': 'EU',  # Gibraltar
    'GL': 'NA',  # Greenland
    'GP': 'LC',  # Guadeloupe
    'GS': 'OC',  # South Georgia and the Islands
    'GT': 'LC',  # Guatemala
    'HM': 'OC',  # Heard & McDonald Islands
    'IE': 'EU',  # Ireland
    'IM': 'EU',  # Isle of Man
    'IO': 'AS',  # British Indian Ocean Territory
    'JE': 'EU',  # Jersey
    'KG': 'AS',  # Kyrgyzstan
    'KN': 'LC',  # Saint Kitts and Nevis
    'KY': 'LC',  # Cayman Islands
    'LA': 'AS',  # Laos
    'LB': 'AS',  # Lebanon
    'LC': 'LC',  # Saint Lucia
    'MF': 'LC',  # Saint Martin (French)
    'MG': 'AF',  # Madagascar
    'MO': 'AS',  # Macau
    'MP': 'OC',  # Northern Mariana Islands
    'MQ': 'LC',  # Martinique
    'MS': 'LC',  # Montserrat
    'NC': 'OC',  # New Caledonia
    'NF': 'OC',  # Norfolk Island
    'NU': 'OC',  # Niue
    'PF': 'OC',  # French Polynesia
    'PM': 'NA',  # Saint Pierre and Miquelon
    'PN': 'OC',  # Pitcairn
    'PS': 'AS',  # Palestine
    'RE': 'AF',  # Réunion
    'SB': 'OC',  # Solomon Islands
    'SJ': 'EU',  # Svalbard & Jan Mayen
    'SM': 'EU',  # San Marino
    'SR': 'LC',  # Suriname
    'SX': 'LC',  # Sint Maarten (Dutch)
    'TC': 'LC',  # Turks and Caicos
    'TF': 'OC',  # French Southern Territories
    'TK': 'OC',  # Tokelau
    'TL': 'AS',  # Timor-Leste
    'UM': 'OC',  # U.S. Minor Outlying Islands
    'UY': 'LC',  # Uruguay
    'UZ': 'AS',  # Uzbekistan
    'VC': 'LC',  # Saint Vincent and the Grenadines
    'VE': 'LC',  # Venezuela
    'VG': 'LC',  # British Virgin Islands
    'VI': 'LC',  # U.S. Virgin Islands
    'WF': 'OC',  # Wallis and Futuna
    'WS': 'OC',  # Samoa
    'YT': 'AF',  # Mayotte
}


def _patch_sdk_regions():
    """Complete vastai-sdk 0.2.5's incomplete ISO 3166-1 reverse map.

    Measured live: the SDK's `queryFormatter` does `_regions_rev[country]`
    for every offer's geolocation and raises KeyError on any code outside
    its 182-entry map, crashing the entire catalog fetch (`fetch_vast.py`
    → 0 GPUs). IE appeared in the marketplace 2026-09-15; MO on
    2026-09-16 — the gap set drifts with the marketplace, so the patch
    completes the map for every officially assigned code instead of
    chasing individual countries.

    `_regions_rev` maps ISO alpha-2 → GEOREGION TOKEN (AF/AS/EU/LC/NA/OC),
    NOT country name — verified: US→NA, DE→EU, JP→AS, AU→OC.

    VERSION-GUARDED, NOT SILENT: this touches a private global in a
    deprecated SDK (renamed to `vastai`). If the SDK is upgraded and no
    longer has `_regions_rev`, the catalog would silently regress to the
    crash — so we RAISE, not no-op. When a newer SDK covers a code
    natively, that code's `setdefault` is a harmless no-op and can be
    dropped from the table.
    """
    from vastai import vastai_sdk
    # Fail loudly if the SDK internals we patch have moved (version drift)
    if not hasattr(vastai_sdk, '_regions_rev'):
        raise RuntimeError(
            'vastai-sdk internal _regions_rev not found — the SDK has '
            'changed shape. The georegion completion shim needs review. '
            'Expected vastai-sdk 0.2.x (pinned 0.2.5 in skypilot-controller).'
        )
    for code, token in _MISSING_GEOREGIONS.items():
        vastai_sdk._regions_rev.setdefault(code, token)


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
