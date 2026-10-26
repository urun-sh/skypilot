// Map view for the Providers section: each provider's regional availability
// plotted on a world map, sized by live stock. The Table view stays the
// source of truth for numbers; this is the same payload, drawn spatially.
//
// - World outlines come from the vendored `world-atlas` topojson (Natural
//   Earth public domain), imported so the bundle carries it — prod CSP is
//   `default-src 'self'`, so no runtime geography fetch would work.
// - Region → coordinates is a curated table keyed by provider when a name
//   is provider-specific (Spheron buckets like `EU West 1`, Vast continent
//   buckets): a bare name table would collide with cloud-region ids
//   (AWS `eu-west-1` is Ireland, Spheron's is Belgium — CodeRabbit round 1).
// - Unknown names are NOT dropped: they render in the "unmapped" row below
//   the map, and the mapped-region list is real DOM (keyboard reachable),
//   so the availability data never hides behind a hover-only affordance.

import React, { useMemo, useState } from 'react';
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  ZoomableGroup,
} from 'react-simple-maps';
import worldData from 'world-atlas/countries-110m.json';
import { regionDisplayName } from '@/utils/infraUtils';

// Region coordinates by display name (hash prefix stripped, lowercased).
// Only names that mean the same place on every provider (cloud-region ids
// and city names) live here.
export const REGION_COORDS = {
  // AWS region ids
  'af-south-1': [-33.9, 18.4],
  'ap-east-1': [22.3, 114.2],
  'ap-northeast-1': [35.7, 139.7],
  'ap-northeast-2': [37.6, 127.0],
  'ap-northeast-3': [34.7, 135.5],
  'ap-south-1': [19.1, 72.9],
  'ap-south-2': [17.4, 78.5],
  'ap-southeast-1': [1.35, 103.8],
  'ap-southeast-2': [-33.9, 151.2],
  'ap-southeast-3': [-6.2, 106.8],
  'ap-southeast-4': [-36.8, 174.8],
  'ca-central-1': [45.5, -73.6],
  'ca-west-1': [49.3, -123.1],
  'eu-central-1': [50.1, 8.7],
  'eu-central-2': [47.4, 8.5],
  'eu-north-1': [59.3, 18.1],
  'eu-south-1': [45.5, 9.2],
  'eu-south-2': [40.4, -3.7],
  'eu-west-1': [53.3, -6.2],
  'eu-west-2': [51.5, -0.1],
  'eu-west-3': [48.9, 2.4],
  'me-central-1': [25.2, 55.3],
  'me-south-1': [26.4, 50.1],
  'us-east-1': [38.9, -77.0],
  'us-east-2': [39.9, -83.0],
  'us-west-1': [37.4, -121.9],
  'us-west-2': [45.6, -122.6],
  // GCP region ids (no dash before the region number)
  'asia-east1': [24.0, 121.0],
  'asia-east2': [22.3, 114.2],
  'asia-northeast1': [35.7, 139.7],
  'asia-northeast3': [37.6, 127.0],
  'asia-south1': [19.1, 72.9],
  'asia-south2': [17.4, 78.5],
  'asia-southeast1': [1.35, 103.8],
  'asia-southeast2': [-6.2, 106.8],
  'australia-southeast1': [-33.9, 151.2],
  'australia-southeast2': [-37.8, 145.0],
  'europe-central2': [52.2, 21.0],
  'europe-north1': [60.2, 24.9],
  'europe-west1': [50.9, 4.6],
  'europe-west10': [52.5, 13.4],
  'europe-west2': [51.5, -0.1],
  'europe-west3': [50.1, 8.7],
  'europe-west4': [52.1, 5.3],
  'europe-west6': [47.4, 8.5],
  'europe-west8': [45.5, 9.2],
  'europe-west9': [48.9, 2.4],
  'me-central2': [26.3, 50.1],
  'me-west1': [32.1, 34.8],
  'northamerica-northeast1': [45.5, -73.6],
  'northamerica-northeast2': [43.7, -79.4],
  'southamerica-east1': [-23.5, -46.6],
  'southamerica-west1': [-33.4, -70.6],
  'us-central1': [41.3, -95.9],
  'us-central2': [35.5, -97.5],
  'us-east1': [33.2, -80.0],
  'us-east4': [38.9, -77.0],
  'us-east5': [39.9, -83.0],
  'us-south1': [32.8, -96.8],
  'us-west1': [45.6, -121.2],
  'us-west2': [34.0, -118.2],
  'us-west3': [40.8, -111.9],
  'us-west4': [36.2, -115.1],
  // Lambda / QuantaCloud style ids
  'us-south-2': [29.4, -98.5],
  'us-southeast-1': [33.7, -84.4],
  'us-midwest-1': [41.9, -87.6],
  'us-midwest-2': [39.8, -89.6],
  'us-midwest-3': [44.9, -93.3],
  // City names (Shadeform / Spheron — same city, same place)
  'atlanta-usa-3': [33.7, -84.4],
  'beltsville-usa-1': [39.0, -76.9],
  'calgary-canada-1': [51.0, -114.1],
  'casper-usa-2': [42.9, -106.3],
  'chennai-india': [13.1, 80.3],
  'culpeper-usa-1': [38.5, -77.9],
  'dallas-usa-3': [32.8, -96.8],
  'desmoines-usa-1': [41.6, -93.6],
  'dulles-usa-1': [39.0, -77.4],
  'frankfurt-germany-6': [50.1, 8.7],
  'helsinki-finland-2': [60.2, 24.9],
  'helsinki-finland-3': [60.2, 24.9],
  'helsinki-finland-5': [60.2, 24.9],
  'houston-usa-1': [29.8, -95.4],
  'kansascity-usa-1': [39.1, -94.6],
  'london-uk-1': [51.5, -0.1],
  'london-uk-2': [51.5, -0.1],
  'losangeles-usa-1': [34.1, -118.2],
  'mumbai-india-1': [19.1, 72.9],
  'newark-usa-1': [40.7, -74.2],
  'newyork-usa-1': [40.7, -74.0],
  'noida-india': [28.6, 77.4],
  'omaha-usa-2': [41.3, -95.9],
  'paris-france-1': [48.9, 2.4],
  'paris-france-5': [48.9, 2.4],
  'phoenix-usa-2': [33.4, -112.1],
  'saltlakecity-usa-1': [40.8, -111.9],
  'sanjose-usa-2': [37.3, -121.9],
  'singapore-singapore-1': [1.35, 103.8],
  'sydney-australia-1': [-33.9, 151.2],
  'tokyo-japan-5': [35.7, 139.7],
  'warsaw-poland-1': [52.2, 21.0],
  'wichita-usa-1': [37.7, -97.3],
  'woodbury-usa-1': [44.9, -92.9],
};

// Provider-scoped names: Spheron's freeform buckets (`EU West 1`,
// `us-1`, …) and Vast's continent buckets (`US`, `EU`, `NA`). These keys
// collide with other providers' region ids (`eu-west-1` is AWS Ireland
// but Spheron Belgium), so they resolve per provider first.
export const PROVIDER_REGION_COORDS = {
  spheron: {
    'canada-1': [51.0, -114.1],
    'eu-north-1': [60.2, 24.9],
    'eu-south-1': [41.1, 16.9],
    'eu-west-1': [50.9, 4.6],
    'eu-west-2': [48.9, 2.4],
    'finland-1': [60.2, 24.9],
    'finland-3': [61.5, 23.8],
    'me-west-1': [32.1, 34.8],
    'uk-south-1': [51.5, -0.1],
    'uk-south-2': [53.4, -1.1],
    'us-central-1': [39.1, -94.6],
    'us-north-1': [47.6, -122.3],
    'us-1': [38.9, -77.0],
    'eu-1': [41.1, 16.9],
    // Space-separated display names
    'eu north 1': [60.2, 24.9],
    'eu south 1': [41.1, 16.9],
    'eu west 1': [50.9, 4.6],
    'eu west 2': [48.9, 2.4],
    'finland 1': [60.2, 24.9],
    'finland 3': [61.5, 23.8],
    'me west 1': [32.1, 34.8],
    'uk south 1': [51.5, -0.1],
    'uk south 2': [53.4, -1.1],
    'us central 1': [39.1, -94.6],
    'us north 1': [47.6, -122.3],
  },
  vast: {
    us: [39.8, -98.6],
    eu: [50.5, 15.0],
    na: [39.8, -98.6],
  },
};

// Coordinates for a provider's region display name: provider-scoped names
// win over the generic table.
const coordsFor = (cloud, name) =>
  (PROVIDER_REGION_COORDS[cloud] || {})[name] ?? REGION_COORDS[name];

// Marker fill by whether the region reports live units.
const MARKER_STYLES = {
  units: '#34d399', // emerald-400
  catalog: '#94a3b8', // slate-400
};

export function ProvidersMap({ providers = {} }) {
  const [hovered, setHovered] = React.useState(null);

  // Aggregate the offerings payload per provider + region: which providers
  // are present and each one's total live units (units tier) or catalog
  // presence. Raw region ids normalize (hash-prefix strip) to display
  // names; per-region units SUM across the raw ids that normalize to the
  // same name — a `find()` per raw id would return only the first detail
  // and double-count it (CodeRabbit round 1).
  const { mapped, unmapped } = useMemo(() => {
    // `${cloud}/${name}` -> { cloud, name, coords, units, hasUnits, mode }
    const byRegion = new Map();
    const unmappedRegions = new Set();
    Object.entries(providers).forEach(([cloud, entry]) => {
      if (entry?.present !== true) {
        return;
      }
      const mode = entry.stock_mode;
      (entry.gpus || []).forEach((gpu) => {
        // Sum units per normalized region name for this GPU first.
        const unitsByName = new Map();
        (gpu.region_details || []).forEach((d) => {
          const name = regionDisplayName(d.region).toLowerCase();
          if (typeof d.units === 'number') {
            unitsByName.set(name, (unitsByName.get(name) || 0) + d.units);
          }
        });
        // Every raw id (and any name `regions` lists without details)
        // contributes presence; units come only from the pre-summed map.
        const names = new Set([
          ...(gpu.regions || []).map((r) => regionDisplayName(r).toLowerCase()),
          ...unitsByName.keys(),
        ]);
        names.forEach((name) => {
          const coords = coordsFor(cloud, name);
          const key = `${cloud}/${name}`;
          if (coords === undefined) {
            unmappedRegions.add(name);
            return;
          }
          if (!byRegion.has(key)) {
            byRegion.set(key, {
              cloud,
              name,
              coords,
              units: 0,
              hasUnits: false,
              mode,
            });
          }
          const agg = byRegion.get(key);
          const units = unitsByName.get(name);
          if (typeof units === 'number') {
            agg.units += units;
            agg.hasUnits = true;
          }
        });
      });
    });
    const mapped = [...byRegion.values()];
    const unmapped = [...unmappedRegions].sort();
    return { mapped, unmapped };
  }, [providers]);

  return (
    <div className="rounded-lg border bg-card text-card-foreground shadow-sm">
      <div className="relative p-3">
        <ComposableMap
          projection="geoNaturalEarth1"
          className="w-full h-auto"
          style={{ background: 'transparent' }}
        >
          <ZoomableGroup zoom={1}>
            <Geographies geography={worldData}>
              {({ geographies }) =>
                geographies.map((geo) => (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    style={{
                      default: { fill: '#e5e7eb', stroke: '#f3f4f6' },
                      hover: { fill: '#d1d5db' },
                      pressed: { fill: '#d1d5db' },
                    }}
                  />
                ))
              }
            </Geographies>
            {mapped.map((region) => (
              <Marker
                key={`${region.cloud}/${region.name}`}
                coordinates={[region.coords[1], region.coords[0]]}
                onMouseEnter={() => setHovered(region)}
                onMouseLeave={() =>
                  setHovered((h) => (h === region ? null : h))
                }
                style={{ cursor: 'pointer' }}
              >
                <circle
                  r={
                    region.hasUnits
                      ? Math.min(10, 4 + Math.sqrt(region.units) * 1.2)
                      : 3
                  }
                  fill={
                    region.hasUnits
                      ? MARKER_STYLES.units
                      : MARKER_STYLES.catalog
                  }
                  stroke="#fff"
                  strokeWidth={1}
                  opacity={0.85}
                />
              </Marker>
            ))}
          </ZoomableGroup>
        </ComposableMap>
        {hovered && (
          <div
            className="absolute z-10 pointer-events-none px-2 py-1 text-xs text-gray-100 bg-gray-500 rounded whitespace-pre-line normal-case"
            style={{
              left: '50%',
              top: '0.5rem',
              transform: 'translateX(-50%)',
            }}
          >
            <div className="font-medium mb-0.5">
              {regionDisplayName(hovered.name)} · {hovered.cloud}
            </div>
            <div>
              {hovered.hasUnits
                ? `${hovered.units} units`
                : hovered.mode === 'in_stock'
                  ? 'in stock'
                  : 'catalog'}
            </div>
          </div>
        )}
        {/* Keyboard-reachable region availability list (CodeRabbit round
            1): the data must not live behind a hover-only marker. Each
            entry focuses/taps into the same hover detail box. */}
        {mapped.length > 0 && (
          <div className="mt-3">
            <p className="text-xs text-gray-400 mb-1.5">
              Regions by provider ({mapped.length})
            </p>
            <div className="flex flex-wrap gap-1.5">
              {mapped.map((region) => (
                <button
                  key={`${region.cloud}/${region.name}`}
                  type="button"
                  onMouseEnter={() => setHovered(region)}
                  onFocus={() => setHovered(region)}
                  onMouseLeave={() =>
                    setHovered((h) => (h === region ? null : h))
                  }
                  onBlur={() => setHovered((h) => (h === region ? null : h))}
                  className="px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs font-medium hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-300"
                >
                  {regionDisplayName(region.name)} · {region.cloud}
                  {region.hasUnits ? `: ${region.units} units` : ''}
                </button>
              ))}
            </div>
          </div>
        )}
        {unmapped.length > 0 && (
          <p className="mt-2 text-xs text-gray-400">
            Unmapped regions (no coordinates):
            {unmapped.map((name) => (
              <span
                key={name}
                className="ml-1 px-1.5 py-0.5 bg-gray-100 text-gray-500 rounded text-xs font-medium"
              >
                {name}
              </span>
            ))}
          </p>
        )}
      </div>
    </div>
  );
}
