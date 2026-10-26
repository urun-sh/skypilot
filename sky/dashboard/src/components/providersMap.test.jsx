// @jest-environment jsdom
import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import {
  ProvidersMap,
  PROVIDER_REGION_COORDS,
  REGION_COORDS,
} from '@/components/providersMap';

const fixture = {
  vast: {
    present: true,
    stock_mode: 'units',
    gpus: [
      {
        name: 'H100',
        regions: ['EU'],
        region_details: [{ region: 'EU', units: 2 }],
      },
    ],
  },
  spheron: {
    present: true,
    stock_mode: 'units',
    gpus: [
      {
        name: 'H200',
        // Two raw ids normalize to the same display name — their units
        // must sum, not double-count the first detail (CodeRabbit round 1).
        regions: [
          '90e8fe74:newyork-usa-1',
          'ce32df2f:newyork-usa-1',
          '90e8fe74:eu-1',
          '90e8fe74:mon1',
        ],
        region_details: [
          { region: '90e8fe74:newyork-usa-1', units: 27 },
          { region: 'ce32df2f:newyork-usa-1', units: 5 },
          { region: '90e8fe74:eu-1', units: 9 },
          { region: '90e8fe74:mon1', units: 4 },
        ],
      },
    ],
  },
  lambda: {
    present: true,
    stock_mode: 'in_stock',
    gpus: [
      {
        name: 'H100',
        regions: ['us-east-1'],
        region_details: [{ region: 'us-east-1' }],
      },
    ],
  },
  ornn: { present: false, stock_mode: null, gpus: [] },
};

describe('ProvidersMap', () => {
  it('plots one marker per mapped provider region', () => {
    const { container } = render(<ProvidersMap providers={fixture} />);
    // vast EU + spheron newyork-usa-1 (raw x2, one marker) + spheron
    // eu-1 (provider-scoped) + lambda us-east-1. `mon1` has no
    // coordinates and renders in the unmapped row instead.
    expect(container.querySelectorAll('circle')).toHaveLength(4);
  });

  it('resolves provider-scoped names before the generic table', () => {
    // Spheron's eu-1 bucket is Belgium; the generic eu-west-1 (AWS
    // Ireland) must not have been overwritten by it.
    expect(PROVIDER_REGION_COORDS.spheron['eu-1']).toBeDefined();
    expect(REGION_COORDS['eu-west-1']).toEqual([53.3, -6.2]);
  });

  it('sums units when raw ids normalize to the same name', () => {
    render(<ProvidersMap providers={fixture} />);
    // 27 + 5 from the two `newyork-usa-1` raw ids — one list entry,
    // summed once (not 27 twice).
    expect(
      screen.getByText('newyork-usa-1 · spheron: 32 units')
    ).toBeInTheDocument();
  });

  it('lists unmapped region names below the map instead of dropping them', () => {
    render(<ProvidersMap providers={fixture} />);
    expect(
      screen.getByText(/Unmapped regions/, { exact: false })
    ).toBeInTheDocument();
    expect(screen.getByText('mon1')).toBeInTheDocument();
  });

  it('exposes region availability as a keyboard-reachable list', () => {
    render(<ProvidersMap providers={fixture} />);
    // The list is real DOM (CodeRabbit round 1): availability is not
    // locked behind a hover-only map marker.
    const eu = screen.getByRole('button', { name: /eu · vast/ });
    fireEvent.focus(eu);
    expect(screen.getByText('2 units')).toBeInTheDocument();
  });

  it('renders nothing for absent providers', () => {
    const { container } = render(
      <ProvidersMap providers={{ ornn: fixture.ornn }} />
    );
    expect(container.querySelectorAll('circle')).toHaveLength(0);
    expect(
      screen.queryByText(/Unmapped regions/, { exact: false })
    ).not.toBeInTheDocument();
  });
});
