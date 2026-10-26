// @jest-environment jsdom
import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ProvidersMap, REGION_COORDS } from '@/components/providersMap';

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
        regions: ['90e8fe74:newyork-usa-1', '90e8fe74:us-1', '90e8fe74:eu-1'],
        region_details: [
          { region: '90e8fe74:newyork-usa-1', units: 27 },
          { region: '90e8fe74:us-1', units: 18 },
          { region: '90e8fe74:eu-1', units: 9 },
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
  it('plots one marker per mapped region across providers', () => {
    const { container } = render(<ProvidersMap providers={fixture} />);
    // EU + newyork-usa-1 + us-1 (hash prefix stripped) + lambda's
    // us-east-1 — four mapped regions; spheron's `eu-1` bucket has no
    // coordinates and renders in the unmapped row instead.
    expect(container.querySelectorAll('circle')).toHaveLength(4);
  });

  it('lists unmapped region names below the map instead of dropping them', () => {
    expect(REGION_COORDS['eu-1']).toBeUndefined();
    render(<ProvidersMap providers={fixture} />);
    expect(
      screen.getByText(/Unmapped regions/, { exact: false })
    ).toBeInTheDocument();
    expect(screen.getByText('eu-1')).toBeInTheDocument();
  });

  it('shows the hovered region with per-provider availability', () => {
    const { container } = render(<ProvidersMap providers={fixture} />);
    const markers = container.querySelectorAll('circle');
    // Hover the marker whose region carries vast's live units (EU).
    fireEvent.mouseEnter(markers[0]);
    expect(screen.getByText('eu')).toBeInTheDocument();
    expect(screen.getByText('vast: 2 units')).toBeInTheDocument();
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
