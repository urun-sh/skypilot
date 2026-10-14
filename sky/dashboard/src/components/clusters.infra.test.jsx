// The Infra column must render the exact provider repr from the API as the
// /infra link text, and must never render a blank link when a record arrives
// without a cloud field: the provider is then parsed out of full_infra.
jest.mock('next/router', () => ({
  __esModule: true,
  useRouter: () => ({ isReady: true, query: {}, asPath: '/clusters' }),
}));
jest.mock('@/lib/cache', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    getCached: jest.fn(() => null),
    invalidate: jest.fn(),
    invalidateFunction: jest.fn(),
    setPreloader: jest.fn(),
  },
}));
jest.mock('@/plugins/PluginSlot', () => ({
  __esModule: true,
  PluginSlot: () => null,
}));
jest.mock('@/plugins/PluginProvider', () => ({
  __esModule: true,
  usePluginComponents: () => [],
  useTableColumns: () => [],
  usePluginRoute: () => null,
}));

import { render, screen, waitFor, within } from '@testing-library/react';
import dashboardCache from '@/lib/cache';
import { Clusters } from '@/components/clusters';

// Rows in the mapped shape the Clusters table consumes.
const row = (name, fields = {}) => ({
  cluster: name,
  name,
  status: 'UP',
  user: 'alice',
  user_hash: 'hash-alice',
  workspace: 'default',
  resources_str: '1x[CPU:2]',
  launched_at: 1700000000,
  autostop: -1,
  to_down: false,
  ...fields,
});

const renderClusters = async (clusters) => {
  window.history.replaceState({}, '', '/clusters');
  dashboardCache.get.mockImplementation(async (fn) => {
    const name = fn?.name || '';
    if (name.includes('Workspaces')) return {};
    return clusters;
  });
  render(<Clusters />);
  await waitFor(() => expect(dashboardCache.get).toHaveBeenCalled());
};

describe('Clusters Infra column (exact provider repr)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the exact cloud repr as the /infra link with its region', async () => {
    await renderClusters([
      row('vast-gpu', {
        cloud: 'Vast',
        infra: 'Vast (us-east)',
        full_infra: 'Vast (us-east)',
      }),
      row('quanta-gpu', {
        cloud: 'Quantacloud',
        infra: 'Quantacloud (cn-north-1)',
        full_infra: 'Quantacloud (cn-north-1)',
      }),
    ]);

    const vastLink = await screen.findByRole('link', { name: 'Vast' });
    expect(vastLink.getAttribute('href')).toBe('/infra');
    const vastRow = vastLink.closest('tr');
    expect(within(vastRow).getByText('(us-east)')).toBeTruthy();

    const quantaLink = screen.getByRole('link', { name: 'Quantacloud' });
    expect(quantaLink.getAttribute('href')).toBe('/infra');
    const quantaRow = quantaLink.closest('tr');
    expect(within(quantaRow).getByText('(cn-north-1)')).toBeTruthy();
  });

  it('falls back to the provider parsed from infra when cloud is missing', async () => {
    await renderClusters([
      row('stale-undefined', {
        infra: 'GCP (us-west3)',
        full_infra: 'GCP (us-west3)',
      }),
      row('stale-null', {
        cloud: null,
        infra: 'GCP (us-west3)',
        full_infra: 'GCP (us-west3)',
      }),
      row('stale-empty', {
        cloud: '',
        infra: 'GCP (us-west3)',
        full_infra: 'GCP (us-west3)',
      }),
    ]);

    const links = await screen.findAllByRole('link', { name: 'GCP' });
    expect(links).toHaveLength(3);
    const staleRow = links[0].closest('tr');
    expect(within(staleRow).getByText('(us-west3)')).toBeTruthy();
  });

  it('renders the cloud alone when infra has no region suffix', async () => {
    await renderClusters([
      row('fluid-only', {
        cloud: 'Fluidstack',
        infra: 'Fluidstack',
        full_infra: 'Fluidstack',
      }),
    ]);

    const fluidLink = await screen.findByRole('link', { name: 'Fluidstack' });
    expect(fluidLink.getAttribute('href')).toBe('/infra');
  });
});
