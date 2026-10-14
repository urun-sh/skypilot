import { act, renderHook, waitFor } from '@testing-library/react';

// Mock the shared dashboard cache so we can observe which cache key the hook
// fetches under (all-users vs. current-user scope) without hitting the network.
jest.mock('@/lib/cache', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    invalidate: jest.fn(),
    invalidateFunction: jest.fn(),
    setPreloader: jest.fn(),
    getCached: jest.fn(),
    clear: jest.fn(),
  },
}));

// Stub the API client so getClusters can be exercised in isolation.
jest.mock('@/data/connectors/client', () => ({
  __esModule: true,
  apiClient: { fetch: jest.fn() },
}));

// Plugin enhancements are a no-op passthrough for these tests.
jest.mock('@/plugins/dataEnhancement', () => ({
  __esModule: true,
  applyEnhancements: jest.fn(async (data) => data),
}));

import dashboardCache from '@/lib/cache';
import { apiClient } from '@/data/connectors/client';
import {
  getClusters,
  getClusterHistory,
  getOtherUsersClustersCount,
  useClusterData,
} from '@/data/connectors/clusters';

describe('getClusters all_users scoping', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.fetch.mockResolvedValue([]);
  });

  it('requests all users by default', async () => {
    await getClusters();

    expect(apiClient.fetch).toHaveBeenCalledTimes(1);
    const [, body] = apiClient.fetch.mock.calls[0];
    expect(body.all_users).toBe(true);
  });

  it('scopes the request to the current user when allUsers is false', async () => {
    await getClusters({ allUsers: false });

    expect(apiClient.fetch).toHaveBeenCalledTimes(1);
    const [, body] = apiClient.fetch.mock.calls[0];
    expect(body.all_users).toBe(false);
  });
});

// Raw /status payload rows. Only the fields the mapping reads are needed;
// everything else falls back inside getClusters.
const rawCluster = (overrides = {}) => ({
  name: 'fixture-cluster',
  user_name: 'alice',
  user_hash: 'hash-alice',
  cluster_hash: 'hash-fixture',
  status: 'UP',
  launched_at: 1700000000,
  nodes: 1,
  cpus: 4,
  memory: 16,
  accelerators: null,
  resources_str: '1x[CPU:4]',
  ...overrides,
});

describe('getClusters infra mapping (exact provider repr)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('passes the exact cloud repr through with the region suffix', async () => {
    apiClient.fetch.mockResolvedValue([
      rawCluster({ cloud: 'GCP', region: 'us-west3' }),
    ]);
    const [cluster] = await getClusters();
    expect(cluster.cloud).toBe('GCP');
    expect(cluster.infra).toBe('GCP (us-west3)');
    expect(cluster.full_infra).toBe('GCP (us-west3)');
  });

  it('prefers the zone over the region when both are present', async () => {
    apiClient.fetch.mockResolvedValue([
      rawCluster({ cloud: 'GCP', region: 'us-west1', zone: 'us-west1-a' }),
    ]);
    const [cluster] = await getClusters();
    expect(cluster.infra).toBe('GCP (us-west1-a)');
    expect(cluster.full_infra).toBe('GCP (us-west1-a)');
  });

  it.each([
    ['Vast', 'us-east'],
    ['Quantacloud', 'cn-north-1'],
    ['Ornn', 'ornn-east'],
    ['Spheron', 'spheron-fra'],
    ['Latitude', 'latitude-1'],
    ['PrimeIntellect', 'us-east'],
    ['Lambda', 'gcp'],
  ])('renders the exact %s repr with its region', async (cloud, region) => {
    apiClient.fetch.mockResolvedValue([rawCluster({ cloud, region })]);
    const [cluster] = await getClusters();
    expect(cluster.cloud).toBe(cloud);
    expect(cluster.infra).toBe(cloud + ' (' + region + ')');
    expect(cluster.full_infra).toBe(cloud + ' (' + region + ')');
  });

  it('strips the ssh- prefix for SSH node pools', async () => {
    apiClient.fetch.mockResolvedValue([
      rawCluster({ cloud: 'SSH', region: 'ssh-pool-42' }),
    ]);
    const [cluster] = await getClusters();
    expect(cluster.infra).toBe('SSH (pool-42)');
    expect(cluster.full_infra).toBe('SSH (pool-42)');
    expect(cluster.region).toBe('ssh-pool-42');
  });

  it('renders the cloud alone when region and zone are missing', async () => {
    apiClient.fetch.mockResolvedValue([rawCluster({ cloud: 'Fluidstack' })]);
    const [cluster] = await getClusters();
    expect(cluster.infra).toBe('Fluidstack');
    expect(cluster.full_infra).toBe('Fluidstack');
  });

  it('truncates a long region in infra but keeps full_infra untruncated', async () => {
    const longRegion = 'a'.repeat(40);
    apiClient.fetch.mockResolvedValue([
      rawCluster({ cloud: 'GCP', region: longRegion }),
    ]);
    const [cluster] = await getClusters();
    const truncated = 'a'.repeat(11) + '...' + 'a'.repeat(11);
    expect(cluster.infra).toBe('GCP (' + truncated + ')');
    expect(cluster.full_infra).toBe('GCP (' + longRegion + ')');
  });
});

describe('getOtherUsersClustersCount', () => {
  const currentUser = { id: 'u-1', name: 'alice' };

  beforeEach(() => {
    jest.clearAllMocks();
    delete window.__skyPaginationFetch;
  });

  it('probes the pagination extension with a single-row page when available', async () => {
    const pluginFetch = jest.fn();
    window.__skyPaginationFetch = pluginFetch;
    dashboardCache.get.mockResolvedValue({ total: 1234, items: [{}] });

    const count = await getOtherUsersClustersCount(currentUser);

    expect(count).toBe(1234);
    // The probe must go through the plugin fetch with limit 1 rather than
    // fetching every row just to count them.
    expect(dashboardCache.get).toHaveBeenCalledWith(pluginFetch, [
      { page: 1, limit: 1, allUsers: true },
    ]);
    expect(dashboardCache.get).not.toHaveBeenCalledWith(getClusters);
  });

  it("falls back to counting other users' rows from the shared cache", async () => {
    dashboardCache.get.mockResolvedValue([
      { cluster: 'mine', user_hash: 'u-1', user: 'alice' },
      { cluster: 'other-1', user_hash: 'u-2', user: 'bob' },
      { cluster: 'other-2', user_hash: 'u-3', user: 'carol' },
    ]);

    const count = await getOtherUsersClustersCount(currentUser);

    expect(count).toBe(2);
    expect(dashboardCache.get).toHaveBeenCalledWith(getClusters);
  });
});

describe('useClusterData ownership scoping (client path)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // No pagination plugin -> client-side path.
    delete window.__skyPaginationFetch;
    dashboardCache.get.mockResolvedValue([]);
  });

  it('fetches the shared all-users cache entry when allUsers is true', async () => {
    renderHook(() => useClusterData({ allUsers: true }));

    await waitFor(() => expect(dashboardCache.get).toHaveBeenCalled());
    // All-users keeps the default (no-arg) cache key so it stays shared with
    // the rest of the dashboard.
    expect(dashboardCache.get).toHaveBeenCalledWith(getClusters);
    expect(dashboardCache.get).not.toHaveBeenCalledWith(getClusters, [
      { allUsers: false },
    ]);
  });

  it('fetches the current-user-scoped cache entry when allUsers is false', async () => {
    renderHook(() =>
      useClusterData({
        allUsers: false,
        currentUser: { id: 'u-1', name: 'alice' },
      })
    );

    await waitFor(() =>
      expect(dashboardCache.get).toHaveBeenCalledWith(getClusters, [
        { allUsers: false },
      ])
    );
    expect(dashboardCache.get).not.toHaveBeenCalledWith(getClusters);
  });

  it("drops other users' terminated clusters from history when scoped to mine", async () => {
    // The cost_report (history) endpoint always returns every user's rows, so
    // the hook scopes them client-side to match the server-scoped active list.
    dashboardCache.get.mockImplementation((fn) => {
      if (fn === getClusterHistory) {
        return Promise.resolve([
          {
            cluster: 'mine-terminated',
            user_hash: 'u-1',
            user: 'alice',
            cluster_hash: 'h2',
            status: 'TERMINATED',
          },
          {
            cluster: 'other-terminated',
            user_hash: 'u-2',
            user: 'bob',
            cluster_hash: 'h3',
            status: 'TERMINATED',
          },
        ]);
      }
      return Promise.resolve([
        {
          cluster: 'mine-active',
          user_hash: 'u-1',
          user: 'alice',
          cluster_hash: 'h1',
          status: 'RUNNING',
        },
      ]);
    });

    const { result } = renderHook(() =>
      useClusterData({
        showHistory: true,
        allUsers: false,
        currentUser: { id: 'u-1', name: 'alice' },
      })
    );

    await waitFor(() => expect(result.current.allData.length).toBe(2));
    const names = result.current.allData.map((c) => c.cluster).sort();
    expect(names).toEqual(['mine-active', 'mine-terminated']);
    expect(names).not.toContain('other-terminated');
  });

  it('drops a stale all-users response that resolves after a newer scoped fetch', async () => {
    // Deep-linking ?owner=mine starts an all-users fetch (initial scope)
    // followed by a scoped fetch once the URL is synced. If the all-users
    // response lands last it must not overwrite the scoped data.
    let resolveAllUsers;
    dashboardCache.get.mockImplementation((fn, args) => {
      if (args && args[0] && args[0].allUsers === false) {
        return Promise.resolve([
          {
            cluster: 'mine-active',
            user_hash: 'u-1',
            user: 'alice',
            status: 'RUNNING',
          },
        ]);
      }
      return new Promise((resolve) => {
        resolveAllUsers = () =>
          resolve([
            {
              cluster: 'other-active',
              user_hash: 'u-2',
              user: 'bob',
              status: 'RUNNING',
            },
          ]);
      });
    });

    // Stable identities: fresh objects every render would recreate the hook's
    // fetch callbacks and trigger spurious refetches.
    const currentUser = { id: 'u-1', name: 'alice' };
    const filters = [];
    const { result, rerender } = renderHook(
      ({ allUsers }) => useClusterData({ allUsers, currentUser, filters }),
      { initialProps: { allUsers: true } }
    );

    // Flip to the scoped fetch while the all-users request is still pending.
    rerender({ allUsers: false });
    await waitFor(() =>
      expect(result.current.allData.map((c) => c.cluster)).toEqual([
        'mine-active',
      ])
    );

    // The stale all-users response resolves last; it must be discarded.
    await act(async () => {
      resolveAllUsers();
    });
    expect(result.current.allData.map((c) => c.cluster)).toEqual([
      'mine-active',
    ]);
    expect(result.current.loading).toBe(false);
  });
});
