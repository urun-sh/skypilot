// The dashboard cache is a pass-through here so the connector's own logic is
// what's under test, not the caching layer.
jest.mock('@/lib/cache', () => ({
  __esModule: true,
  default: { get: jest.fn((fn, args = []) => fn(...args)) },
}));

jest.mock('@/data/connectors/client', () => ({
  __esModule: true,
  apiClient: { post: jest.fn(), get: jest.fn() },
}));
jest.mock('@/data/connectors/workspaces', () => ({
  __esModule: true,
  getWorkspaces: jest.fn(),
  getEnabledCloudsBatch: jest.fn(),
}));
jest.mock('@/data/connectors/clusters', () => ({
  __esModule: true,
  getClusters: jest.fn(),
}));
jest.mock('@/data/connectors/jobs', () => ({
  __esModule: true,
  getManagedJobs: jest.fn(),
}));
import { apiClient } from '@/data/connectors/client';
import {
  getCloudInfrastructure,
  getEnabledCloudsList,
  getSlurmInfrastructure,
} from '@/data/connectors/infra';
import { getClusters } from '@/data/connectors/clusters';
import { getManagedJobs } from '@/data/connectors/jobs';
import {
  getWorkspaces,
  getEnabledCloudsBatch,
} from '@/data/connectors/workspaces';

// Every Slurm endpoint answers the same way: POST returns a request id in a
// header, then /api/get returns the result under a JSON-encoded return_value.
const scheduled = (requestId) => ({
  ok: true,
  status: 200,
  headers: { get: () => requestId },
});

const result = (returnValue) => ({
  ok: true,
  status: 200,
  json: async () => ({ return_value: JSON.stringify(returnValue) }),
});

// The Infra page must list a cluster that is configured but currently
// unreachable, so the configured-cluster query has to be independent of the
// node and GPU queries rather than derived from them.
describe('getSlurmInfrastructure configured clusters', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.post.mockImplementation(async (path) => {
      if (path === '/slurm_cluster_names') return scheduled('req-clusters');
      if (path === '/slurm_gpu_availability') return scheduled('req-gpus');
      if (path === '/slurm_node_info') return scheduled('req-nodes');
      throw new Error(`unexpected POST to ${path}`);
    });
  });

  it('reports a configured cluster whose node and GPU queries are empty', async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path.includes('req-clusters')) return result(['offline-cluster']);
      return result([]);
    });

    const data = await getSlurmInfrastructure();

    expect(data.slurmClusterNames).toEqual(['offline-cluster']);
    expect(data.perNodeSlurmGPUs).toEqual([]);
    expect(data.perClusterSlurmGPUs).toEqual([]);
  });

  it('does not hold the node and GPU queries behind the cluster list', async () => {
    // Assertions stay in the test body: the connector catches everything a
    // mock throws, so an assertion made inside one would be swallowed and the
    // test would pass regardless.
    let releaseClusters;
    const clusterListPending = new Promise((resolve) => {
      releaseClusters = resolve;
    });
    apiClient.get.mockImplementation(async (path) => {
      if (path.includes('req-clusters')) {
        await clusterListPending;
        return result(['cluster-a']);
      }
      return result([]);
    });

    const pending = getSlurmInfrastructure();
    await new Promise((resolve) => setTimeout(resolve, 0));

    // The cluster list is still in flight, yet the other two have already got
    // past their own POST to fetching a result. Serializing them behind it
    // would leave these uncalled.
    const fetched = apiClient.get.mock.calls.map(([path]) => path);
    expect(fetched).toEqual(
      expect.arrayContaining([
        expect.stringContaining('req-nodes'),
        expect.stringContaining('req-gpus'),
      ])
    );

    releaseClusters();
    expect((await pending).slurmClusterNames).toEqual(['cluster-a']);
  });

  it('falls back to an empty list when the cluster query fails', async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path.includes('req-clusters')) return { ok: false, status: 500 };
      return result([]);
    });

    const data = await getSlurmInfrastructure();

    expect(data.slurmClusterNames).toEqual([]);
  });
});

// The Cloud panel must list every provider the org operates, not just the
// ones sky check reports as enabled. Rows come from the union of enabled
// clouds and clouds with live clusters or jobs, and count joins are
// case-insensitive because API `cloud` values are provider _REPR strings
// ('GCP', 'Vast', 'Quantacloud') while rows are named from CLOUDS_LIST.
describe('Cloud panel provider rows', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    getWorkspaces.mockResolvedValue({ team: {} });
    getEnabledCloudsBatch.mockResolvedValue({ team: ['aws'] });
    getClusters.mockResolvedValue([]);
    getManagedJobs.mockResolvedValue({ jobs: [] });
  });

  const gcpClusters = () =>
    getClusters.mockResolvedValue(
      Array.from({ length: 6 }, () => ({ cloud: 'GCP' }))
    );

  it('lists a provider absent from the old hardcoded list with its cluster count', async () => {
    getClusters.mockResolvedValue([
      { cloud: 'Quantacloud' },
      { cloud: 'Quantacloud' },
    ]);

    const data = await getEnabledCloudsList();

    expect(data.clouds.find((c) => c.name === 'Quantacloud')).toEqual({
      name: 'Quantacloud',
      enabled: false,
      clusters: 2,
      jobs: 0,
    });
  });

  it('joins counts case-insensitively so Vast clusters land on the Vast row', async () => {
    getClusters.mockResolvedValue([{ cloud: 'Vast' }]);
    getManagedJobs.mockResolvedValue({ jobs: [{ cloud: 'Vast' }] });

    const data = await getEnabledCloudsList();

    expect(data.clouds.find((c) => c.name === 'Vast')).toEqual({
      name: 'Vast',
      enabled: false,
      clusters: 1,
      jobs: 1,
    });
  });

  it('renders a cloud with clusters even when sky check does not enable it', async () => {
    // Production shape: six GCP clusters, only AWS enabled.
    gcpClusters();

    const data = await getEnabledCloudsList();

    expect(data.clouds.find((c) => c.name === 'GCP')).toEqual({
      name: 'GCP',
      enabled: false,
      clusters: 6,
      jobs: 0,
    });
  });

  it('still renders an enabled provider with no clusters or jobs', async () => {
    gcpClusters();

    const data = await getEnabledCloudsList();

    expect(data.clouds.find((c) => c.name === 'AWS')).toEqual({
      name: 'AWS',
      enabled: true,
      clusters: 0,
      jobs: 0,
    });
    expect(data.enabledClouds).toBe(1);
    expect(data.totalClouds).toBe(29);
  });

  it('aggregates the same rows in the legacy getCloudInfrastructure path', async () => {
    getClusters.mockResolvedValue([
      ...Array.from({ length: 6 }, () => ({ cloud: 'GCP' })),
      { cloud: 'Vast' },
    ]);
    getManagedJobs.mockResolvedValue({
      jobs: [{ cloud: 'Vast' }, { cloud: 'Quantacloud' }],
    });

    const data = await getCloudInfrastructure();

    const byName = Object.fromEntries(data.clouds.map((c) => [c.name, c]));
    expect(byName.GCP).toEqual({
      name: 'GCP',
      enabled: false,
      clusters: 6,
      jobs: 0,
    });
    expect(byName.Vast).toEqual({
      name: 'Vast',
      enabled: false,
      clusters: 1,
      jobs: 1,
    });
    expect(byName.Quantacloud.jobs).toBe(1);
    expect(byName.AWS).toEqual({
      name: 'AWS',
      enabled: true,
      clusters: 0,
      jobs: 0,
    });
  });
});
