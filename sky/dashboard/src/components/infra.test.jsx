// Expose each plugin slot's name and row context as data attributes so the
// row-identity contract can be asserted. Renders no text, so the cell-content
// assertions below are unaffected.
jest.mock('@/plugins/PluginSlot', () => ({
  __esModule: true,
  PluginSlot: ({ name, context }) => (
    <span
      data-slot={name}
      data-row-id={context?.id}
      data-row-kind={context?.kind}
    />
  ),
}));

import { fireEvent, render, screen } from '@testing-library/react';
import {
  InfrastructureSection,
  CloudGpuCapacityCell,
  ProvidersSection,
  slurmRequestableCounts,
} from '@/components/infra';
import { PROVIDER_OFFERINGS_CLOUDS } from '@/data/connectors/constants';
import {
  countByCloud,
  gpuCapacityByCloud,
  recordsInWorkspace,
  visibleCloudRows,
} from '@/utils/infraUtils';

// The infrastructure table renders one sub-row per GPU type. A Slurm cluster
// can additionally be expanded into its partitions, which is where the
// overlap lives: a node in several partitions is counted in each, so the
// partition rows do not add up to the collapsed cluster totals.
const normalize = (el) => el.textContent.replace(/\s+/g, ' ').trim();

const renderSection = (props) =>
  render(
    <InfrastructureSection
      title="Slurm"
      isLoading={false}
      isDataLoaded={true}
      handleContextClick={() => {}}
      isInitialLoad={false}
      groupedPerContextGPUs={{}}
      groupedPerNodeGPUs={{}}
      {...props}
    />
  );

// Desktop table and mobile cards render the same data, so scope queries to the
// table.
const tableRows = (container) =>
  Array.from(container.querySelectorAll('table tbody tr'));

const cellTexts = (row) =>
  Array.from(row.querySelectorAll('td')).map(normalize);

const expandPartitions = () =>
  fireEvent.click(screen.getAllByTitle('Show partitions')[0]);

describe('InfrastructureSection Slurm partition rows', () => {
  const slurmNode = (nodeName, partition, gpuName, total, free) => ({
    node_name: nodeName,
    partition,
    gpu_name: gpuName,
    gpu_total: total,
    gpu_free: free,
    cluster: 'nebius-slinky',
  });

  // 'real*' is the default partition (sinfo marks it with a trailing '*');
  // n3/n4 sit in both fabric-a and mock, and mock also holds an A100 node so
  // one partition spans two GPU types.
  const nodes = [
    slurmNode('n1', 'real*', 'H100', 8, 0),
    slurmNode('n2', 'real*', 'H100', 8, 4),
    slurmNode('n3', 'fabric-a,mock', 'H100', 8, 8),
    slurmNode('n4', 'fabric-a,mock', 'H100', 8, 0),
    slurmNode('n5', 'mock', 'A100', 4, 4),
  ];

  const renderSlurm = () =>
    renderSection({
      isSlurm: true,
      contexts: ['nebius-slinky'],
      gpus: [{ gpu_name: 'H100', gpu_total: 32, gpu_free: 12 }],
      groupedPerContextGPUs: {
        'nebius-slinky': [
          {
            gpu_name: 'H100',
            gpu_total: 32,
            gpu_free: 12,
            cluster: 'nebius-slinky',
          },
        ],
      },
      groupedPerNodeGPUs: { 'nebius-slinky': nodes },
    });

  it('collapses to the cluster totals, counting what expanding reveals', () => {
    const { container } = renderSlurm();
    const rows = tableRows(container);
    // One row for the cluster's single GPU type, not one per partition.
    expect(rows).toHaveLength(1);
    expect(cellTexts(rows[0])).toContain('3 partitions');
    expect(cellTexts(rows[0])).toEqual(
      expect.arrayContaining(['H100', '12 of 32 free'])
    );
    expect(normalize(rows[0])).toMatch(/nebius-slinky\s*5/);
    expect(container.querySelector('table').textContent).not.toContain(
      'fabric-a'
    );
  });

  it('appends the partition rows under the cluster totals when expanded', () => {
    const { container } = renderSlurm();
    expandPartitions();
    const rows = tableRows(container);
    // The cluster's own row stays on top, keeping the aggregate on screen.
    expect(rows).toHaveLength(5);
    expect(cellTexts(rows[0])).toEqual(
      expect.arrayContaining(['3 partitions', 'H100', '12 of 32 free'])
    );
    expect(cellTexts(rows[1])).toContain('fabric-a');
    expect(cellTexts(rows[2])).toContain('mock');
    // The cluster's node count stays deduplicated: 5 nodes, not one per
    // partition membership.
    expect(normalize(rows[0])).toMatch(/nebius-slinky\s*5/);
  });

  it('counts a single-partition cluster the same way', () => {
    renderSection({
      isSlurm: true,
      contexts: ['crusoe-slurm-use1'],
      gpus: [{ gpu_name: 'A100', gpu_total: 8, gpu_free: 8 }],
      groupedPerContextGPUs: {
        'crusoe-slurm-use1': [{ gpu_name: 'A100', gpu_total: 8, gpu_free: 8 }],
      },
      groupedPerNodeGPUs: {
        'crusoe-slurm-use1': [
          {
            node_name: 'c1',
            partition: 'all*',
            gpu_name: 'A100',
            gpu_total: 8,
            gpu_free: 8,
          },
        ],
      },
    });
    expect(screen.getAllByText(/1 partition$/)[0]).toBeTruthy();
    expandPartitions();
    expect(screen.getAllByText('all')[0]).toBeTruthy();
  });

  it('sums GPU capacity per partition from its member nodes', () => {
    const { container } = renderSlurm();
    expandPartitions();
    const rows = tableRows(container);
    // fabric-a and mock both count n3 + n4's 16 H100s: shared capacity is
    // reachable from either partition, so it is not split between them.
    expect(cellTexts(rows[1])).toEqual(
      expect.arrayContaining(['H100', '8 of 16 free'])
    );
    expect(cellTexts(rows[2])).toEqual(
      expect.arrayContaining(['H100', '8 of 16 free'])
    );
    expect(cellTexts(rows[3])).toEqual(
      expect.arrayContaining(['A100', '4 of 4 free'])
    );
    expect(cellTexts(rows[4])).toEqual(
      expect.arrayContaining(['H100', '4 of 16 free'])
    );
  });

  it('marks the default partition and spans multi-GPU-type partitions', () => {
    const { container } = renderSlurm();
    expandPartitions();
    const rows = tableRows(container);
    // The gap before the tag is a margin, not whitespace in the text.
    expect(cellTexts(rows[4])).toContain('real(default)');
    // 'mock' has H100 and A100 rows, so its cell spans both; the second row
    // carries only the GPU cells.
    const mockCell = Array.from(rows[2].querySelectorAll('td')).find((td) =>
      td.textContent.includes('mock')
    );
    expect(mockCell.getAttribute('rowspan')).toBe('2');
    expect(rows[3].querySelectorAll('td')).toHaveLength(3);
  });

  it('gives the cluster cells a rowSpan covering totals and partitions', () => {
    const { container } = renderSlurm();
    expandPartitions();
    const nameCell = Array.from(
      container.querySelectorAll('table tbody td')
    ).find((td) => td.textContent.includes('nebius-slinky'));
    expect(nameCell.getAttribute('rowspan')).toBe('5');
    // The partition count spans only the cluster's own rows.
    const summaryCell = Array.from(
      container.querySelectorAll('table tbody td')
    ).find((td) => td.textContent.includes('3 partitions'));
    expect(summaryCell.getAttribute('rowspan')).toBe('1');
  });
});

// A cluster configured in ~/.slurm/config is listed whether or not it answers
// a query, so an unreachable login node leaves the section with a cluster name
// and nothing else to show for it.
describe('InfrastructureSection unreachable Slurm cluster', () => {
  it('renders one row with empty node, partition and GPU cells', () => {
    const { container } = renderSection({
      isSlurm: true,
      contexts: ['offline-cluster'],
      gpus: [],
      groupedPerContextGPUs: {},
      groupedPerNodeGPUs: {},
    });
    const rows = tableRows(container);
    expect(rows).toHaveLength(1);
    expect(normalize(rows[0])).toMatch(/offline-cluster\s*0/);
    // Partition cell plus the three GPU cells, all with nothing to report.
    expect(cellTexts(rows[0]).filter((text) => text === '-')).toHaveLength(4);
    expect(container.textContent).toContain('1 cluster');
    expect(container.textContent).not.toContain('not configured');
  });

  // The row's identity is what a plugin keys its status off, so it has to be
  // the same for an unreachable cluster as for a healthy one.
  it('keeps the namePrefix slot on the row, keyed slurm/<cluster name>', () => {
    const { container } = renderSection({
      isSlurm: true,
      contexts: ['offline-cluster'],
      gpus: [],
      groupedPerContextGPUs: {},
      groupedPerNodeGPUs: {},
    });
    const slot = container.querySelector(
      'table tbody [data-slot="infra.row.namePrefix"]'
    );
    expect(slot).not.toBeNull();
    expect(slot.getAttribute('data-row-kind')).toBe('slurm');
    expect(slot.getAttribute('data-row-id')).toBe('offline-cluster');
  });
});

// Mirrors slurm_catalog.list_accelerators_realtime, which feeds the
// "Requestable: N / node" tooltip. If the backend rule changes, this is the
// copy that has to move with it.
describe('slurmRequestableCounts', () => {
  it('returns powers of two up to a power-of-two node size', () => {
    expect(slurmRequestableCounts(8)).toEqual([1, 2, 4, 8]);
  });

  it('appends the node size when it is not a power of two', () => {
    expect(slurmRequestableCounts(6)).toEqual([1, 2, 4, 6]);
  });

  it('returns nothing for a CPU-only node', () => {
    expect(slurmRequestableCounts(0)).toEqual([]);
  });
});

describe('InfrastructureSection Kubernetes GPU-type rows', () => {
  it('still renders one row per GPU type with no partition column', () => {
    const { container } = renderSection({
      title: 'Kubernetes',
      contexts: ['usw9b'],
      gpus: [
        { gpu_name: 'H100', gpu_total: 256, gpu_free: 67 },
        { gpu_name: 'B200', gpu_total: 256, gpu_free: 88 },
      ],
      groupedPerContextGPUs: {
        usw9b: [
          { gpu_name: 'H100', gpu_total: 256, gpu_free: 67, context: 'usw9b' },
          { gpu_name: 'B200', gpu_total: 256, gpu_free: 88, context: 'usw9b' },
        ],
      },
      groupedPerNodeGPUs: { usw9b: [] },
      loadedContexts: new Set(['usw9b']),
    });
    const rows = tableRows(container);
    expect(rows).toHaveLength(2);
    expect(cellTexts(rows[0])).toEqual(
      expect.arrayContaining(['H100', '67 of 256 free'])
    );
    expect(cellTexts(rows[1])).toEqual(
      expect.arrayContaining(['B200', '88 of 256 free'])
    );
    expect(container.querySelector('table').textContent).not.toContain(
      'Partition'
    );
  });
});

// The Cloud panel row filter: a provider with live clusters or jobs must
// stay visible even when sky check does not list it as enabled, and count
// joins are lowercase so provider _REPR values ('GCP', 'Vast') always land.

describe('Cloud panel row visibility', () => {
  const rows = (names) => names.map((name) => ({ name, clusters: 0, jobs: 0 }));

  it('keeps a cloud with live clusters when the workspace only enables others', () => {
    // Production shape: GCP runs 6 clusters, sky check reports only AWS.
    const kept = visibleCloudRows(
      rows(['AWS', 'GCP']),
      ['aws'],
      { gcp: 6 },
      {}
    );
    expect(kept.map((cloud) => cloud.name)).toEqual(['AWS', 'GCP']);
  });

  it('keeps a cloud whose row carries counts even before the count maps load', () => {
    const kept = visibleCloudRows(
      [{ name: 'Quantacloud', clusters: 2, jobs: 0 }],
      ['aws'],
      {},
      {}
    );
    expect(kept).toHaveLength(1);
  });

  it('drops a provider that is neither enabled nor has live resources', () => {
    const kept = visibleCloudRows(
      rows(['AWS', 'Azure', 'GCP']),
      ['aws'],
      {},
      {}
    );
    expect(kept.map((cloud) => cloud.name)).toEqual(['AWS']);
  });

  it('shows all rows while workspace cloud data is still loading', () => {
    const all = rows(['AWS', 'GCP']);
    expect(visibleCloudRows(all, null, {}, {})).toEqual(all);
  });
});

describe('countByCloud', () => {
  it('keys counts by lowercase cloud name', () => {
    expect(
      countByCloud([{ cloud: 'Vast' }, { cloud: 'GCP' }, { cloud: 'GCP' }])
    ).toEqual({ vast: 1, gcp: 2 });
  });

  it('skips records without a cloud and tolerates null input', () => {
    expect(countByCloud([{ cloud: 'GCP' }, { name: 'no-cloud' }])).toEqual({
      gcp: 1,
    });
    expect(countByCloud(null)).toEqual({});
  });
});

// Workspace scoping for the Cloud panel: with a workspace selected, counts
// must come only from that workspace's records so activity in another
// workspace does not leak into the visible rows or their counts.
describe('recordsInWorkspace', () => {
  const records = [
    { cloud: 'GCP', workspace: 'team-a' },
    { cloud: 'GCP', workspace: 'team-b' },
    { cloud: 'AWS', workspace: 'team-b' },
  ];

  it('scopes counts to the selected workspace', () => {
    expect(countByCloud(recordsInWorkspace(records, 'team-a'))).toEqual({
      gcp: 1,
    });
    expect(countByCloud(recordsInWorkspace(records, 'team-b'))).toEqual({
      gcp: 1,
      aws: 1,
    });
  });

  it('keeps every record for the All Workspaces option', () => {
    expect(countByCloud(recordsInWorkspace(records, 'all'))).toEqual({
      gcp: 2,
      aws: 1,
    });
  });

  it('treats a record without a workspace as outside a specific workspace', () => {
    expect(
      countByCloud(recordsInWorkspace([{ cloud: 'GCP' }], 'team-a'))
    ).toEqual({});
  });
});

// Per-provider GPU capacity for the Cloud panel: accelerators hold per-node
// counts, so chips = perNode × nodes; canonical types aggregate across
// clusters, and cluster status splits ready capacity from the rest.
describe('gpuCapacityByCloud', () => {
  it('multiplies per-node accelerators by the cluster node count', () => {
    const capacity = gpuCapacityByCloud([
      { cloud: 'GCP', accelerators: { H100: 2 }, nodes: 3, status: 'UP' },
    ]);
    expect(capacity.gcp).toEqual([
      { type: 'H100', chips: 6, nodes: 3, readyChips: 6, notReadyChips: 0 },
    ]);
  });

  it('treats the dashboard-mapped RUNNING status as ready', () => {
    // Live cluster records from the clusters connector carry the mapped
    // status ('RUNNING'), not the backend's raw 'UP' — both must count as
    // ready capacity (prod shape: 8 RTXPRO6000 workers all RUNNING).
    const capacity = gpuCapacityByCloud([
      {
        cloud: 'GCP',
        accelerators: { RTXPRO6000: 1 },
        nodes: 1,
        status: 'RUNNING',
      },
      {
        cloud: 'GCP',
        accelerators: { RTXPRO6000: 1 },
        nodes: 1,
        status: 'RUNNING',
      },
    ]);
    expect(capacity.gcp).toEqual([
      {
        type: 'RTXPRO6000',
        chips: 2,
        nodes: 2,
        readyChips: 2,
        notReadyChips: 0,
      },
    ]);
  });

  it('aggregates multiple clusters of the same type per provider', () => {
    const capacity = gpuCapacityByCloud([
      { cloud: 'GCP', gpus: { H100: 1 }, num_nodes: 2, status: 'UP' },
      { cloud: 'GCP', gpus: { H100: 4 }, num_nodes: 1, status: 'LAUNCHING' },
      { cloud: 'Vast', accelerators: { H100: 1 }, nodes: 1, status: 'UP' },
    ]);
    expect(capacity.gcp).toEqual([
      { type: 'H100', chips: 6, nodes: 3, readyChips: 2, notReadyChips: 4 },
    ]);
    expect(capacity.vast).toEqual([
      { type: 'H100', chips: 1, nodes: 1, readyChips: 1, notReadyChips: 0 },
    ]);
  });

  it('canonicalizes mixed GPU type names into one entry', () => {
    const capacity = gpuCapacityByCloud([
      {
        cloud: 'GCP',
        accelerators: { 'NVIDIA H100': 1 },
        nodes: 1,
        status: 'UP',
      },
      { cloud: 'GCP', accelerators: { h100: 2 }, nodes: 1, status: 'UP' },
      {
        cloud: 'GCP',
        accelerators: { 'A100-SXM4-80GB': 1 },
        nodes: 1,
        status: 'UP',
      },
    ]);
    expect(capacity.gcp).toEqual([
      { type: 'H100', chips: 3, nodes: 2, readyChips: 3, notReadyChips: 0 },
      { type: 'A100', chips: 1, nodes: 1, readyChips: 1, notReadyChips: 0 },
    ]);
  });

  it('counts a record nodes once per canonical type across aliases', () => {
    const capacity = gpuCapacityByCloud([
      {
        cloud: 'GCP',
        // Two aliases of one canonical type in ONE accelerator map: the
        // chips include both, but the record's nodes count once.
        accelerators: { 'NVIDIA H100': 2, H100: 1 },
        nodes: 2,
        status: 'UP',
      },
    ]);
    expect(capacity.gcp).toEqual([
      { type: 'H100', chips: 6, nodes: 2, readyChips: 6, notReadyChips: 0 },
    ]);
  });

  it('tolerates CPU-only records and missing fields', () => {
    expect(gpuCapacityByCloud(null)).toEqual({});
    const capacity = gpuCapacityByCloud([
      // CPU-only: no accelerators at all.
      { cloud: 'GCP', accelerators: null, nodes: 2, status: 'UP' },
      { cloud: 'GCP', gpus: {}, num_nodes: 1, status: 'UP' },
      { cloud: 'GCP', accelerators: 'None', nodes: 1, status: 'UP' },
      // No cloud means the record cannot join a provider row.
      { accelerators: { H100: 1 }, nodes: 1, status: 'UP' },
      // Zero counts are not capacity.
      { cloud: 'GCP', accelerators: { H100: 0 }, nodes: 1, status: 'UP' },
      // Missing nodes still counts as the cluster's one node.
      { cloud: 'GCP', accelerators: { H100: 1 }, status: 'UP' },
    ]);
    expect(capacity.gcp).toEqual([
      { type: 'H100', chips: 1, nodes: 1, readyChips: 1, notReadyChips: 0 },
    ]);
  });

  it('scopes capacity to the selected workspace like the panel counts', () => {
    const records = [
      {
        cloud: 'GCP',
        accelerators: { H100: 1 },
        nodes: 1,
        status: 'UP',
        workspace: 'team-a',
      },
      {
        cloud: 'GCP',
        accelerators: { H100: 2 },
        nodes: 2,
        status: 'UP',
        workspace: 'team-b',
      },
    ];
    expect(gpuCapacityByCloud(recordsInWorkspace(records, 'team-a'))).toEqual({
      gcp: [
        { type: 'H100', chips: 1, nodes: 1, readyChips: 1, notReadyChips: 0 },
      ],
    });
    expect(gpuCapacityByCloud(recordsInWorkspace(records, 'all'))).toEqual({
      gcp: [
        { type: 'H100', chips: 5, nodes: 3, readyChips: 5, notReadyChips: 0 },
      ],
    });
  });
});

// The Cloud panel renders each provider's GPU capacity inline in its row: one
// line per GPU type with the total chips and node count. A fixture provider
// with six single-GPU RTXPRO6000 clusters shows one RTXPRO6000 line.
describe('CloudGpuCapacityCell', () => {
  it('renders the fixture provider capacity', () => {
    const records = Array.from({ length: 6 }, () => ({
      cloud: 'GCP',
      accelerators: { RTXPRO6000: 1 },
      nodes: 1,
      status: 'UP',
      workspace: 'default',
    }));
    render(
      <CloudGpuCapacityCell
        entries={gpuCapacityByCloud(records).gcp}
        loading={false}
      />
    );
    expect(screen.getByText('RTXPRO6000')).toBeInTheDocument();
    expect(screen.getByText('×6 chips · 6 nodes')).toBeInTheDocument();
  });

  it('shows the up/not-ready split when some clusters are not UP', () => {
    const capacity = gpuCapacityByCloud([
      { cloud: 'GCP', accelerators: { H100: 2 }, nodes: 2, status: 'UP' },
      {
        cloud: 'GCP',
        accelerators: { H100: 1 },
        nodes: 1,
        status: 'LAUNCHING',
      },
    ]);
    render(<CloudGpuCapacityCell entries={capacity.gcp} loading={false} />);
    expect(screen.getByText('×5 chips · 3 nodes')).toBeInTheDocument();
    expect(screen.getByText('4 up')).toBeInTheDocument();
    expect(screen.getByText('1 not ready')).toBeInTheDocument();
  });

  it('renders an em dash for CPU-only providers', () => {
    render(<CloudGpuCapacityCell entries={[]} loading={false} />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });
});

// The unified Providers section renders exactly one compact row per
// provider — the live fleet (clusters, jobs, our GPUs) merged with the
// catalog tier, cheapest price and freshness — and expands each row into
// the per-GPU detail table (offerings + live per-region stock) that used to
// be the separate Provider Offerings and GPU Stock by Region sections.
describe('ProvidersSection', () => {
  // three hours ago, in unix seconds (the CSV mtime contract)
  const threeHoursAgo = Math.floor(Date.now() / 1000) - 3 * 3600;

  const providersFixture = {
    aws: {
      present: true,
      updated_at: threeHoursAgo,
      stock_mode: 'catalog',
      gpus: [
        {
          name: 'P5',
          counts: [8],
          price: 98.32,
          spot_price: 40.96,
          regions: ['us-east-1', 'eu-west-1'],
        },
      ],
    },
    vast: {
      present: true,
      updated_at: threeHoursAgo,
      stock_mode: 'units',
      gpus: [
        {
          name: 'H100',
          counts: [1],
          price: 2.4,
          spot_price: 1.1,
          regions: ['US, NA', 'EU'],
          region_details: [
            {
              region: 'US, NA',
              units: 6,
              listings: 4,
              price: 2.4,
              spot_price: 1.1,
            },
            {
              region: 'EU',
              units: 2,
              listings: 1,
              price: 2.6,
              spot_price: null,
            },
          ],
        },
        {
          name: 'A100',
          counts: [2, 1],
          price: 0.93,
          spot_price: 0.18,
          regions: ['US, NA'],
          region_details: [
            {
              region: 'US, NA',
              units: 10,
              listings: 3,
              price: 0.93,
              spot_price: 0.18,
            },
          ],
        },
      ],
    },
    lambda: {
      present: true,
      updated_at: threeHoursAgo,
      stock_mode: 'in_stock',
      gpus: [
        {
          name: 'H100',
          counts: [1, 2],
          price: 2.99,
          spot_price: 1.49,
          regions: ['us-east-1'],
          region_details: [
            {
              region: 'us-east-1',
              units: null,
              listings: null,
              price: 2.99,
              spot_price: 1.49,
            },
          ],
        },
      ],
    },
    // Spheron-style raw region ids: hash-prefixed, five regions so the
    // live-stock chips exercise the all-regions listing.
    spheron: {
      present: true,
      updated_at: threeHoursAgo,
      stock_mode: 'units',
      gpus: [
        {
          name: 'H200',
          counts: [1],
          price: 2.15,
          spot_price: null,
          regions: [
            '90e8fe74:newyork-usa-1',
            '90e8fe74:us-1',
            '90e8fe74:eu-1',
            '90e8fe74:la-1',
            '90e8fe74:ap-1',
          ],
          region_details: [
            {
              region: '90e8fe74:newyork-usa-1',
              units: 27,
              listings: 12,
              price: 2.15,
              spot_price: null,
            },
            { region: '90e8fe74:us-1', units: 18, listings: 8 },
            { region: '90e8fe74:eu-1', units: 9, listings: 4 },
            { region: '90e8fe74:la-1', units: 5, listings: 2 },
            { region: '90e8fe74:ap-1', units: 3, listings: 1 },
          ],
        },
      ],
    },
    ornn: { present: false, updated_at: null, gpus: null },
  };

  const cloudRowsFixture = [
    { name: 'Vast', clusters: 3, jobs: 5, enabled: true },
    // Extra cloud row outside the defaults: still a summary row.
    { name: 'Fluidstack', clusters: 1, jobs: 0, enabled: true },
  ];

  const capacityFixture = {
    vast: [
      { type: 'H100', chips: 6, nodes: 3, readyChips: 6, notReadyChips: 0 },
    ],
  };

  const renderProviders = (props) =>
    render(
      <ProvidersSection
        cloudRows={cloudRowsFixture}
        providers={providersFixture}
        gpuCapacity={capacityFixture}
        loading={false}
        error={null}
        {...props}
      />
    );

  // The summary table is the first table in the section; expanded detail
  // rows live in a nested table inside a colspan cell.
  const summaryRows = (container) =>
    Array.from(
      container.querySelectorAll('table')[0].querySelectorAll('tbody tr')
    );

  const detailRows = (container) =>
    container.querySelectorAll('table').length > 1
      ? Array.from(
          container.querySelectorAll('table')[1].querySelectorAll('tbody tr')
        ).filter((row) => !row.closest('thead'))
      : [];

  const rowByName = (container, name) =>
    summaryRows(container).find((row) => normalize(row).includes(name));

  it('renders exactly one row per provider in the union, with the header badge', () => {
    const { container } = renderProviders();
    // 11 defaults + the Fluidstack cloud row extra.
    expect(summaryRows(container)).toHaveLength(
      PROVIDER_OFFERINGS_CLOUDS.length + 1
    );
    ['AWS', 'Spheron', 'Lambda', 'Vast', 'Fluidstack', 'Ornn'].forEach((name) =>
      expect(screen.getByText(name)).toBeInTheDocument()
    );
    expect(screen.getByText('4 of 12 with catalog')).toBeInTheDocument();
    expect(screen.getByText('Providers')).toBeInTheDocument();
  });

  it('renders the Stock column per data tier', () => {
    const { container } = renderProviders();
    // units: summed chips · distinct regions reporting them
    expect(screen.getByText('18 chips · 2 regions')).toBeInTheDocument(); // vast
    expect(screen.getByText('62 chips · 5 regions')).toBeInTheDocument(); // spheron
    // in_stock: distinct regions where offered
    expect(screen.getByText('1 region')).toBeInTheDocument(); // lambda
    // catalog: the catalog's GPU count
    expect(screen.getByText('1 GPU')).toBeInTheDocument(); // aws
    // absent: nothing to report
    const ornnRow = rowByName(container, 'Ornn');
    expect(ornnRow).toBeTruthy();
    expect(cellTexts(ornnRow)).toContain('—');
  });

  it('renders the mode badge under each provider name', () => {
    renderProviders();
    // +1 occurrence of each label: the legend line reuses the badges.
    expect(screen.getAllByText('live units')).toHaveLength(3); // vast, spheron, legend
    expect(screen.getAllByText('in stock')).toHaveLength(2); // lambda, legend
    expect(screen.getAllByText('catalog')).toHaveLength(2); // aws, legend
    // Every absent provider (7 defaults + Fluidstack) keeps the outline
    // no-catalog badge, plus the legend's own.
    expect(screen.getAllByText('no catalog')).toHaveLength(9);
  });

  it('glosses the stock tiers in a compact legend line', () => {
    renderProviders();
    expect(screen.getByText('Stock tiers:')).toBeInTheDocument();
    // The legend answers the "is this what we rented?" reading: every
    // tier is provider-side bookable capacity, never our own fleet.
    expect(
      screen.getByText(
        'bookable at the provider now, with live unit counts per region'
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'offered right now, but the provider publishes no unit counts'
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText('price list only, no live availability')
    ).toBeInTheDocument();
    expect(
      screen.getByText('local catalog data is unavailable')
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Stock is the provider’s side; our fleet is Clusters · Jobs · Our GPUs.'
      )
    ).toBeInTheDocument();
  });

  it('merges the live fleet: clusters, jobs and our GPUs in the row', () => {
    const { container } = renderProviders();
    const vastRow = rowByName(container, 'Vast');
    expect(cellTexts(vastRow)).toEqual(expect.arrayContaining(['3', '5']));
    expect(normalize(vastRow)).toContain('×6 chips · 3 nodes');
    const ornnRow = rowByName(container, 'Ornn');
    expect(cellTexts(ornnRow)).toEqual(expect.arrayContaining(['0', '0']));
  });

  it('renders the cheapest catalog price and the relative freshness', () => {
    renderProviders();
    expect(screen.getByText('from $0.93')).toBeInTheDocument(); // vast
    expect(screen.getByText('from $98.32')).toBeInTheDocument(); // aws
    expect(screen.getByText('from $2.15')).toBeInTheDocument(); // spheron
    expect(screen.getAllByText(/3h ago/)).toHaveLength(4); // one per catalog
  });

  it('expands into the per-GPU detail table on chevron click', () => {
    const { container } = renderProviders();
    expect(detailRows(container)).toHaveLength(0);
    fireEvent.click(screen.getByRole('button', { name: 'Expand Vast' }));
    // GPU rows sort by name: A100 before H100.
    const rows = detailRows(container);
    expect(rows).toHaveLength(2);
    expect(normalize(rows[0])).toContain('A100');
    expect(normalize(rows[1])).toContain('H100');
    expect(normalize(rows[0])).toContain('$0.93');
    expect(normalize(rows[0])).toContain('$0.18');
    // Counts carries the GPU's total stock: A100 10 units, H100 6+2.
    expect(normalize(rows[0])).toContain('10 units');
    expect(normalize(rows[1])).toContain('8 units');
    // Regions collapse to a count; hover carries per-region stock, one
    // region per line (aria-label mirrors the tooltip so it stays
    // queryable).
    const h100Regions = screen.getByRole('button', { name: /US, NA ×6/ });
    expect(h100Regions.getAttribute('aria-label')).toBe('US, NA ×6\nEU ×2');
    expect(
      screen
        .getByRole('button', { name: 'US, NA ×10' })
        .getAttribute('aria-label')
    ).toBe('US, NA ×10');
    // Collapsing removes the detail rows again.
    fireEvent.click(screen.getByRole('button', { name: 'Collapse Vast' }));
    expect(detailRows(container)).toHaveLength(0);
  });

  it('carries per-region stock on the region hover, busiest first', () => {
    const { container } = renderProviders();
    fireEvent.click(screen.getByRole('button', { name: 'Expand Spheron' }));
    const rows = detailRows(container);
    expect(rows).toHaveLength(1);
    // Counts carries the GPU total: 27+18+9+5+3.
    expect(normalize(rows[0])).toContain('62 units');
    // Hover carries every region's count, busiest first, one per line.
    const regions = screen.getByRole('button', { name: /newyork-usa-1 ×27/ });
    expect(regions.getAttribute('aria-label')).toBe(
      'newyork-usa-1 ×27\nus-1 ×18\neu-1 ×9\nla-1 ×5\nap-1 ×3'
    );
  });

  it('links each provider row to its verified console', () => {
    renderProviders();
    expect(
      screen.getByRole('link', { name: 'Open Vast console' })
    ).toHaveAttribute('href', 'https://cloud.vast.ai');
    // Spheron's console URL is not verifiable — no link rendered.
    expect(
      screen.queryByRole('link', { name: 'Open Spheron console' })
    ).not.toBeInTheDocument();
    // Providers without a console entry render no link at all.
    expect(
      screen.queryByRole('link', { name: 'Open Fluidstack console' })
    ).not.toBeInTheDocument();
  });

  it('lists regions without counts when the provider publishes none', () => {
    const { container } = renderProviders();
    fireEvent.click(screen.getByRole('button', { name: 'Expand Lambda' }));
    const rows = detailRows(container);
    expect(rows).toHaveLength(1);
    // No unit data: Counts falls back to the offering's config counts.
    expect(normalize(rows[0])).toContain('1, 2');
    // The region hover lists regions, one per line, without counts.
    expect(
      screen
        .getByRole('button', { name: 'us-east-1' })
        .getAttribute('aria-label')
    ).toBe('us-east-1');
  });

  it('renders the absent provider muted with no expand chevron', () => {
    const { container } = renderProviders();
    const ornnRow = rowByName(container, 'Ornn');
    const nameSpan = Array.from(ornnRow.querySelectorAll('span')).find(
      (el) => el.textContent === 'Ornn'
    );
    expect(nameSpan.className).toContain('text-gray-400');
    expect(
      screen.queryByRole('button', { name: 'Expand Ornn' })
    ).not.toBeInTheDocument();
  });

  it('renders skeleton rows while loading and no provider names', () => {
    const { container } = renderProviders({
      loading: true,
      providers: {},
      cloudRows: [],
    });
    expect(summaryRows(container)).toHaveLength(4);
    expect(container.textContent).not.toContain('AWS');
    expect(container.textContent).toContain('Providers');
  });

  it('renders the error in-section instead of crashing', () => {
    const { container } = renderProviders({
      error: 'Failed to get provider GPU offerings with status 502',
    });
    expect(container.textContent).toContain(
      'Failed to load provider catalogs: Failed to get provider GPU offerings with status 502'
    );
  });
  it('keeps live cloud rows when the catalog fetch fails', () => {
    const { container } = renderProviders({
      error: 'Failed to get provider GPU offerings with status 502',
    });
    // The cloud fetch and the catalog fetch are independent: the error is
    // a notice above the table, never a takeover that hides fleet rows.
    expect(summaryRows(container).length).toBeGreaterThan(0);
    expect(screen.getByText('AWS')).toBeInTheDocument();
    expect(container.textContent).toContain(
      'catalog columns are blank until the next refresh'
    );
  });

  it('keeps live cloud rows while the catalog is refreshing', () => {
    const { container } = renderProviders({
      loading: true,
      providers: {},
    });
    // cloudRows present -> the rows render; the full skeleton only owns
    // the no-data-at-all first load.
    expect(summaryRows(container).length).toBeGreaterThan(0);
    expect(screen.getByText('AWS')).toBeInTheDocument();
  });
});
