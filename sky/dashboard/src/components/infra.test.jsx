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
  ProviderOfferingsSection,
  GpuStockSection,
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

// The Provider Offerings section lists what every default provider's catalog
// currently offers, independent of live clusters: a present provider renders
// one row per GPU type (sorted by name), and an absent or missing provider
// still gets its row with a muted "no catalog yet" note.
describe('ProviderOfferingsSection', () => {
  // three hours ago, in unix seconds (the CSV mtime contract)
  const threeHoursAgo = Math.floor(Date.now() / 1000) - 3 * 3600;

  const fixture = {
    vast: {
      present: true,
      updated_at: threeHoursAgo,
      gpus: [
        // deliberately unsorted: the section must sort rows by GPU name
        {
          name: 'RTX6000Ada',
          counts: [2, 1],
          price: 0.35,
          spot_price: 0.18,
          regions: ['us-west', 'us-east'],
        },
        {
          name: 'H100',
          counts: [1],
          price: 2.4,
          spot_price: null,
          regions: ['us-east'],
        },
      ],
    },
    ornn: { present: false, updated_at: null, gpus: null },
  };

  const renderOfferings = (props) =>
    render(
      <ProviderOfferingsSection
        providers={fixture}
        loading={false}
        error={null}
        {...props}
      />
    );

  it('renders every default provider, present or absent', () => {
    const { container } = renderOfferings({ providers: fixture });
    // Every default provider keeps a row, plus Vast's second GPU type.
    expect(tableRows(container)).toHaveLength(
      PROVIDER_OFFERINGS_CLOUDS.length + 1
    );
    ['AWS', 'GCP', 'Shadeform', 'Spheron', 'Lambda', 'Vast'].forEach((name) =>
      expect(screen.getByText(name)).toBeInTheDocument()
    );
  });

  it('renders the present provider GPU rows sorted by name with formatted prices', () => {
    const { container } = renderOfferings({ providers: fixture });
    const rows = tableRows(container);
    // Vast is the 6th default provider, so its rows follow five absent
    // providers' single rows; H100 sorts before RTX6000Ada.
    const vastRows = rows.slice(5, 7);
    expect(normalize(vastRows[0])).toContain('H100');
    expect(normalize(vastRows[1])).toContain('RTX6000Ada');
    // counts joined, prices $X.XX
    expect(normalize(vastRows[1])).toContain('1, 2');
    expect(screen.getByText('$0.35')).toBeInTheDocument();
    expect(screen.getByText('$0.18')).toBeInTheDocument();
    expect(screen.getByText('$2.40')).toBeInTheDocument();
    // region list renders in full (truncation is CSS-only, tooltip has all)
    expect(screen.getByText('us-east, us-west')).toBeInTheDocument();
  });

  it('renders the absent provider with the no-catalog note, not a price row', () => {
    const { container } = renderOfferings({ providers: fixture });
    const ornnRow = tableRows(container).find((row) =>
      normalize(row).includes('Ornn')
    );
    expect(ornnRow).toBeTruthy();
    expect(normalize(ornnRow)).toContain(
      'No catalog yet — refreshed by the controller every few hours'
    );
    expect(normalize(ornnRow)).not.toContain('$');
  });

  it('shows the catalog freshness as a relative time', () => {
    renderOfferings({ providers: fixture });
    expect(screen.getByText(/3h ago/)).toBeInTheDocument();
  });

  it('defaults missing provider entries to absent rows', () => {
    const { container } = renderOfferings({ providers: {} });
    // No provider was reported at all: every default provider still shows.
    expect(tableRows(container)).toHaveLength(PROVIDER_OFFERINGS_CLOUDS.length);
    expect(screen.getAllByText(/No catalog yet/)).toHaveLength(
      PROVIDER_OFFERINGS_CLOUDS.length
    );
  });

  it('renders skeleton rows while loading and no provider names', () => {
    const { container } = renderOfferings({ loading: true, providers: {} });
    expect(tableRows(container)).toHaveLength(4);
    expect(container.textContent).not.toContain('AWS');
    expect(container.textContent).toContain('Provider Offerings');
  });

  it('renders the error in-section instead of crashing', () => {
    const { container } = renderOfferings({
      error: 'Failed to get provider GPU offerings with status 502',
    });
    expect(container.textContent).toContain(
      'Failed to load provider GPU offerings: Failed to get provider GPU offerings with status 502'
    );
  });

  it('renders a present provider with an empty GPU list as a no-GPU row', () => {
    const { container } = renderOfferings({
      providers: {
        vast: { present: true, updated_at: threeHoursAgo, gpus: [] },
      },
    });
    const vastRow = tableRows(container).find((row) =>
      normalize(row).includes('Vast')
    );
    expect(normalize(vastRow)).toContain('No GPUs in the current catalog');
  });
});

// The "GPU Stock by Region" section pivots the same provider-offerings
// response into a provider x region x GPU heatmap. Providers come in three
// data tiers (`stock_mode`): `units` marketplaces publish live chip counts,
// `in_stock` catalogs are live-filtered to stocked regions without counts,
// and `catalog` clouds only have static offering coverage. Older servers
// send no `stock_mode` — the section then derives it from region_details.
describe('GpuStockSection', () => {
  const threeHoursAgo = Math.floor(Date.now() / 1000) - 3 * 3600;

  const stockFixture = {
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
          region_details: [
            {
              region: 'us-east-1',
              units: null,
              listings: null,
              price: 98.32,
              spot_price: 40.96,
            },
            {
              region: 'eu-west-1',
              units: null,
              listings: null,
              price: 101.12,
              spot_price: 42.0,
            },
          ],
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
          counts: [1, 2],
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
    primeintellect: {
      present: true,
      updated_at: threeHoursAgo,
      stock_mode: 'units',
      gpus: [
        {
          name: 'MI300X',
          counts: [8],
          price: 3.0,
          spot_price: null,
          regions: ['US'],
          region_details: [
            {
              region: 'US',
              units: 5,
              listings: 2,
              price: 3.0,
              spot_price: null,
            },
          ],
        },
      ],
    },
    ornn: { present: false, updated_at: null, gpus: null },
  };

  const renderStock = (props) =>
    render(
      <GpuStockSection
        providers={stockFixture}
        loading={false}
        error={null}
        {...props}
      />
    );

  const ariaLabels = (container) =>
    Array.from(container.querySelectorAll('[aria-label]')).map((element) =>
      element.getAttribute('aria-label')
    );

  it('renders provider chips with live stock totals and honest tier labels', () => {
    renderStock();
    // Two stock providers show their total live chips, the in-stock catalog
    // shows its tier, and the on-demand cloud shows plain catalog coverage.
    expect(screen.getByText('18 chips')).toBeInTheDocument(); // vast: 6+2+10
    expect(screen.getByText('5 chips')).toBeInTheDocument(); // primeintellect
    expect(screen.getByText('in stock')).toBeInTheDocument(); // lambda
    expect(screen.getByText('catalog')).toBeInTheDocument(); // aws
    // Absent providers keep a greyed, non-clickable pill with the shared
    // no-catalog note.
    const ornnChip = screen
      .getAllByLabelText(
        'No catalog yet — refreshed by the controller every few hours'
      )
      .find((element) => element.textContent === 'Ornn');
    expect(ornnChip).toBeTruthy();
    expect(ornnChip.className).toContain('cursor-not-allowed');
    expect(ornnChip.tagName).toBe('SPAN');
  });

  it('renders the default stock provider as a heatmap with intensity and tooltips', () => {
    const { container } = renderStock();
    // Vast is the first default provider with live stock, so it is the
    // default selection.
    expect(
      screen.getByRole('button', { name: /Vast/ }).getAttribute('aria-pressed')
    ).toBe('true');
    // GPU rows sort by total units desc (A100 10 > H100 8), and region
    // columns by the provider's stock in that region (US, NA 16 > EU 2).
    const rowLabels = Array.from(
      container.querySelectorAll('tbody tr td:first-child')
    ).map(normalize);
    expect(rowLabels[0]).toContain('A100');
    expect(rowLabels[1]).toContain('H100');
    const headerTexts = Array.from(container.querySelectorAll('thead th')).map(
      normalize
    );
    expect(headerTexts[1]).toContain('US, NA');
    expect(headerTexts[2]).toContain('EU');
    // Cells show live chip counts with the emerald intensity scaled to the
    // matrix maximum (10 = max -> darkest step, white text).
    expect(screen.getByText('10').className).toContain('bg-emerald-600');
    expect(screen.getByText('10').className).toContain('text-white');
    expect(screen.getByText('6').className).toContain('bg-emerald-400');
    expect(screen.getByText('2').className).toContain('bg-emerald-100');
    // The A100 row has no EU cell: not offered stays empty.
    expect(screen.queryByLabelText(/A100 · EU/)).not.toBeInTheDocument();
    // Cell tooltip (mirrored on aria-label): chips + listings, per-region
    // prices, and the catalog freshness.
    const cellTooltip = ariaLabels(container).find((label) =>
      label.includes('10 chips · 3 listings')
    );
    expect(cellTooltip).toContain('A100 · US, NA');
    expect(cellTooltip).toContain('cheapest $0.93/hr');
    expect(cellTooltip).toContain('spot $0.18/hr');
    expect(cellTooltip).toContain('updated 3h ago');
    // Summary strip: total chips, region count, tier note, freshness.
    expect(screen.getByText(/18 chips across 2 regions/)).toBeInTheDocument();
    expect(screen.getByText('live units per region')).toBeInTheDocument();
    expect(screen.getByText(/3h ago/)).toBeInTheDocument();
  });

  it('shows the catalog-only provider as neutral dots with the catalog note', () => {
    const { container } = renderStock();
    fireEvent.click(screen.getByRole('button', { name: /AWS/ }));
    // On-demand clouds publish no stock: offered regions render a neutral
    // dot, never a number.
    expect(screen.getAllByText('·')).toHaveLength(2);
    const dotTooltips = ariaLabels(container).filter((label) =>
      label.includes('offered in this region')
    );
    expect(dotTooltips).toHaveLength(2);
    expect(dotTooltips.join('\n')).toContain('P5 · us-east-1');
    expect(dotTooltips.join('\n')).toContain('P5 · eu-west-1');
    expect(container.querySelector('table [class*="bg-emerald"]')).toBeNull();
    // Summary words the tier honestly instead of implying stock.
    expect(
      screen.getByText('catalog coverage — on-demand clouds publish no stock')
    ).toBeInTheDocument();
  });

  it('shows in-stock regions with a check for in_stock providers', () => {
    const { container } = renderStock();
    fireEvent.click(screen.getByRole('button', { name: /Lambda/ }));
    expect(screen.getAllByText('✓')).toHaveLength(1);
    expect(
      screen.getByLabelText(/in stock at last refresh/)
    ).toBeInTheDocument();
    expect(
      screen.getByText('in-stock regions at last refresh — no counts published')
    ).toBeInTheDocument();
    expect(container.querySelector('table [class*="bg-emerald"]')).toBeNull();
  });

  it('re-renders the matrix when switching providers', () => {
    const { container } = renderStock();
    fireEvent.click(screen.getByRole('button', { name: /PrimeIntellect/ }));
    expect(
      screen.getByRole('button', { name: /Vast/ }).getAttribute('aria-pressed')
    ).toBe('false');
    expect(screen.getByText('MI300X')).toBeInTheDocument();
    expect(screen.getByText('US')).toBeInTheDocument();
    expect(screen.getByText(/5 chips across 1 region/)).toBeInTheDocument();
    expect(screen.queryByText('A100')).not.toBeInTheDocument();
    expect(screen.queryByText('US, NA')).not.toBeInTheDocument();
  });

  it('degrades to dots everywhere when region_details is missing (older server)', () => {
    const { container } = renderStock({
      providers: {
        vast: {
          present: true,
          updated_at: threeHoursAgo,
          gpus: [
            {
              name: 'H100',
              counts: [1],
              price: 2.4,
              spot_price: null,
              regions: ['US, NA'],
            },
          ],
        },
      },
    });
    // No stock_mode and no region_details: the provider degrades to catalog
    // coverage (dot cells, catalog chip label) without crashing.
    expect(container.textContent).toContain('GPU Stock by Region');
    expect(screen.getByText('catalog')).toBeInTheDocument();
    expect(screen.getAllByText('·')).toHaveLength(1);
    expect(
      screen.getByText('catalog coverage — on-demand clouds publish no stock')
    ).toBeInTheDocument();
    expect(container.querySelector('table [class*="bg-emerald"]')).toBeNull();
  });

  it('derives units mode from region_details when stock_mode is absent', () => {
    renderStock({
      providers: {
        vast: {
          present: true,
          updated_at: threeHoursAgo,
          gpus: [
            {
              name: 'H100',
              counts: [1],
              price: 2.4,
              spot_price: null,
              regions: ['US, NA'],
              region_details: [
                {
                  region: 'US, NA',
                  units: 6,
                  listings: 4,
                  price: 2.4,
                  spot_price: null,
                },
              ],
            },
          ],
        },
      },
    });
    // Any numeric units value means the provider publishes live counts.
    expect(screen.getByText('6 chips')).toBeInTheDocument();
    expect(screen.getByText('6')).toBeInTheDocument();
    expect(screen.getByText(/6 chips across 1 region/)).toBeInTheDocument();
    expect(screen.getByText('live units per region')).toBeInTheDocument();
  });

  it('words each stock tier honestly', () => {
    renderStock();
    // units tier
    expect(screen.getByText('live units per region')).toBeInTheDocument();
    // in_stock tier
    fireEvent.click(screen.getByRole('button', { name: /Lambda/ }));
    expect(
      screen.getByText('in-stock regions at last refresh — no counts published')
    ).toBeInTheDocument();
    // catalog tier
    fireEvent.click(screen.getByRole('button', { name: /AWS/ }));
    expect(
      screen.getByText('catalog coverage — on-demand clouds publish no stock')
    ).toBeInTheDocument();
  });

  it('renders a present provider with an empty GPU list as the no-GPU note', () => {
    renderStock({
      providers: {
        vast: { present: true, updated_at: threeHoursAgo, gpus: [] },
      },
    });
    expect(
      screen.getByText('No GPUs in the current catalog')
    ).toBeInTheDocument();
  });

  it('shows the empty state when no provider has a catalog yet', () => {
    renderStock({ providers: {} });
    expect(screen.getByText('No provider catalogs yet')).toBeInTheDocument();
  });

  it('renders a skeleton while loading and no provider names', () => {
    const { container } = renderStock({ loading: true, providers: {} });
    expect(container.textContent).toContain('GPU Stock by Region');
    expect(container.textContent).not.toContain('AWS');
    expect(container.querySelectorAll('.animate-pulse')).toHaveLength(1);
  });

  it('renders the error in-section instead of crashing', () => {
    const { container } = renderStock({
      error: 'Failed to get provider GPU offerings with status 502',
    });
    expect(container.textContent).toContain(
      'Failed to load GPU stock by region: Failed to get provider GPU offerings with status 502'
    );
  });
});
