/**
 * Infrastructure utility functions for context key generation
 */

/**
 * Builds a context stats key for use in contextStats mapping.
 * This key is used consistently across the application to identify
 * infrastructure contexts (Kubernetes, SSH Node Pools, Slurm clusters).
 *
 * @param {string} contextName - The context name (e.g., 'my-context', 'ssh-pool1', 'slurm-cluster')
 * @param {Object} options - Options for determining the context type
 * @param {boolean} [options.isSSH] - Whether this is an SSH Node Pool context
 * @param {boolean} [options.isSlurm] - Whether this is a Slurm cluster context
 * @param {string} [options.cloud] - Cloud type ('Kubernetes', 'SSH', 'slurm', or 'Slurm')
 * @returns {string} - The context stats key (e.g., 'kubernetes/my-context', 'ssh/pool1', 'slurm/cluster')
 */
export function buildContextStatsKey(contextName, options = {}) {
  if (!contextName) {
    return null;
  }

  const { isSSH, isSlurm, cloud } = options;

  // Determine context type from options or infer from context name
  let contextType = null;
  if (isSSH || cloud === 'SSH') {
    contextType = 'ssh';
  } else if (isSlurm || cloud?.toLowerCase() === 'slurm') {
    contextType = 'slurm';
  } else if (cloud === 'Kubernetes') {
    contextType = 'kubernetes';
  } else if (contextName.startsWith('ssh-')) {
    // Infer from context name if no explicit type provided
    contextType = 'ssh';
  } else {
    // Default to Kubernetes for backward compatibility
    contextType = 'kubernetes';
  }

  // Process context name based on type
  let processedName = contextName;
  if (contextType === 'ssh') {
    // Remove 'ssh-' prefix if present
    processedName = contextName.replace(/^ssh-/, '');
  }

  return `${contextType}/${processedName}`;
}

/**
 * Builds a context stats key from a cloud/region pair (used in jobs and clusters).
 * This is a convenience wrapper around buildContextStatsKey for the common
 * pattern where we have a cloud type and region/context name.
 *
 * @param {string} cloud - Cloud type ('Kubernetes', 'SSH', 'slurm', or 'Slurm')
 * @param {string} region - Region/context name (may include 'ssh-' prefix for SSH)
 * @returns {string|null} - The context stats key or null if invalid
 */
export function buildContextStatsKeyFromCloud(cloud, region) {
  if (!cloud || !region) {
    return null;
  }
  return buildContextStatsKey(region, { cloud });
}

/**
 * Counts records per `cloud` field, keyed by lowercase cloud name so joins
 * with provider display names are case-insensitive. API `cloud` values are
 * the provider's _REPR (e.g. 'GCP', 'Vast', 'Quantacloud'), which must land
 * on the Cloud panel row named the same way.
 *
 * @param {Array<Object>} records - Records with a `cloud` field
 * @returns {Object} Map of lowercase cloud name to record count
 */
export function countByCloud(records) {
  const counts = {};
  (records || []).forEach((record) => {
    if (record && record.cloud) {
      const key = record.cloud.toLowerCase();
      counts[key] = (counts[key] || 0) + 1;
    }
  });
  return counts;
}

/**
 * Filters Cloud panel rows for the selected workspace. A cloud stays
 * visible when the workspace sky check reports it enabled OR it has live
 * clusters/jobs, so a provider actually running work is never hidden.
 *
 * @param {Array<Object>} rows - Cloud rows ({name, enabled, clusters, jobs})
 * @param {string[]|null} workspaceEnabledClouds - Lowercase enabled clouds
 *   for the workspace, or null to skip filtering (data still loading).
 * @param {Object} cloudClusterCounts - Lowercase-keyed cluster counts
 * @param {Object} cloudJobCounts - Lowercase-keyed job counts
 * @returns {Array<Object>} Rows visible for the workspace
 */
export function visibleCloudRows(
  rows,
  workspaceEnabledClouds,
  cloudClusterCounts = {},
  cloudJobCounts = {}
) {
  if (!rows || rows.length === 0) return [];
  if (workspaceEnabledClouds === null) return rows;
  return rows.filter((cloud) => {
    const key = (cloud.name || '').toLowerCase();
    const hasLiveResources =
      (cloud.clusters || 0) > 0 ||
      (cloud.jobs || 0) > 0 ||
      (cloudClusterCounts[key] || 0) > 0 ||
      (cloudJobCounts[key] || 0) > 0;
    return hasLiveResources || workspaceEnabledClouds.includes(key);
  });
}

/**
 * Records belonging to the selected workspace. Each record's `workspace`
 * field decides membership; 'all' returns every record. Used to scope Cloud
 * panel counts so activity in another workspace does not leak in.
 *
 * @param {Array<Object>} records - Records with a `workspace` field
 * @param {string} selectedWorkspace - Workspace name or 'all'
 * @returns {Array<Object>} Records visible for the workspace
 */
export function recordsInWorkspace(records, selectedWorkspace) {
  if (!records) {
    return [];
  }
  if (selectedWorkspace === 'all' || !selectedWorkspace) {
    return records;
  }
  return records.filter((record) => record.workspace === selectedWorkspace);
}
