import { apiClient } from '@/data/connectors/client';

/**
 * What every GPU provider's catalog CURRENTLY OFFERS — GPU types, requestable
 * counts, prices, regions — as read by the server from the SkyPilot catalog
 * CSVs (`~/.sky/catalogs/v8/<cloud>/vms.csv`) that the skypilot-controller
 * runner refreshes every few hours. Independent of live clusters: a provider
 * with zero clusters still reports its offerings.
 *
 * Uses the same scheduled-request pattern as every other endpoint: POST
 * `/provider_gpu_offerings` answers with a request id in the
 * `X-Skypilot-Request-ID` header, then GET `/api/get?request_id=...` answers
 * with the result under a JSON-encoded `return_value`.
 *
 * `clouds: null` asks the server for its default provider list (the same set
 * the dashboard keeps in `PROVIDER_OFFERINGS_CLOUDS`), so the section can
 * render every default provider even when the response omits one.
 *
 * Response contract:
 * {
 *   providers: {
 *     <cloud>: {
 *       present: boolean,        // catalog CSV exists on disk (never fetched)
 *       updated_at: float|null,  // CSV mtime, unix seconds
 *       gpus: [
 *         {
 *           name: string,
 *           counts: number[],    // distinct accelerator counts offered
 *           price: number|null,  // cheapest on-demand $/hr
 *           spot_price: number|null, // cheapest spot $/hr
 *           regions: string[],
 *         },
 *       ] | null,
 *     },
 *   },
 * }
 */
export async function getProviderOfferings() {
  const response = await apiClient.post(`/provider_gpu_offerings`, {
    clouds: null,
  });
  if (!response.ok) {
    const msg = `Failed to get provider GPU offerings with status ${response.status}`;
    throw new Error(msg);
  }
  const id = response.headers.get('X-Skypilot-Request-ID');
  // Handle empty request ID
  if (!id) {
    const msg = 'No request ID received from server for provider GPU offerings';
    throw new Error(msg);
  }
  const fetchedData = await apiClient.get(`/api/get?request_id=${id}`);
  if (!fetchedData.ok) {
    const msg = `Failed to get provider GPU offerings result with status ${fetchedData.status}`;
    throw new Error(msg);
  }
  const data = await fetchedData.json();
  const parsed = data.return_value ? JSON.parse(data.return_value) : null;
  // Normalize so callers always get the contract shape even if the server
  // ever answers with just the providers map or nothing at all.
  return { providers: parsed?.providers || {} };
}
