import { apiClient } from '@/data/connectors/client';

/**
 * What every GPU provider's catalog CURRENTLY OFFERS — GPU types, requestable
 * counts, prices, regions — as read by the server from the SkyPilot catalog
 * CSVs (`~/.sky/catalogs/v8/<cloud>/vms.csv`) that the skypilot-controller
 * runner refreshes every few hours. Independent of live clusters: a provider
 * with zero clusters still reports its offerings.
 *
 * Uses `apiClient.fetch` — the shared scheduled-request helper (POST answers
 * with a request id, GET `/api/get?request_id=...` answers with the result
 * under a JSON-encoded `return_value`) — so server-side error details are
 * preserved the same way every other connector reports them.
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
  const parsed = await apiClient.fetch('/provider_gpu_offerings', {
    clouds: null,
  });
  // Normalize so callers always get the contract shape even if the server
  // ever answers with just the providers map or nothing at all.
  return { providers: parsed?.providers || {} };
}
