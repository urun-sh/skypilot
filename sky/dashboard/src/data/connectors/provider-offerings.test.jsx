// getProviderOfferings delegates to apiClient.fetch — the shared
// scheduled-request helper that owns the POST + /api/get two-step and the
// error-detail extraction — so its tests mock apiClient.fetch directly and
// pin the request shape and the response normalization.

jest.mock('@/data/connectors/client', () => ({
  __esModule: true,
  apiClient: { fetch: jest.fn() },
}));

import { apiClient } from '@/data/connectors/client';
import { getProviderOfferings } from '@/data/connectors/provider-offerings';

const providers = {
  vast: {
    present: true,
    updated_at: 1760000000.0,
    gpus: [
      {
        name: 'RTX6000Ada',
        counts: [1, 2, 4],
        price: 0.35,
        spot_price: 0.18,
        regions: ['us-east', 'us-west'],
      },
    ],
  },
  ornn: { present: false, updated_at: null, gpus: null },
};

describe('getProviderOfferings', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('fetches /provider_gpu_offerings with the default body', async () => {
    apiClient.fetch.mockResolvedValue({ providers });

    const data = await getProviderOfferings();

    expect(apiClient.fetch).toHaveBeenCalledWith('/provider_gpu_offerings', {
      clouds: null,
    });
    expect(data).toEqual({ providers });
  });

  it('normalizes a response without a providers object', async () => {
    apiClient.fetch.mockResolvedValue({});

    const data = await getProviderOfferings();

    expect(data).toEqual({ providers: {} });
  });

  it('normalizes an empty return_value', async () => {
    apiClient.fetch.mockResolvedValue([]);

    const data = await getProviderOfferings();

    expect(data).toEqual({ providers: {} });
  });

  it('propagates fetch errors', async () => {
    apiClient.fetch.mockRejectedValue(
      new Error('Initial API request failed with status 502')
    );

    await expect(getProviderOfferings()).rejects.toThrow(/502/);
  });
});
