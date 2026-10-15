// getProviderOfferings follows the same two-step scheduled pattern as every
// other dashboard connector, so its tests mock apiClient the same way: POST
// answers with a request id in a header, then /api/get answers with the
// JSON-encoded return_value.

jest.mock('@/data/connectors/client', () => ({
  __esModule: true,
  apiClient: { post: jest.fn(), get: jest.fn() },
}));

import { apiClient } from '@/data/connectors/client';
import { getProviderOfferings } from '@/data/connectors/provider-offerings';

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

  it('POSTs /provider_gpu_offerings with the default body, then polls /api/get', async () => {
    apiClient.post.mockResolvedValue(scheduled('req-42'));
    apiClient.get.mockResolvedValue(result({ providers }));

    const data = await getProviderOfferings();

    expect(apiClient.post).toHaveBeenCalledWith('/provider_gpu_offerings', {
      clouds: null,
    });
    expect(apiClient.get).toHaveBeenCalledWith('/api/get?request_id=req-42');
    expect(data).toEqual({ providers });
  });

  it('normalizes a response without a providers object', async () => {
    apiClient.post.mockResolvedValue(scheduled('req-43'));
    apiClient.get.mockResolvedValue(result(null));

    const data = await getProviderOfferings();

    expect(data).toEqual({ providers: {} });
  });

  it('normalizes an empty return_value', async () => {
    apiClient.post.mockResolvedValue(scheduled('req-44'));
    apiClient.get.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({}),
    });

    const data = await getProviderOfferings();

    expect(data).toEqual({ providers: {} });
  });

  it('surfaces a failed scheduling POST as an error', async () => {
    apiClient.post.mockResolvedValue({
      ok: false,
      status: 502,
      headers: { get: () => null },
    });

    await expect(getProviderOfferings()).rejects.toThrow(/502/);
  });

  it('surfaces a missing request id as an error', async () => {
    apiClient.post.mockResolvedValue(scheduled(null));

    await expect(getProviderOfferings()).rejects.toThrow(/No request ID/);
  });

  it('surfaces a failed /api/get poll as an error', async () => {
    apiClient.post.mockResolvedValue(scheduled('req-45'));
    apiClient.get.mockResolvedValue({ ok: false, status: 500 });

    await expect(getProviderOfferings()).rejects.toThrow(/500/);
  });
});
