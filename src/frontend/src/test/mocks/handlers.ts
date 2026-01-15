import { http, HttpResponse } from 'msw';

// Mock API responses
export const mockHealthResponse = {
  status: 'healthy',
  timestamp: '2026-01-06T12:00:00Z',
  version: '1.0.0',
  environment: 'test',
  checks: {
    database: true,
    cache: true,
  },
};

export const mockVerifyResponse = {
  id: '550e8400-e29b-41d4-a716-446655440000',
  trust_score: {
    score: 0.85,
    confidence: 0.9,
  },
  summary: 'The text contains 2 claims, 1 verified and 1 unverified.',
  claims: [
    {
      id: '550e8400-e29b-41d4-a716-446655440001',
      text: 'The sky is blue.',
      status: 'verified',
      confidence: 0.95,
      reasoning: 'This is a well-known scientific fact.',
    },
    {
      id: '550e8400-e29b-41d4-a716-446655440002',
      text: 'Water is wet.',
      status: 'verified',
      confidence: 0.92,
      reasoning: 'Common knowledge confirmed by physics.',
    },
  ],
  processing_time_ms: 150,
  cached: false,
};

export const mockBatchVerifyResponse = {
  results: [mockVerifyResponse],
  total_processing_time_ms: 300,
};

export const handlers = [
  // Health endpoint
  http.get('*/health/live', () => {
    return HttpResponse.json(mockHealthResponse);
  }),

  // Verify endpoint
  http.post('*/api/v1/verify', async ({ request }) => {
    const body = await request.json() as { text?: string };
    
    if (!body.text) {
      return new HttpResponse('Missing text field', { status: 400 });
    }

    return HttpResponse.json(mockVerifyResponse);
  }),

  // Batch verify endpoint
  http.post('*/api/v1/verify/batch', async ({ request }) => {
    const body = await request.json() as { texts?: string[] };
    
    if (!body.texts || !Array.isArray(body.texts)) {
      return new HttpResponse('Missing texts field', { status: 400 });
    }

    return HttpResponse.json(mockBatchVerifyResponse);
  }),

  // Checkout endpoint
  http.post('/api/checkout', async ({ request }) => {
    const body = await request.json() as { packageId?: string };
    
    if (!body.packageId) {
      return new HttpResponse('Missing packageId', { status: 400 });
    }

    if (!['10', '100', '500'].includes(body.packageId)) {
      return new HttpResponse('Invalid packageId', { status: 400 });
    }

    return HttpResponse.json({
      url: 'https://checkout.stripe.com/test-session',
    });
  }),

  // Auth callback
  http.get('/api/auth/callback', () => {
    return HttpResponse.redirect('/dashboard', 302);
  }),
];

// Error handlers for testing error scenarios
export const errorHandlers = {
  healthCheckFailed: http.get('*/health/live', () => {
    return new HttpResponse('Service unavailable', { status: 503 });
  }),
  
  verifyFailed: http.post('*/api/v1/verify', () => {
    return new HttpResponse('Internal server error', { status: 500 });
  }),
  
  verifyTimeout: http.post('*/api/v1/verify', async () => {
    await new Promise((resolve) => setTimeout(resolve, 30000));
    return HttpResponse.json(mockVerifyResponse);
  }),
};
