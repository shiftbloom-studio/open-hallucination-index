import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import VerifyAIOutputForm from '@/components/dashboard/verify-ai-output-form';
import { server } from '@/test/mocks/server';
import { http, HttpResponse } from 'msw';

// Mock sonner toast
vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

// Mock the API client
vi.mock('@/lib/api', () => ({
  createApiClient: vi.fn(() => ({
    verifyText: vi.fn().mockResolvedValue({
      id: 'test-id',
      trust_score: { score: 0.85 },
      summary: 'Analyzed 3 claim(s): 2 supported, 1 refuted. Trust level: high.',
      claims: [
        {
          id: 'claim-1',
          text: 'The Eiffel Tower is in Paris',
          status: 'supported',
          confidence: 0.95,
          reasoning: 'Verified against knowledge base',
        },
        {
          id: 'claim-2',
          text: 'It was built in 1889',
          status: 'supported',
          confidence: 0.90,
          reasoning: 'Verified against knowledge base',
        },
        {
          id: 'claim-3',
          text: 'It is made of wood',
          status: 'refuted',
          confidence: 0.99,
          reasoning: 'The Eiffel Tower is made of iron',
        },
      ],
      processing_time_ms: 150,
      cached: false,
    }),
    getHealth: vi.fn().mockResolvedValue({
      status: 'healthy',
    }),
  })),
}));

describe('VerifyAIOutputForm', () => {
  const defaultProps = {
    userTokens: 10,
    onTokensUpdated: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  it('should render the form with text area and verify button', () => {
    render(<VerifyAIOutputForm {...defaultProps} />);

    expect(screen.getByText('Verify AI Output')).toBeInTheDocument();
    expect(screen.getByLabelText(/AI Output Text/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /verify text/i })).toBeInTheDocument();
  });

  it('should display character count and token estimate', async () => {
    const user = userEvent.setup();
    render(<VerifyAIOutputForm {...defaultProps} />);

    const textArea = screen.getByPlaceholderText(/paste your ai-generated text/i);
    const inputText = 'This is a test text that is about 50 characters long';
    await user.type(textArea, inputText);

    // Check that character count is displayed
    expect(screen.getByText(new RegExp(`^${inputText.length}\\s+characters$`, 'i'))).toBeInTheDocument();
    // Check that token estimate is displayed
    expect(screen.getByText(/token\w*\s+needed/i)).toBeInTheDocument();
  });

  it('should show current token balance', () => {
    render(<VerifyAIOutputForm {...defaultProps} userTokens={15} />);

    expect(screen.getByText(/15 token/i)).toBeInTheDocument();
  });

  it('should disable verify button when no text is entered', () => {
    render(<VerifyAIOutputForm {...defaultProps} />);

    const verifyButton = screen.getByRole('button', { name: /verify text/i });
    expect(verifyButton).toBeDisabled();
  });

  it('should enable verify button when text is entered and tokens are sufficient', async () => {
    const user = userEvent.setup();
    render(<VerifyAIOutputForm {...defaultProps} />);

    const textArea = screen.getByPlaceholderText(/paste your ai-generated text/i);
    await user.type(textArea, 'Some AI generated text to verify');

    const verifyButton = screen.getByRole('button', { name: /verify text/i });
    expect(verifyButton).not.toBeDisabled();
  });

  it('should show warning when insufficient tokens', async () => {
    const user = userEvent.setup();
    render(<VerifyAIOutputForm {...defaultProps} userTokens={0} />);

    const textArea = screen.getByPlaceholderText(/paste your ai-generated text/i);
    await user.type(textArea, 'Some text to verify');

    expect(screen.getByText(/you need\s+\d+\s+more token/i)).toBeInTheDocument();
  });

  it('should calculate tokens needed based on text length', async () => {
    const user = userEvent.setup();
    render(<VerifyAIOutputForm {...defaultProps} userTokens={5} />);

    const textArea = screen.getByPlaceholderText(/paste your ai-generated text/i);
    
    // Type text that's less than 1000 chars - should need 1 token
    await user.type(textArea, 'Short text');
    expect(screen.getByText(/1 token.*needed/i)).toBeInTheDocument();
  });

  it('should show context field as optional', () => {
    render(<VerifyAIOutputForm {...defaultProps} />);

    expect(screen.getByLabelText(/context.*optional/i)).toBeInTheDocument();
  });

  it('should call token deduction API and OHI API on verify', async () => {
    const user = userEvent.setup();
    const onTokensUpdated = vi.fn();

    server.use(
      http.post('/api/tokens', () => {
        return HttpResponse.json({
          success: true,
          tokensDeducted: 1,
          tokensRemaining: 9,
        });
      })
    );

    render(<VerifyAIOutputForm {...defaultProps} onTokensUpdated={onTokensUpdated} />);

    const textArea = screen.getByPlaceholderText(/paste your ai-generated text/i);
    await user.type(textArea, 'The Eiffel Tower is in Paris and was built in 1889.');

    const verifyButton = screen.getByRole('button', { name: /verify text/i });
    await user.click(verifyButton);

    await waitFor(() => {
      expect(onTokensUpdated).toHaveBeenCalledWith(9);
    });
  });

  it('should display verification results after successful verification', async () => {
    const user = userEvent.setup();

    server.use(
      http.post('/api/tokens', () => {
        return HttpResponse.json({
          success: true,
          tokensDeducted: 1,
          tokensRemaining: 9,
        });
      })
    );

    render(<VerifyAIOutputForm {...defaultProps} />);

    const textArea = screen.getByPlaceholderText(/paste your ai-generated text/i);
    await user.type(textArea, 'The Eiffel Tower is in Paris.');

    const verifyButton = screen.getByRole('button', { name: /verify text/i });
    await user.click(verifyButton);

    await waitFor(() => {
      expect(screen.getByText('Verification Results')).toBeInTheDocument();
    });

    // Check trust score is displayed
    expect(screen.getByText(/85(\.0)?%/)).toBeInTheDocument();
    
    // Check claims are displayed
    expect(screen.getByText('Claims Analyzed')).toBeInTheDocument();
  });
});
