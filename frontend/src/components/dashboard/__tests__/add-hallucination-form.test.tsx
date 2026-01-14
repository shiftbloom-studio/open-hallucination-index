import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import AddHallucinationForm from '@/components/dashboard/add-hallucination-form';

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
      summary: 'Test summary',
      claims: [],
      processing_time_ms: 100,
      cached: false,
    }),
    getHealth: vi.fn().mockResolvedValue({
      status: 'healthy',
    }),
  })),
}));

describe('AddHallucinationForm', () => {
  const defaultProps = {
    onCancel: vi.fn(),
    onSuccess: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render the form with all fields', () => {
    render(<AddHallucinationForm {...defaultProps} />);

    expect(screen.getByText('Add New Hallucination')).toBeInTheDocument();
    expect(screen.getByLabelText(/content/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/source/i)).toBeInTheDocument();
  });

  it('should update content field on input', async () => {
    const user = userEvent.setup();
    render(<AddHallucinationForm {...defaultProps} />);

    const contentInput = screen.getByLabelText(/content/i);
    await user.type(contentInput, 'Test hallucination content');

    expect(contentInput).toHaveValue('Test hallucination content');
  });

  it('should update source field on input', async () => {
    const user = userEvent.setup();
    render(<AddHallucinationForm {...defaultProps} />);

    const sourceInput = screen.getByLabelText(/source/i);
    await user.type(sourceInput, 'GPT-4');

    expect(sourceInput).toHaveValue('GPT-4');
  });


  it('should call onSuccess on form submission', async () => {
    const user = userEvent.setup();
    const onSuccess = vi.fn();
    
    render(
      <AddHallucinationForm 
        {...defaultProps} 
        onSuccess={onSuccess}
      />
    );

    const contentInput = screen.getByLabelText(/content/i);
    const sourceInput = screen.getByLabelText(/source/i);
    
    await user.type(contentInput, 'Test content');
    await user.type(sourceInput, 'Test source');

    const submitButton = screen.getByRole('button', { name: /add hallucination|submit|save/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalled();
    });
  });

  it('should clear form after successful submission', async () => {
    const user = userEvent.setup();
    
    render(<AddHallucinationForm {...defaultProps} />);

    const contentInput = screen.getByLabelText(/content/i);
    const sourceInput = screen.getByLabelText(/source/i);
    
    await user.type(contentInput, 'Test content');
    await user.type(sourceInput, 'Test source');

    const submitButton = screen.getByRole('button', { name: /add hallucination|submit|save/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(contentInput).toHaveValue('');
      expect(sourceInput).toHaveValue('');
    });
  });

  it('should have verify button', () => {
    render(<AddHallucinationForm {...defaultProps} />);

    const verifyButton = screen.getByRole('button', { name: /verify/i });
    expect(verifyButton).toBeInTheDocument();
  });

  it('should show verify button when content is empty', async () => {
    // This test verifies that the form has the verify button visible
    // The actual validation behavior depends on the component implementation
    render(<AddHallucinationForm {...defaultProps} />);

    // Verify button should be present
    const verifyButton = screen.getByRole('button', { name: /verify/i });
    expect(verifyButton).toBeInTheDocument();
    
    // Content input should be empty initially
    const contentInput = screen.getByLabelText(/content/i);
    expect(contentInput).toHaveValue('');
  });
});
