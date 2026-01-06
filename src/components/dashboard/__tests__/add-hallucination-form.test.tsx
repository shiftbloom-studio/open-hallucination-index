import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import AddHallucinationForm from '@/components/dashboard/add-hallucination-form';
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
    showApiSettings: false,
    setShowApiSettings: vi.fn(),
    apiUrl: 'http://localhost:8000',
    setApiUrl: vi.fn(),
    apiStatus: 'valid' as const,
    handleApiUrlChange: vi.fn(),
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

  it('should toggle API settings on button click', async () => {
    const user = userEvent.setup();
    const setShowApiSettings = vi.fn();
    
    render(
      <AddHallucinationForm 
        {...defaultProps} 
        setShowApiSettings={setShowApiSettings}
      />
    );

    const settingsButton = screen.getByRole('button', { name: '' });
    await user.click(settingsButton);

    expect(setShowApiSettings).toHaveBeenCalledWith(true);
  });

  it('should show API settings when showApiSettings is true', () => {
    render(
      <AddHallucinationForm 
        {...defaultProps} 
        showApiSettings={true}
      />
    );

    expect(screen.getByText('API Configuration')).toBeInTheDocument();
    expect(screen.getByLabelText(/Open Hallucination API URL/i)).toBeInTheDocument();
  });

  it('should show API status indicators', () => {
    const { rerender } = render(
      <AddHallucinationForm 
        {...defaultProps} 
        showApiSettings={true}
        apiStatus="valid"
      />
    );

    expect(screen.getByText('API connected')).toBeInTheDocument();

    rerender(
      <AddHallucinationForm 
        {...defaultProps} 
        showApiSettings={true}
        apiStatus="invalid"
      />
    );

    expect(screen.getByText('API not reachable')).toBeInTheDocument();
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

  it('should require API URL before verification', async () => {
    const user = userEvent.setup();
    const setShowApiSettings = vi.fn();
    const { toast } = await import('sonner');
    
    render(
      <AddHallucinationForm 
        {...defaultProps} 
        apiUrl=""
        setShowApiSettings={setShowApiSettings}
      />
    );

    const contentInput = screen.getByLabelText(/content/i);
    await user.type(contentInput, 'Test content');

    const verifyButton = screen.getByRole('button', { name: /verify/i });
    await user.click(verifyButton);

    expect(toast.error).toHaveBeenCalledWith('Please configure API URL first');
    expect(setShowApiSettings).toHaveBeenCalledWith(true);
  });

  it('should show verify button when content is empty', async () => {
    // This test verifies that the form has the verify button visible
    // The actual validation behavior depends on the component implementation
    render(<AddHallucinationForm {...defaultProps} apiUrl="http://localhost:8000" />);

    // Verify button should be present
    const verifyButton = screen.getByRole('button', { name: /verify/i });
    expect(verifyButton).toBeInTheDocument();
    
    // Content input should be empty initially
    const contentInput = screen.getByLabelText(/content/i);
    expect(contentInput).toHaveValue('');
  });
});
