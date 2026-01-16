import { describe, it, expect } from 'vitest';
import { render, screen } from '@/test/test-utils';
import userEvent from '@testing-library/user-event';
import CitationTraceViewer from '@/components/dashboard/citation-trace-viewer';
import { CitationTrace } from '@/lib/api';

describe('CitationTraceViewer', () => {
  const mockTrace: CitationTrace = {
    claim_id: 'test-claim-id',
    status: 'supported',
    reasoning: 'This claim is supported by multiple reliable sources.',
    supporting_evidence: [
      {
        id: 'evidence-1',
        source: 'wikipedia',
        content: 'Wikipedia confirms this fact.',
        similarity_score: 0.95,
        classification_confidence: 0.9,
        retrieved_at: '2026-01-16T12:00:00Z',
        source_uri: 'https://en.wikipedia.org/wiki/Example',
      },
      {
        id: 'evidence-2',
        source: 'graph_exact',
        content: 'Knowledge graph has an exact match.',
        similarity_score: 1.0,
        classification_confidence: 0.95,
        retrieved_at: '2026-01-16T12:00:00Z',
      },
    ],
    refuting_evidence: [],
    confidence: 0.92,
    verification_strategy: 'adaptive',
  };

  const mockTraceWithRefuting: CitationTrace = {
    ...mockTrace,
    status: 'refuted',
    reasoning: 'This claim is contradicted by evidence.',
    supporting_evidence: [],
    refuting_evidence: [
      {
        id: 'evidence-3',
        source: 'pubmed',
        content: 'Scientific study shows the opposite.',
        similarity_score: 0.88,
        classification_confidence: 0.85,
        retrieved_at: '2026-01-16T12:00:00Z',
        source_uri: 'https://pubmed.ncbi.nlm.nih.gov/12345',
      },
    ],
  };

  it('should render evidence trail with supporting evidence', () => {
    render(
      <CitationTraceViewer 
        trace={mockTrace} 
        claimText="Test claim text" 
      />
    );

    expect(screen.getByText('Evidence Trail')).toBeInTheDocument();
    expect(screen.getByText(/2 sources? analyzed/i)).toBeInTheDocument();
    expect(screen.getByText('This claim is supported by multiple reliable sources.')).toBeInTheDocument();
    expect(screen.getByText('Supporting Evidence (2)')).toBeInTheDocument();
  });

  it('should render evidence cards with correct content', () => {
    render(
      <CitationTraceViewer 
        trace={mockTrace} 
        claimText="Test claim" 
      />
    );

    expect(screen.getByText('Wikipedia confirms this fact.')).toBeInTheDocument();
    expect(screen.getByText('Knowledge graph has an exact match.')).toBeInTheDocument();
  });

  it('should render refuting evidence', () => {
    render(
      <CitationTraceViewer 
        trace={mockTraceWithRefuting} 
        claimText="Test claim" 
      />
    );

    expect(screen.getByText('Refuting Evidence (1)')).toBeInTheDocument();
    expect(screen.getByText('Scientific study shows the opposite.')).toBeInTheDocument();
  });

  it('should display source links when available', () => {
    render(
      <CitationTraceViewer 
        trace={mockTrace} 
        claimText="Test claim" 
      />
    );

    const links = screen.getAllByText('View source');
    expect(links.length).toBeGreaterThan(0);
  });

  it('should display match scores', () => {
    render(
      <CitationTraceViewer 
        trace={mockTrace} 
        claimText="Test claim" 
      />
    );

    expect(screen.getByText(/Match: 95%/i)).toBeInTheDocument();
    expect(screen.getByText(/Match: 100%/i)).toBeInTheDocument();
  });

  it('should be collapsible', async () => {
    const user = userEvent.setup();
    render(
      <CitationTraceViewer 
        trace={mockTrace} 
        claimText="Test claim" 
      />
    );

    const hideButton = screen.getByRole('button', { name: /hide/i });
    await user.click(hideButton);

    // Evidence should be hidden after clicking
    expect(screen.queryByText('Wikipedia confirms this fact.')).not.toBeInTheDocument();
  });

  it('should show message when no evidence is found', () => {
    const emptyTrace: CitationTrace = {
      ...mockTrace,
      supporting_evidence: [],
      refuting_evidence: [],
    };

    render(
      <CitationTraceViewer 
        trace={emptyTrace} 
        claimText="Test claim" 
      />
    );

    expect(screen.getByText('No evidence found for this claim')).toBeInTheDocument();
  });
});
