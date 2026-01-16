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

  it('should render evidence trail with supporting evidence', async () => {
    const user = userEvent.setup();
    render(
      <CitationTraceViewer 
        trace={mockTrace} 
        claimText="Test claim text" 
      />
    );

    expect(screen.getByText('Evidence Trail')).toBeInTheDocument();
    expect(screen.getByText(/2 sources? analyzed/i)).toBeInTheDocument();
    
    // Click expand button to show evidence
    const expandButton = screen.getByRole('button', { name: '' });
    await user.click(expandButton);
    
    // Switch to list view to see evidence cards
    const listButton = screen.getByRole('button', { name: 'List' });
    await user.click(listButton);
    
    expect(screen.getByText('This claim is supported by multiple reliable sources.')).toBeInTheDocument();
    expect(screen.getByText('Supporting Evidence (2)')).toBeInTheDocument();
  });

  it('should render evidence cards with correct content', async () => {
    const user = userEvent.setup();
    render(
      <CitationTraceViewer 
        trace={mockTrace} 
        claimText="Test claim" 
      />
    );

    // Click expand button to show evidence
    const expandButton = screen.getByRole('button', { name: '' });
    await user.click(expandButton);
    
    // Switch to list view to see evidence cards
    const listButton = screen.getByRole('button', { name: 'List' });
    await user.click(listButton);

    expect(screen.getByText('Wikipedia confirms this fact.')).toBeInTheDocument();
    expect(screen.getByText('Knowledge graph has an exact match.')).toBeInTheDocument();
  });

  it('should render refuting evidence', async () => {
    const user = userEvent.setup();
    render(
      <CitationTraceViewer 
        trace={mockTraceWithRefuting} 
        claimText="Test claim" 
      />
    );

    // Click expand button to show evidence
    const expandButton = screen.getByRole('button', { name: '' });
    await user.click(expandButton);
    
    // Switch to list view
    const listButton = screen.getByRole('button', { name: 'List' });
    await user.click(listButton);

    expect(screen.getByText('Refuting Evidence (1)')).toBeInTheDocument();
    expect(screen.getByText('Scientific study shows the opposite.')).toBeInTheDocument();
  });

  it('should display source links when available', async () => {
    const user = userEvent.setup();
    render(
      <CitationTraceViewer 
        trace={mockTrace} 
        claimText="Test claim" 
      />
    );

    // Click expand button to show evidence
    const expandButton = screen.getByRole('button', { name: '' });
    await user.click(expandButton);
    
    // Switch to list view
    const listButton = screen.getByRole('button', { name: 'List' });
    await user.click(listButton);

    const links = screen.getAllByText('View source');
    expect(links.length).toBeGreaterThan(0);
  });

  it('should display match scores', async () => {
    const user = userEvent.setup();
    render(
      <CitationTraceViewer 
        trace={mockTrace} 
        claimText="Test claim" 
      />
    );

    // Click expand button to show evidence
    const expandButton = screen.getByRole('button', { name: '' });
    await user.click(expandButton);
    
    // Switch to list view
    const listButton = screen.getByRole('button', { name: 'List' });
    await user.click(listButton);

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

    // Initially collapsed - content should not be visible
    expect(screen.queryByText('Wikipedia confirms this fact.')).not.toBeInTheDocument();
    
    // Click expand button
    const expandButton = screen.getByRole('button', { name: '' });
    await user.click(expandButton);
    
    // Switch to list view to see content
    const listButton = screen.getByRole('button', { name: 'List' });
    await user.click(listButton);
    
    // Evidence should now be visible
    expect(screen.getByText('Wikipedia confirms this fact.')).toBeInTheDocument();
    
    // Click collapse button
    await user.click(expandButton);
    
    // Evidence should be hidden again
    expect(screen.queryByText('Wikipedia confirms this fact.')).not.toBeInTheDocument();
  });

  it('should show message when no evidence is found', async () => {
    const user = userEvent.setup();
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

    // Click expand button to show evidence section
    const expandButton = screen.getByRole('button', { name: '' });
    await user.click(expandButton);
    
    // Switch to list view
    const listButton = screen.getByRole('button', { name: 'List' });
    await user.click(listButton);

    expect(screen.getByText('No evidence found for this claim')).toBeInTheDocument();
  });
});
