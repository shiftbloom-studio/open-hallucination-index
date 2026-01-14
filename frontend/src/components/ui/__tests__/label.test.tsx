import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/test-utils';
import { Label } from '@/components/ui/label';

describe('Label', () => {
  it('should render with default styles', () => {
    render(<Label>Test Label</Label>);
    
    const label = screen.getByText('Test Label');
    expect(label).toBeInTheDocument();
    expect(label).toHaveClass('text-sm', 'font-medium', 'leading-none');
  });

  it('should render text content', () => {
    render(<Label>Email Address</Label>);
    
    expect(screen.getByText('Email Address')).toBeInTheDocument();
  });

  it('should apply custom className', () => {
    render(<Label className="custom-label">Custom</Label>);
    
    expect(screen.getByText('Custom')).toHaveClass('custom-label');
  });

  it('should forward refs correctly', () => {
    const ref = vi.fn();
    render(<Label ref={ref}>Ref Label</Label>);
    
    expect(ref).toHaveBeenCalled();
  });

  it('should associate with input using htmlFor', () => {
    render(
      <>
        <Label htmlFor="test-input">Test Label</Label>
        <input id="test-input" />
      </>
    );
    
    const label = screen.getByText('Test Label');
    expect(label).toHaveAttribute('for', 'test-input');
  });

  it('should pass through additional props', () => {
    render(<Label data-testid="test-label" id="my-label">Props Label</Label>);
    
    const label = screen.getByTestId('test-label');
    expect(label).toHaveAttribute('id', 'my-label');
  });

  it('should have correct display name', () => {
    expect(Label.displayName).toBe('Label');
  });

  it('should work with form inputs', () => {
    render(
      <div>
        <Label htmlFor="email">Email</Label>
        <input id="email" type="email" />
      </div>
    );
    
    expect(screen.getByText('Email')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveAttribute('id', 'email');
  });

  it('should have peer-disabled styles', () => {
    render(<Label>Disabled Style Label</Label>);
    
    const label = screen.getByText('Disabled Style Label');
    expect(label).toHaveClass('peer-disabled:cursor-not-allowed');
    expect(label).toHaveClass('peer-disabled:opacity-70');
  });

  it('should render children correctly', () => {
    render(
      <Label>
        <span>Nested</span> Content
      </Label>
    );
    
    expect(screen.getByText('Nested')).toBeInTheDocument();
    expect(screen.getByText(/Content/)).toBeInTheDocument();
  });
});
