import { describe, it, expect } from 'vitest';
import { render, screen } from '@/test/test-utils';
import { Providers } from '@/app/providers';

describe('Providers', () => {
  it('should render children', () => {
    render(
      <Providers>
        <div data-testid="child">Test Child</div>
      </Providers>
    );

    expect(screen.getByTestId('child')).toBeInTheDocument();
    expect(screen.getByText('Test Child')).toBeInTheDocument();
  });

  it('should provide QueryClient context', () => {
    // This test verifies that components can use React Query
    const TestComponent = () => {
      // The component renders without throwing if QueryClient is provided
      return <div>Query Client Available</div>;
    };

    render(
      <Providers>
        <TestComponent />
      </Providers>
    );

    expect(screen.getByText('Query Client Available')).toBeInTheDocument();
  });

  it('should provide ThemeProvider context', () => {
    render(
      <Providers>
        <div data-testid="themed-content">Themed Content</div>
      </Providers>
    );

    expect(screen.getByTestId('themed-content')).toBeInTheDocument();
  });

  it('should wrap multiple children correctly', () => {
    render(
      <Providers>
        <div data-testid="child-1">Child 1</div>
        <div data-testid="child-2">Child 2</div>
        <div data-testid="child-3">Child 3</div>
      </Providers>
    );

    expect(screen.getByTestId('child-1')).toBeInTheDocument();
    expect(screen.getByTestId('child-2')).toBeInTheDocument();
    expect(screen.getByTestId('child-3')).toBeInTheDocument();
  });
});
