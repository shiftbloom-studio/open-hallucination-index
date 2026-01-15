import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/test-utils';
import {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/card';

describe('Card Components', () => {
  describe('Card', () => {
    it('should render with default styles', () => {
      render(<Card data-testid="card">Card Content</Card>);
      
      const card = screen.getByTestId('card');
      expect(card).toBeInTheDocument();
      expect(card).toHaveClass('rounded-lg', 'border', 'bg-card', 'shadow-sm');
    });

    it('should apply custom className', () => {
      render(<Card className="custom-card" data-testid="card">Content</Card>);
      
      expect(screen.getByTestId('card')).toHaveClass('custom-card');
    });

    it('should forward refs correctly', () => {
      const ref = vi.fn();
      render(<Card ref={ref}>Ref Card</Card>);
      
      expect(ref).toHaveBeenCalled();
    });

    it('should pass through additional props', () => {
      render(<Card data-testid="card" role="article">Content</Card>);
      
      expect(screen.getByTestId('card')).toHaveAttribute('role', 'article');
    });
  });

  describe('CardHeader', () => {
    it('should render with default styles', () => {
      render(<CardHeader data-testid="header">Header</CardHeader>);
      
      const header = screen.getByTestId('header');
      expect(header).toHaveClass('flex', 'flex-col', 'space-y-1.5', 'p-6');
    });

    it('should apply custom className', () => {
      render(<CardHeader className="custom-header" data-testid="header">Header</CardHeader>);
      
      expect(screen.getByTestId('header')).toHaveClass('custom-header');
    });
  });

  describe('CardTitle', () => {
    it('should render with default styles', () => {
      render(<CardTitle data-testid="title">Title</CardTitle>);
      
      const title = screen.getByTestId('title');
      expect(title).toHaveClass('text-2xl', 'font-semibold', 'leading-none', 'tracking-tight');
    });

    it('should render text content', () => {
      render(<CardTitle>My Card Title</CardTitle>);
      
      expect(screen.getByText('My Card Title')).toBeInTheDocument();
    });
  });

  describe('CardDescription', () => {
    it('should render with default styles', () => {
      render(<CardDescription data-testid="desc">Description</CardDescription>);
      
      const desc = screen.getByTestId('desc');
      expect(desc).toHaveClass('text-sm', 'text-muted-foreground');
    });

    it('should render text content', () => {
      render(<CardDescription>Card description text</CardDescription>);
      
      expect(screen.getByText('Card description text')).toBeInTheDocument();
    });
  });

  describe('CardContent', () => {
    it('should render with default styles', () => {
      render(<CardContent data-testid="content">Content</CardContent>);
      
      const content = screen.getByTestId('content');
      expect(content).toHaveClass('p-6', 'pt-0');
    });

    it('should render children', () => {
      render(
        <CardContent>
          <p>Paragraph content</p>
          <span>Span content</span>
        </CardContent>
      );
      
      expect(screen.getByText('Paragraph content')).toBeInTheDocument();
      expect(screen.getByText('Span content')).toBeInTheDocument();
    });
  });

  describe('CardFooter', () => {
    it('should render with default styles', () => {
      render(<CardFooter data-testid="footer">Footer</CardFooter>);
      
      const footer = screen.getByTestId('footer');
      expect(footer).toHaveClass('flex', 'items-center', 'p-6', 'pt-0');
    });
  });

  describe('Full Card Composition', () => {
    it('should render complete card with all components', () => {
      render(
        <Card data-testid="full-card">
          <CardHeader>
            <CardTitle>Card Title</CardTitle>
            <CardDescription>Card Description</CardDescription>
          </CardHeader>
          <CardContent>
            <p>Main content goes here</p>
          </CardContent>
          <CardFooter>
            <button>Action</button>
          </CardFooter>
        </Card>
      );

      expect(screen.getByTestId('full-card')).toBeInTheDocument();
      expect(screen.getByText('Card Title')).toBeInTheDocument();
      expect(screen.getByText('Card Description')).toBeInTheDocument();
      expect(screen.getByText('Main content goes here')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Action' })).toBeInTheDocument();
    });

    it('should maintain proper structure and styling', () => {
      render(
        <Card data-testid="structured-card">
          <CardHeader>
            <CardTitle>Title</CardTitle>
          </CardHeader>
          <CardContent>Content</CardContent>
        </Card>
      );

      // Check that card renders with content
      const card = screen.getByTestId('structured-card');
      expect(card).toBeInTheDocument();
      expect(screen.getByText('Title')).toBeInTheDocument();
      expect(screen.getByText('Content')).toBeInTheDocument();
    });
  });
});
