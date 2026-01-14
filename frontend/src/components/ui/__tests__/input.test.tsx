import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/test-utils';
import { Input } from '@/components/ui/input';
import userEvent from '@testing-library/user-event';

describe('Input', () => {
  it('should render with default styles', () => {
    render(<Input placeholder="Enter text" />);
    
    const input = screen.getByPlaceholderText('Enter text');
    expect(input).toBeInTheDocument();
    expect(input).toHaveClass('flex', 'h-10', 'w-full', 'rounded-md');
  });

  it('should accept text input', async () => {
    const user = userEvent.setup();
    render(<Input placeholder="Type here" />);
    
    const input = screen.getByPlaceholderText('Type here');
    await user.type(input, 'Hello World');
    
    expect(input).toHaveValue('Hello World');
  });

  it('should handle different input types', () => {
    const { rerender } = render(<Input type="text" data-testid="input" />);
    expect(screen.getByTestId('input')).toHaveAttribute('type', 'text');

    rerender(<Input type="email" data-testid="input" />);
    expect(screen.getByTestId('input')).toHaveAttribute('type', 'email');

    rerender(<Input type="password" data-testid="input" />);
    expect(screen.getByTestId('input')).toHaveAttribute('type', 'password');

    rerender(<Input type="number" data-testid="input" />);
    expect(screen.getByTestId('input')).toHaveAttribute('type', 'number');
  });

  it('should be disabled when disabled prop is passed', () => {
    render(<Input disabled placeholder="Disabled" />);
    
    const input = screen.getByPlaceholderText('Disabled');
    expect(input).toBeDisabled();
  });

  it('should apply custom className', () => {
    render(<Input className="custom-input" placeholder="Custom" />);
    
    expect(screen.getByPlaceholderText('Custom')).toHaveClass('custom-input');
  });

  it('should call onChange handler', async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();
    
    render(<Input onChange={handleChange} placeholder="Change test" />);
    
    await user.type(screen.getByPlaceholderText('Change test'), 'a');
    
    expect(handleChange).toHaveBeenCalled();
  });

  it('should call onFocus and onBlur handlers', async () => {
    const user = userEvent.setup();
    const handleFocus = vi.fn();
    const handleBlur = vi.fn();
    
    render(<Input onFocus={handleFocus} onBlur={handleBlur} placeholder="Focus test" />);
    
    const input = screen.getByPlaceholderText('Focus test');
    
    await user.click(input);
    expect(handleFocus).toHaveBeenCalled();
    
    await user.tab();
    expect(handleBlur).toHaveBeenCalled();
  });

  it('should support controlled value', () => {
    const { rerender } = render(<Input value="controlled" onChange={() => {}} placeholder="Controlled" />);
    
    expect(screen.getByPlaceholderText('Controlled')).toHaveValue('controlled');
    
    rerender(<Input value="updated" onChange={() => {}} placeholder="Controlled" />);
    expect(screen.getByPlaceholderText('Controlled')).toHaveValue('updated');
  });

  it('should forward refs correctly', () => {
    const ref = vi.fn();
    render(<Input ref={ref} placeholder="Ref test" />);
    
    expect(ref).toHaveBeenCalled();
  });

  it('should support required attribute', () => {
    render(<Input required placeholder="Required" />);
    
    expect(screen.getByPlaceholderText('Required')).toBeRequired();
  });

  it('should support maxLength attribute', async () => {
    const user = userEvent.setup();
    render(<Input maxLength={5} placeholder="Max length" />);
    
    const input = screen.getByPlaceholderText('Max length');
    await user.type(input, '1234567890');
    
    expect(input).toHaveValue('12345');
  });

  it('should support pattern attribute', () => {
    render(<Input pattern="[0-9]*" placeholder="Pattern" />);
    
    expect(screen.getByPlaceholderText('Pattern')).toHaveAttribute('pattern', '[0-9]*');
  });

  it('should support aria attributes', () => {
    render(
      <Input 
        aria-label="Accessible input"
        aria-describedby="help-text"
        placeholder="Aria test"
      />
    );
    
    const input = screen.getByPlaceholderText('Aria test');
    expect(input).toHaveAttribute('aria-label', 'Accessible input');
    expect(input).toHaveAttribute('aria-describedby', 'help-text');
  });
});
