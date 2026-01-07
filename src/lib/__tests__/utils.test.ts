import { describe, it, expect } from 'vitest';
import { cn } from '@/lib/utils';

describe('utils', () => {
  describe('cn', () => {
    it('should merge class names correctly', () => {
      const result = cn('px-2 py-1', 'px-4');
      expect(result).toBe('py-1 px-4');
    });

    it('should handle conditional classes', () => {
      const result = cn('base', true && 'conditional', false && 'hidden');
      expect(result).toBe('base conditional');
    });

    it('should handle empty inputs', () => {
      const result = cn();
      expect(result).toBe('');
    });

    it('should handle undefined and null values', () => {
      const result = cn('base', undefined, null, 'extra');
      expect(result).toBe('base extra');
    });

    it('should handle arrays of class names', () => {
      const result = cn(['class1', 'class2'], 'class3');
      expect(result).toBe('class1 class2 class3');
    });

    it('should handle object syntax for conditional classes', () => {
      const result = cn('base', {
        'active': true,
        'disabled': false,
        'highlight': true,
      });
      expect(result).toBe('base active highlight');
    });

    it('should merge tailwind classes correctly with conflicts', () => {
      const result = cn('text-red-500', 'text-blue-500');
      expect(result).toBe('text-blue-500');
    });

    it('should handle responsive classes', () => {
      const result = cn('w-full', 'md:w-1/2', 'lg:w-1/3');
      expect(result).toBe('w-full md:w-1/2 lg:w-1/3');
    });

    it('should handle hover and focus states', () => {
      const result = cn('bg-blue-500', 'hover:bg-blue-600', 'focus:bg-blue-700');
      expect(result).toBe('bg-blue-500 hover:bg-blue-600 focus:bg-blue-700');
    });

    it('should handle dark mode classes', () => {
      const result = cn('bg-white', 'dark:bg-gray-900');
      expect(result).toBe('bg-white dark:bg-gray-900');
    });

    it('should handle complex nested conditional classes', () => {
      const isActive = true;
      const isDisabled = false;
      type Size = 'small' | 'medium' | 'large';
      const getSize = (): Size => 'large';
      const size = getSize();
      
      const result = cn(
        'base-class',
        isActive && 'is-active',
        isDisabled && 'is-disabled',
        {
          'size-small': size === 'small',
          'size-medium': size === 'medium',
          'size-large': size === 'large',
        }
      );
      
      expect(result).toBe('base-class is-active size-large');
    });

    it('should handle spacing classes', () => {
      const result = cn('p-2', 'p-4');
      expect(result).toBe('p-4');
    });

    it('should handle multiple conflicting classes', () => {
      const result = cn('m-2 p-2 text-sm', 'm-4 p-4 text-lg');
      expect(result).toBe('m-4 p-4 text-lg');
    });
  });
});
