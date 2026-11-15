import { describe, expect, it } from 'vitest';
import { formatAspectRatio, formatDimensions, formatFileSize, formatZoomPercent } from './format';

describe('format utilities', () => {
  it('formats file sizes cleanly across B, KB, MB, GB', () => {
    expect(formatFileSize(512)).toBe('512 B');
    expect(formatFileSize(1536)).toBe('1.50 KB');
    expect(formatFileSize(3 * 1024 * 1024)).toBe('3.00 MB');
    expect(formatFileSize(25.4 * 1024 * 1024)).toBe('25.4 MB');
  });

  it('formats zoom percentages with extra precision below 10%', () => {
    expect(formatZoomPercent(1)).toBe('100%');
    expect(formatZoomPercent(0.678)).toBe('68%');
    expect(formatZoomPercent(0.045)).toBe('4.5%');
    expect(formatZoomPercent(0)).toBe('—');
  });

  it('formats dimensions and aspect ratios', () => {
    expect(formatDimensions(4032, 3024)).toBe('4032 × 3024 px');
    expect(formatDimensions(0, 100)).toBe('—');
    expect(formatAspectRatio(1920, 1080)).toBe('16:9 (1.78)');
    expect(formatAspectRatio(4032, 3024)).toBe('4:3 (1.33)');
    expect(formatAspectRatio(1000, 777)).toBe('1.29:1');
  });
});
