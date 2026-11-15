import { describe, expect, it } from 'vitest';
import { detectImageFormatFromBytes, validateImageFile } from './fileValidation';

describe('fileValidation', () => {
  describe('validateImageFile', () => {
    it('accepts valid JPEG and PNG MIME types', () => {
      expect(validateImageFile({ name: 'photo.jpg', type: 'image/jpeg', size: 1024 })).toEqual({
        valid: true,
        mimeType: 'image/jpeg',
        formatLabel: 'JPEG',
      });

      expect(validateImageFile({ name: 'alpha.png', type: 'image/png', size: 2048 })).toEqual({
        valid: true,
        mimeType: 'image/png',
        formatLabel: 'PNG',
      });
    });

    it('falls back to extension when MIME type is empty', () => {
      expect(validateImageFile({ name: 'RAW_SCAN.JPEG', type: '', size: 512 })).toEqual({
        valid: true,
        mimeType: 'image/jpeg',
        formatLabel: 'JPEG',
      });

      expect(validateImageFile({ name: 'diagram.PNG', type: '', size: 512 })).toEqual({
        valid: true,
        mimeType: 'image/png',
        formatLabel: 'PNG',
      });
    });

    it('rejects empty files and unsupported formats', () => {
      const empty = validateImageFile({ name: 'empty.jpg', type: 'image/jpeg', size: 0 });
      expect(empty.valid).toBe(false);

      const webp = validateImageFile({ name: 'img.webp', type: 'image/webp', size: 1024 });
      expect(webp.valid).toBe(false);

      const pdf = validateImageFile({ name: 'doc.pdf', type: 'application/pdf', size: 4096 });
      expect(pdf.valid).toBe(false);

      const unknownExt = validateImageFile({ name: 'notes.txt', type: '', size: 128 });
      expect(unknownExt.valid).toBe(false);
    });
  });

  describe('detectImageFormatFromBytes', () => {
    it('detects PNG and JPEG signatures and rejects non-image bytes', () => {
      const pngBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00]);
      expect(detectImageFormatFromBytes(pngBytes)).toBe('image/png');

      const jpegBytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
      expect(detectImageFormatFromBytes(jpegBytes)).toBe('image/jpeg');

      const fakeBytes = new TextEncoder().encode('not actually a png');
      expect(detectImageFormatFromBytes(fakeBytes)).toBeNull();

      expect(detectImageFormatFromBytes(new Uint8Array([0xff, 0xd8]))).toBeNull();
    });
  });
});
