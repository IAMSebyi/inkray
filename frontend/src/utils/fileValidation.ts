import type { ImageFormatLabel, SupportedImageMimeType } from '../types/editor';

export const SUPPORTED_IMAGE_MIME_TYPES: readonly SupportedImageMimeType[] = [
  'image/jpeg',
  'image/png',
];

export const SUPPORTED_IMAGE_EXTENSIONS: readonly string[] = ['.jpg', '.jpeg', '.png'];

export const MAX_IMAGE_FILE_BYTES = 32 * 1024 * 1024;
export const MAX_IMAGE_PIXELS = 64_000_000;

export const IMAGE_FILE_ACCEPT = 'image/jpeg,image/png,.jpg,.jpeg,.png';

export type FileValidationResult =
  | {
      valid: true;
      mimeType: SupportedImageMimeType;
      formatLabel: ImageFormatLabel;
    }
  | {
      valid: false;
      error: string;
    };

function getLowercaseExtension(filename: string): string {
  const dotIndex = filename.lastIndexOf('.');
  if (dotIndex <= 0 || dotIndex === filename.length - 1) {
    return '';
  }
  return filename.slice(dotIndex).toLowerCase();
}

export function mimeTypeToFormatLabel(mimeType: SupportedImageMimeType): ImageFormatLabel {
  return mimeType === 'image/png' ? 'PNG' : 'JPEG';
}

export function validateImageFile(
  file: Pick<File, 'name' | 'type' | 'size'>,
): FileValidationResult {
  if (file.size <= 0) {
    return {
      valid: false,
      error: `"${file.name}" is empty and cannot be loaded.`,
    };
  }

  if (file.size > MAX_IMAGE_FILE_BYTES) {
    return { valid: false, error: 'Image files must be no larger than 32 MiB.' };
  }

  const normalizedMime = file.type.trim().toLowerCase();

  if (normalizedMime !== '') {
    if (normalizedMime === 'image/jpeg' || normalizedMime === 'image/jpg') {
      return {
        valid: true,
        mimeType: 'image/jpeg',
        formatLabel: 'JPEG',
      };
    }

    if (normalizedMime === 'image/png') {
      return {
        valid: true,
        mimeType: 'image/png',
        formatLabel: 'PNG',
      };
    }

    return {
      valid: false,
      error: `Unsupported file type "${file.type}" for "${file.name}". Only JPEG and PNG images are supported.`,
    };
  }

  const ext = getLowercaseExtension(file.name);
  if (ext === '.jpg' || ext === '.jpeg') {
    return {
      valid: true,
      mimeType: 'image/jpeg',
      formatLabel: 'JPEG',
    };
  }

  if (ext === '.png') {
    return {
      valid: true,
      mimeType: 'image/png',
      formatLabel: 'PNG',
    };
  }

  return {
    valid: false,
    error: `Unsupported file format for "${file.name}". Please select a JPEG (.jpg, .jpeg) or PNG (.png) image.`,
  };
}

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] as const;
const JPEG_SIGNATURE = [0xff, 0xd8, 0xff] as const;

function startsWithBytes(bytes: Uint8Array, signature: readonly number[]): boolean {
  if (bytes.length < signature.length) {
    return false;
  }
  for (let i = 0; i < signature.length; i++) {
    if (bytes[i] !== signature[i]) {
      return false;
    }
  }
  return true;
}

// Verify header magic bytes to catch renamed or corrupted files before decoding.
export function detectImageFormatFromBytes(bytes: Uint8Array): SupportedImageMimeType | null {
  if (startsWithBytes(bytes, PNG_SIGNATURE)) {
    return 'image/png';
  }
  if (startsWithBytes(bytes, JPEG_SIGNATURE)) {
    return 'image/jpeg';
  }
  return null;
}
