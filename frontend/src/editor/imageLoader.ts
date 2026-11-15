import type { LoadedImage } from '../types/editor';
import {
  detectImageFormatFromBytes,
  mimeTypeToFormatLabel,
  validateImageFile,
} from '../utils/fileValidation';

export class ImageLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ImageLoadError';
  }
}

export function extractImagePixelData(image: HTMLImageElement): ImageData {
  const width = image.naturalWidth;
  const height = image.naturalHeight;

  if (width <= 0 || height <= 0) {
    throw new ImageLoadError('Cannot extract pixel data from an empty image.');
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new ImageLoadError('Failed to acquire 2D canvas context for pixel extraction.');
  }

  ctx.drawImage(image, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

function decodeHtmlImage(
  objectUrl: string,
  filename: string,
  signal?: AbortSignal,
): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Image loading was cancelled.', 'AbortError'));
      return;
    }

    const img = new Image();
    img.decoding = 'async';

    const cleanupListeners = () => {
      img.onload = null;
      img.onerror = null;
      signal?.removeEventListener('abort', onAbort);
    };

    const onAbort = () => {
      cleanupListeners();
      img.src = '';
      reject(new DOMException('Image loading was cancelled.', 'AbortError'));
    };

    signal?.addEventListener('abort', onAbort, { once: true });

    img.onload = () => {
      cleanupListeners();
      if (img.naturalWidth <= 0 || img.naturalHeight <= 0) {
        reject(new ImageLoadError(`"${filename}" has invalid image dimensions.`));
        return;
      }
      if (typeof img.decode === 'function') {
        img
          .decode()
          .then(() => resolve(img))
          .catch(() => {
            reject(
              new ImageLoadError(
                `Failed to decode "${filename}". The image file may be corrupted or truncated.`,
              ),
            );
          });
      } else {
        resolve(img);
      }
    };

    img.onerror = () => {
      cleanupListeners();
      reject(
        new ImageLoadError(
          `Failed to load "${filename}". The file may be corrupted or unreadable.`,
        ),
      );
    };

    img.src = objectUrl;
  });
}

let nextImageId = 1;

export async function loadImageFromFile(file: File, signal?: AbortSignal): Promise<LoadedImage> {
  const validation = validateImageFile(file);
  if (!validation.valid) {
    throw new ImageLoadError(validation.error);
  }

  const headerBuffer = await file.slice(0, 12).arrayBuffer();
  if (signal?.aborted) {
    throw new DOMException('Image loading was cancelled.', 'AbortError');
  }

  const detectedMime = detectImageFormatFromBytes(new Uint8Array(headerBuffer));
  if (!detectedMime) {
    throw new ImageLoadError(
      `"${file.name}" is not a valid JPEG or PNG image, or its file header is corrupted.`,
    );
  }

  const objectUrl = URL.createObjectURL(file);
  let disposed = false;

  const dispose = () => {
    if (!disposed) {
      disposed = true;
      URL.revokeObjectURL(objectUrl);
    }
  };

  try {
    const element = await decodeHtmlImage(objectUrl, file.name, signal);
    if (signal?.aborted) {
      throw new DOMException('Image loading was cancelled.', 'AbortError');
    }
    const id = `img-${nextImageId++}`;

    return {
      id,
      metadata: {
        name: file.name,
        width: element.naturalWidth,
        height: element.naturalHeight,
        fileSize: file.size,
        mimeType: detectedMime,
        formatLabel: mimeTypeToFormatLabel(detectedMime),
      },
      element,
      sourceFile: file,
      objectUrl,
      getPixelData: () => extractImagePixelData(element),
      dispose,
    };
  } catch (error) {
    dispose();
    throw error;
  }
}
