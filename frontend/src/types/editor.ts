export interface Point2D {
  x: number;
  y: number;
}

export interface Size2D {
  width: number;
  height: number;
}

export interface ViewportRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface ViewportTransform {
  /** CSS pixels per source image pixel (1 = 100%). */
  zoom: number;
  panX: number;
  panY: number;
}

export type SupportedImageMimeType = 'image/jpeg' | 'image/png';

export type ImageFormatLabel = 'JPEG' | 'PNG';

export interface ImageMetadata {
  name: string;
  width: number;
  height: number;
  fileSize: number;
  mimeType: SupportedImageMimeType;
  formatLabel: ImageFormatLabel;
}

export interface LoadedImage {
  id: string;
  metadata: ImageMetadata;
  element: HTMLImageElement;
  sourceFile: File;
  objectUrl: string;
  /** Lazily extracts unmodified RGBA pixels at full native resolution. */
  getPixelData: () => ImageData;
  dispose: () => void;
}

export interface ImageCoordinateSample {
  /** Continuous coordinates in source image space, top-left at (0, 0). */
  imagePoint: Point2D;
  /** Integer pixel index [0..width-1, 0..height-1], or null outside bounds. */
  pixel: Point2D | null;
  inBounds: boolean;
}

export type EditorTool = 'pointer' | 'hand';

export type ZoomMode = 'fit' | 'custom';
