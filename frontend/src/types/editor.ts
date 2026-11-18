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

export type EditorTool = 'pointer' | 'hand' | 'select-surface' | 'brush' | 'eraser';

export type WorkspaceMode = 'photo' | 'artwork';

export type ZoomMode = 'fit' | 'custom';

export type ViewMode =
  | 'rgb'
  | 'depth'
  | 'depth-difference'
  | 'normals'
  | 'mask'
  | 'pointcloud'
  | 'mesh'
  | 'albedo'
  | 'shading'
  | 'residual';

export type GeometryStatus = 'idle' | 'loading' | 'ready' | 'error';

export type AppearanceStatus = 'idle' | 'loading' | 'ready' | 'error';

export type PaintMode = 'flat' | 'shading' | 'texture';

export interface GeometryArraySpec {
  shape: number[];
  dtype: string;
}

export interface GeometryMetadata {
  id: string;
  cache_version: number;
  image_sha256: string;
  model_id: string;
  model?: GeometryModel;
  normal_model_id?: string | null;
  smooth_iterations?: number;
  resolution_level: number;
  refine_steps: number;
  use_fp16: boolean;
  max_image_size: number;
  original_width: number;
  original_height: number;
  inference_width: number;
  inference_height: number;
  coordinate_system: 'opencv';
  intrinsics_convention: 'normalized';
  intrinsics: number[][];
  depth_range: [number, number] | null;
  valid_pixel_count: number;
  has_normals: boolean;
  cached: boolean;
  arrays: Record<string, GeometryArraySpec>;
}

export interface GeometryPreviews {
  depth: HTMLCanvasElement;
  normals: HTMLCanvasElement | null;
  mask: HTMLCanvasElement;
}

export interface GeometryData {
  imageId: string;
  metadata: GeometryMetadata;
  depth: Float32Array;
  points: Float32Array;
  mask: Uint8Array;
  normals: Float32Array | null;
  colors: Uint8ClampedArray;
  previews: GeometryPreviews;
}

export interface GeometryState {
  status: GeometryStatus;
  data: GeometryData | null;
  error: string | null;
}

export const GEOMETRY_MODELS = {
  'moge-3': 'MoGe-3',
  'depth-pro': 'Depth Pro',
} as const;

export const NORMAL_MODELS = {
  'marigold-normals': 'Marigold Normals v1.1',
  'marigold-v2': 'Marigold V2 Normals',
} as const;

export const APPEARANCE_MODELS = {
  'marigold-iid': 'Marigold IID Lighting',
  'marigold-v2': 'Marigold V2 (albedo)',
  'latent-intrinsics': 'Latent Intrinsics (albedo)',
  'iid-in-the-wild': 'IID-in-the-Wild',
} as const;

export type GeometryModel = keyof typeof GEOMETRY_MODELS;
export type NormalModel = keyof typeof NORMAL_MODELS;
export type AppearanceModel = keyof typeof APPEARANCE_MODELS;
export type AnyModelId = GeometryModel | NormalModel | AppearanceModel;

export const MODEL_USAGE = {
  'moge-3': 'Commercial use allowed',
  'depth-pro': 'Commercial use allowed under Apple terms',
  'marigold-normals': 'Commercial use allowed with Open RAIL restrictions',
  'marigold-iid': 'Commercial use allowed with Open RAIL restrictions',
  'marigold-v2': 'Commercial use allowed',
  'latent-intrinsics': 'Non-commercial',
  'iid-in-the-wild': 'Non-commercial (academic use only)',
} as const satisfies Record<AnyModelId, string>;

export const MODEL_OPTION_NOTES: Partial<Record<AnyModelId, string>> = {
  'latent-intrinsics': 'non-commercial',
  'iid-in-the-wild': 'non-commercial',
};

export const MODEL_DOWNLOAD_NOTES: Partial<Record<AnyModelId, string>> = {
  'marigold-v2':
    'The first run downloads up to about 43 GB of weights. Albedo and Normals share the 41 GB Qwen download.',
};

export interface AppearanceMetadata {
  id: string;
  cache_version: number;
  image_sha256: string;
  model_id: string;
  denoising_steps: number;
  processing_resolution: number;
  ensemble_size: number;
  seed: number;
  use_fp16: boolean;
  max_image_size: number;
  original_width: number;
  original_height: number;
  appearance_width: number;
  appearance_height: number;
  prediction_space: string;
  scale_conventions: Record<string, string>;
  albedo_range: [number, number] | null;
  shading_range: [number, number] | null;
  residual_range: [number, number] | null;
  cached: boolean;
  arrays: Record<string, GeometryArraySpec>;
}

export interface AppearancePreviews {
  albedo: HTMLCanvasElement;
  shading: HTMLCanvasElement | null;
  residual: HTMLCanvasElement | null;
}

export interface AppearanceData {
  imageId: string;
  metadata: AppearanceMetadata;
  albedo: Float32Array;
  shading: Float32Array | null;
  residual: Float32Array | null;
  linearReference: Float32Array;
  derivedShading?: boolean;
  previews: AppearancePreviews;
}

export interface AppearanceState {
  status: AppearanceStatus;
  data: AppearanceData | null;
  error: string | null;
}
