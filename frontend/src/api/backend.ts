import {
  GEOMETRY_MODELS,
  NORMAL_MODELS,
  type AnyModelId,
  type GeometryModel,
  type NormalModel,
} from '../types/editor';

export const API_BASE = ((import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '').replace(
  /\/+$/,
  '',
);

export interface ModelAvailability {
  default_geometry: GeometryModel;
  default_normal?: NormalModel;
  unavailable: Partial<Record<AnyModelId, string>>;
}

export async function getModelAvailability(signal: AbortSignal): Promise<ModelAvailability | null> {
  const response = await fetch(`${API_BASE}/api/models`, { signal });
  if (!response.ok) return null;
  const models = (await response.json()) as Partial<ModelAvailability> | null;
  if (
    !models ||
    !models.default_geometry ||
    !(models.default_geometry in GEOMETRY_MODELS) ||
    !models.unavailable ||
    typeof models.unavailable !== 'object'
  ) {
    return null;
  }
  const defaultNormal =
    models.default_normal && models.default_normal in NORMAL_MODELS
      ? models.default_normal
      : undefined;
  return {
    default_geometry: models.default_geometry,
    default_normal: defaultNormal,
    unavailable: models.unavailable,
  };
}

export async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === 'string' && body.detail.trim() !== '') {
      return body.detail;
    }
  } catch {
    // Ignore non-JSON error bodies.
  }
  return `Request failed with HTTP ${response.status}.`;
}
