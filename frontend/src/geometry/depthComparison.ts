import type { GeometryData } from '../types/editor';

export interface DepthComparison {
  width: number;
  height: number;
  scale: number;
  overlap: number;
  differingFraction: number;
  logDifference: Float32Array;
}

function medianSorted(values: Float64Array): number {
  const length = values.length;
  const lo = values[Math.floor((length - 1) / 2)] ?? NaN;
  const hi = values[Math.floor(length / 2)] ?? NaN;
  return (lo + hi) / 2;
}

export function compareDepth(
  reference: GeometryData,
  candidate: GeometryData,
): DepthComparison | null {
  if (
    reference.imageId !== candidate.imageId ||
    reference.metadata.image_sha256 !== candidate.metadata.image_sha256
  ) {
    return null;
  }

  const width = Math.min(reference.metadata.inference_width, candidate.metadata.inference_width);
  const height = Math.min(reference.metadata.inference_height, candidate.metadata.inference_height);
  const count = width * height;
  const logDifference = new Float32Array(count).fill(NaN);
  const valid = new Float64Array(count);
  let overlap = 0;

  const sample = (data: GeometryData, u: number, v: number): number => {
    const w = data.metadata.inference_width;
    const h = data.metadata.inference_height;
    const index = Math.floor(v * h) * w + Math.floor(u * w);
    return data.mask[index] ? (data.depth[index] ?? NaN) : NaN;
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const u = (x + 0.5) / width;
      const v = (y + 0.5) / height;
      const a = sample(reference, u, v);
      const b = sample(candidate, u, v);
      if (Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0) {
        const difference = Math.log(b) - Math.log(a);
        logDifference[y * width + x] = difference;
        valid[overlap++] = difference;
      }
    }
  }

  if (!overlap) {
    return null;
  }
  const values = valid.subarray(0, overlap).sort();
  const middle = medianSorted(values);
  const threshold = Math.log(1.25);
  let differing = 0;
  for (const [i, value] of logDifference.entries()) {
    const aligned = value - middle;
    logDifference[i] = aligned;
    if (Math.abs(aligned) > threshold) differing++;
  }
  return {
    width,
    height,
    scale: Math.exp(-middle),
    overlap: overlap / count,
    differingFraction: differing / overlap,
    logDifference,
  };
}
