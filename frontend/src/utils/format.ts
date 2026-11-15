export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return '—';
  }
  if (bytes < 1024) {
    return `${Math.round(bytes)} B`;
  }
  const units = ['KB', 'MB', 'GB'] as const;
  let value = bytes / 1024;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  const unit = units[unitIndex] ?? 'GB';
  const decimals = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(decimals)} ${unit}`;
}

export function formatZoomPercent(zoom: number): string {
  if (!Number.isFinite(zoom) || zoom <= 0) {
    return '—';
  }
  const percent = zoom * 100;
  if (percent < 10) {
    return `${percent.toFixed(1)}%`;
  }
  return `${Math.round(percent)}%`;
}

export function formatDimensions(width: number, height: number): string {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return '—';
  }
  return `${Math.round(width)} × ${Math.round(height)} px`;
}

function gcd(a: number, b: number): number {
  let x = Math.abs(Math.round(a));
  let y = Math.abs(Math.round(b));
  while (y !== 0) {
    const t = y;
    y = x % y;
    x = t;
  }
  return x || 1;
}

export function formatAspectRatio(width: number, height: number): string {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return '—';
  }
  const w = Math.round(width);
  const h = Math.round(height);
  const divisor = gcd(w, h);
  const ratioW = w / divisor;
  const ratioH = h / divisor;
  const decimal = (w / h).toFixed(2);

  if (ratioW <= 32 && ratioH <= 32) {
    return `${ratioW}:${ratioH} (${decimal})`;
  }
  return `${decimal}:1`;
}
