import { Box, FolderOpen, Layers, SunMedium, X } from 'lucide-react';
import { computeFovYDegrees } from '../editor/pointCloudBuilder';
import { useEditor } from '../hooks/useEditor';
import { formatAspectRatio, formatDimensions, formatFileSize } from '../utils/format';

export function RightSidebar() {
  const { state, openFilePicker, generateGeometry, cancelGeometry } = useEditor();
  const metadata = state.image?.metadata ?? null;
  const geometry = state.geometry.data;
  const geometryStatus = state.geometry.status;

  let geometryBadge = 'Idle';
  let geometryBadgeClass = 'inspector-badge inspector-badge-muted';
  if (geometryStatus === 'loading') {
    geometryBadge = 'Running';
    geometryBadgeClass = 'inspector-badge inspector-badge-accent';
  } else if (geometryStatus === 'ready' && geometry) {
    geometryBadge = geometry.metadata.cached ? 'Cached' : 'Ready';
    geometryBadgeClass = 'inspector-badge';
  } else if (geometryStatus === 'error') {
    geometryBadge = 'Error';
    geometryBadgeClass = 'inspector-badge inspector-badge-danger';
  }

  const totalInferencePixels = geometry
    ? geometry.metadata.inference_width * geometry.metadata.inference_height
    : 0;
  const validCoveragePct =
    geometry && totalInferencePixels > 0
      ? ((geometry.metadata.valid_pixel_count / totalInferencePixels) * 100).toFixed(1)
      : null;
  const fovYDegrees = geometry ? computeFovYDegrees(geometry.metadata.intrinsics) : null;
  const depthRangeLabel =
    geometry?.metadata.depth_range !== null && geometry?.metadata.depth_range !== undefined
      ? `${geometry.metadata.depth_range[0].toFixed(2)} - ${geometry.metadata.depth_range[1].toFixed(2)}`
      : 'Unavailable';

  return (
    <aside className="right-sidebar" aria-label="Inspector">
      <section className="inspector-section">
        <div className="inspector-header">
          <h2 className="inspector-title">Image</h2>
          {metadata && <span className="inspector-badge">{metadata.formatLabel}</span>}
        </div>

        {metadata ? (
          <div className="inspector-content">
            <dl className="metadata-grid">
              <dt>Filename</dt>
              <dd className="metadata-filename" title={metadata.name}>
                {metadata.name}
              </dd>

              <dt>Dimensions</dt>
              <dd className="mono">{formatDimensions(metadata.width, metadata.height)}</dd>

              <dt>Aspect Ratio</dt>
              <dd className="mono">{formatAspectRatio(metadata.width, metadata.height)}</dd>

              <dt>Format</dt>
              <dd className="mono">{metadata.formatLabel}</dd>

              <dt>File Size</dt>
              <dd className="mono">{formatFileSize(metadata.fileSize)}</dd>
            </dl>

            <button
              type="button"
              className="sidebar-action-button"
              onClick={(e) => {
                e.currentTarget.blur();
                openFilePicker();
              }}
              disabled={state.isLoading}
            >
              <FolderOpen size={13} strokeWidth={1.75} />
              <span>Replace Image</span>
            </button>
          </div>
        ) : (
          <p className="inspector-empty-note">
            No image loaded. Import a JPEG or PNG file to inspect its properties.
          </p>
        )}
      </section>

      <section className="inspector-section">
        <div className="inspector-header">
          <h2 className="inspector-title">
            <Box size={13} strokeWidth={1.75} />
            <span>Scene Geometry</span>
          </h2>
          <span className={geometryBadgeClass}>{geometryBadge}</span>
        </div>

        {!metadata ? (
          <p className="inspector-empty-note">
            Import an image first to infer monocular depth, surface normals, and 3D points.
          </p>
        ) : (
          <div className="inspector-content">
            {geometryStatus === 'error' && state.geometry.error && (
              <p className="inspector-error-note" role="alert">
                {state.geometry.error}
              </p>
            )}

            {geometryStatus === 'loading' && (
              <p className="inspector-empty-note">
                Running MoGe-3 inference on the backend. You can cancel or switch images safely.
              </p>
            )}

            {geometry && (
              <dl className="metadata-grid">
                <dt>Model</dt>
                <dd className="mono" title={geometry.metadata.model_id}>
                  {geometry.metadata.model_id}
                </dd>

                <dt>Source</dt>
                <dd className="mono">{geometry.metadata.cached ? 'Disk cache' : 'Inferred'}</dd>

                <dt>Inference</dt>
                <dd className="mono">
                  {formatDimensions(
                    geometry.metadata.inference_width,
                    geometry.metadata.inference_height,
                  )}
                </dd>

                <dt>Valid Pixels</dt>
                <dd className="mono">
                  {geometry.metadata.valid_pixel_count.toLocaleString()}
                  {validCoveragePct !== null ? ` (${validCoveragePct}%)` : ''}
                </dd>

                <dt>Depth Z</dt>
                <dd className="mono">{depthRangeLabel}</dd>

                <dt>Est. FOV Y</dt>
                <dd className="mono">
                  {fovYDegrees !== null ? `${fovYDegrees.toFixed(1)} deg` : '—'}
                </dd>

                <dt>Normals</dt>
                <dd className="mono">{geometry.metadata.has_normals ? 'Available' : 'None'}</dd>
              </dl>
            )}

            {geometryStatus === 'loading' ? (
              <button
                type="button"
                className="sidebar-action-button toolbar-button-busy"
                onClick={(e) => {
                  e.currentTarget.blur();
                  cancelGeometry();
                }}
              >
                <X size={13} strokeWidth={1.75} />
                <span>Cancel Inference</span>
              </button>
            ) : (
              <button
                type="button"
                className="sidebar-action-button"
                onClick={(e) => {
                  e.currentTarget.blur();
                  void generateGeometry();
                }}
                disabled={state.isLoading}
              >
                <Box size={13} strokeWidth={1.75} />
                <span>
                  {geometryStatus === 'error'
                    ? 'Retry Geometry'
                    : geometry
                      ? 'Regenerate Geometry'
                      : 'Generate Geometry'}
                </span>
              </button>
            )}
          </div>
        )}
      </section>

      <section className="inspector-section inspector-section-planned">
        <div className="inspector-header">
          <h2 className="inspector-title">
            <SunMedium size={13} strokeWidth={1.75} />
            <span>Intrinsic Lighting</span>
          </h2>
          <span className="inspector-badge inspector-badge-muted">Planned</span>
        </div>
        <p className="inspector-planned-note">
          Albedo, shading, and residual decomposition controls (Marigold IID) are not implemented
          yet.
        </p>
      </section>

      <section className="inspector-section inspector-section-planned">
        <div className="inspector-header">
          <h2 className="inspector-title">
            <Layers size={13} strokeWidth={1.75} />
            <span>Layers</span>
          </h2>
          <span className="inspector-badge inspector-badge-muted">Planned</span>
        </div>
        <p className="inspector-planned-note">
          Non-destructive paint layers and surface masks will be managed here.
        </p>
      </section>
    </aside>
  );
}
