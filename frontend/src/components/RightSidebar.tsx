import { Box, FolderOpen, Layers, SunMedium } from 'lucide-react';
import { useEditor } from '../hooks/useEditor';
import { formatAspectRatio, formatDimensions, formatFileSize } from '../utils/format';

export function RightSidebar() {
  const { state, openFilePicker } = useEditor();
  const metadata = state.image?.metadata ?? null;

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

      <section className="inspector-section inspector-section-planned">
        <div className="inspector-header">
          <h2 className="inspector-title">
            <Box size={13} strokeWidth={1.75} />
            <span>Scene Geometry</span>
          </h2>
          <span className="inspector-badge inspector-badge-muted">Planned</span>
        </div>
        <p className="inspector-planned-note">
          Monocular depth, surface normals, and camera intrinsics (MoGe-3) are not implemented yet.
        </p>
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
