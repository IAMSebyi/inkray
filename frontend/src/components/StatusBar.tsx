import { useSyncExternalStore } from 'react';
import { sampleGeometryDepthAtImagePoint } from '../editor/geometryVisualizations';
import { useEditor } from '../hooks/useEditor';
import { formatDimensions, formatZoomPercent } from '../utils/format';

function CursorPositionReadout({
  hasImage,
  isPointCloud,
}: {
  hasImage: boolean;
  isPointCloud: boolean;
}) {
  const { state, cursorStore } = useEditor();
  const sample = useSyncExternalStore(
    cursorStore.subscribe,
    cursorStore.getSnapshot,
    cursorStore.getSnapshot,
  );

  if (!hasImage || isPointCloud || !sample || !sample.inBounds || !sample.pixel) {
    return (
      <span className="status-item mono" aria-label="Cursor image coordinates">
        <span className="status-label">XY</span>
        <span className="status-muted">—, —</span>
      </span>
    );
  }

  const geometry = state.geometry.data;
  const depthSample =
    geometry && state.image
      ? sampleGeometryDepthAtImagePoint(geometry, sample.imagePoint, {
          width: state.image.metadata.width,
          height: state.image.metadata.height,
        })
      : null;

  return (
    <span className="status-item mono" aria-label="Cursor image coordinates">
      <span className="status-label">X</span>
      <span>{sample.pixel.x}</span>
      <span className="status-label status-label-y">Y</span>
      <span>{sample.pixel.y}</span>
      {geometry && (
        <>
          <span className="status-label status-label-y">Z</span>
          <span className={depthSample?.valid ? undefined : 'status-muted'}>
            {depthSample?.valid && depthSample.depth !== null
              ? depthSample.depth.toFixed(2)
              : 'invalid'}
          </span>
        </>
      )}
    </span>
  );
}

export function StatusBar() {
  const { state } = useEditor();
  const metadata = state.image?.metadata ?? null;
  const hasImage = metadata !== null;
  const isPointCloud =
    (state.viewMode === 'pointcloud' || state.viewMode === 'mesh') && state.geometry.data !== null;

  return (
    <footer className="status-bar" aria-label="Editor status">
      <div className="status-left">
        <span className="status-hint">
          {!hasImage
            ? 'JPEG / PNG • Drag & drop an image into the viewport'
            : isPointCloud
              ? 'Left-Drag: Orbit • Right-Drag: Pan • Wheel: Dolly'
              : 'Wheel: Zoom to cursor • Space + Drag or Middle-Drag: Pan'}
        </span>
      </div>

      <div className="status-right">
        <CursorPositionReadout hasImage={hasImage} isPointCloud={isPointCloud} />

        <span className="status-divider" aria-hidden="true" />

        <span className="status-item mono" aria-label="Image dimensions">
          {metadata ? formatDimensions(metadata.width, metadata.height) : '—'}
        </span>

        <span className="status-divider" aria-hidden="true" />

        <span className="status-item mono" aria-label="Zoom percentage">
          {!hasImage ? '—' : isPointCloud ? '3D' : formatZoomPercent(state.transform.zoom)}
        </span>
      </div>
    </footer>
  );
}
