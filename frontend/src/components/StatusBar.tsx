import { useSyncExternalStore } from 'react';
import { sampleGeometryDepthAtImagePoint } from '../editor/geometryVisualizations';
import { useEditor } from '../hooks/useEditor';
import { PaintDocument } from '../painting/paintDocument';
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
  const isArtworkMode = state.workspaceMode === 'artwork';
  const activeArtwork = PaintDocument.getActiveArtwork(state.paint);
  const activePlacement = PaintDocument.getActivePlacement(state.paint);
  const isPointCloud =
    !isArtworkMode &&
    (state.viewMode === 'pointcloud' || state.viewMode === 'mesh') &&
    state.geometry.data !== null;
  const lastAnchor = state.paint.lastAnchorMode;
  const surfaceModeLabel = activePlacement
    ? activePlacement.mapping.type === 'curved'
      ? 'Curved Art'
      : activePlacement.mapping.type === 'planar'
        ? activePlacement.mapping.isApproximatelyPlanar
          ? 'Approx. Planar'
          : 'Planar Art'
        : '2D Art'
    : lastAnchor === 'planar'
      ? 'Planar'
      : lastAnchor === 'curved'
        ? 'Curved'
        : lastAnchor === 'fallback-2d'
          ? '2D Fallback'
          : state.geometry.data
            ? 'Auto 3D'
            : '2D';

  const hintText = isArtworkMode
    ? 'Draw on transparent canvas with Brush (B) / Eraser (E) • Return to Photo to place artwork'
    : !hasImage
      ? 'JPEG / PNG • Drag & drop an image into the viewport, or open Artwork Canvas'
      : isPointCloud
        ? 'Left-Drag: Orbit • Right-Drag: Pan • Wheel: Dolly'
        : activePlacement && state.activeTool === 'pointer'
          ? 'Drag artwork to move • Corners: scale • Top knob: rotate • Delete: remove'
          : 'Wheel: Zoom to cursor • Space + Drag or Middle-Drag: Pan';

  return (
    <footer className="status-bar" aria-label="Editor status">
      <div className="status-left">
        <span className="status-hint">{hintText}</span>
      </div>

      <div className="status-right">
        {!isArtworkMode && hasImage && !isPointCloud && (
          <>
            <span className="status-item mono" aria-label="Active brush anchoring mode">
              <span className="status-label">Mode</span>
              <span>{surfaceModeLabel}</span>
            </span>
            <span className="status-divider" aria-hidden="true" />
          </>
        )}

        <CursorPositionReadout hasImage={!isArtworkMode && hasImage} isPointCloud={isPointCloud} />

        <span className="status-divider" aria-hidden="true" />

        <span className="status-item mono" aria-label="Image dimensions">
          {isArtworkMode && activeArtwork
            ? formatDimensions(activeArtwork.width, activeArtwork.height)
            : metadata
              ? formatDimensions(metadata.width, metadata.height)
              : '—'}
        </span>

        <span className="status-divider" aria-hidden="true" />

        <span className="status-item mono" aria-label="Zoom percentage">
          {isArtworkMode
            ? 'Canvas'
            : !hasImage
              ? '—'
              : isPointCloud
                ? '3D'
                : formatZoomPercent(state.transform.zoom)}
        </span>
      </div>
    </footer>
  );
}
