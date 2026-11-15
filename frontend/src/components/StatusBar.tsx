import { useSyncExternalStore } from 'react';
import { useEditor } from '../hooks/useEditor';
import { formatDimensions, formatZoomPercent } from '../utils/format';

function CursorPositionReadout({ hasImage }: { hasImage: boolean }) {
  const { cursorStore } = useEditor();
  const sample = useSyncExternalStore(
    cursorStore.subscribe,
    cursorStore.getSnapshot,
    cursorStore.getSnapshot,
  );

  if (!hasImage || !sample || !sample.inBounds || !sample.pixel) {
    return (
      <span className="status-item mono" aria-label="Cursor image coordinates">
        <span className="status-label">XY</span>
        <span className="status-muted">—, —</span>
      </span>
    );
  }

  return (
    <span className="status-item mono" aria-label="Cursor image coordinates">
      <span className="status-label">X</span>
      <span>{sample.pixel.x}</span>
      <span className="status-label status-label-y">Y</span>
      <span>{sample.pixel.y}</span>
    </span>
  );
}

export function StatusBar() {
  const { state } = useEditor();
  const metadata = state.image?.metadata ?? null;
  const hasImage = metadata !== null;

  return (
    <footer className="status-bar" aria-label="Editor status">
      <div className="status-left">
        <span className="status-hint">
          {hasImage
            ? 'Wheel: Zoom to cursor • Space + Drag or Middle-Drag: Pan'
            : 'JPEG / PNG • Drag & drop an image into the viewport'}
        </span>
      </div>

      <div className="status-right">
        <CursorPositionReadout hasImage={hasImage} />

        <span className="status-divider" aria-hidden="true" />

        <span className="status-item mono" aria-label="Image dimensions">
          {metadata ? formatDimensions(metadata.width, metadata.height) : '—'}
        </span>

        <span className="status-divider" aria-hidden="true" />

        <span className="status-item mono" aria-label="Zoom percentage">
          {hasImage ? formatZoomPercent(state.transform.zoom) : '—'}
        </span>
      </div>
    </footer>
  );
}
