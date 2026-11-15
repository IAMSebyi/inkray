import { FolderOpen, Maximize2, Minus, Plus } from 'lucide-react';
import { useEditor } from '../hooks/useEditor';
import { formatZoomPercent } from '../utils/format';

function InkrayMark() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 20 20"
      fill="none"
      aria-hidden="true"
      className="brand-mark"
    >
      <path
        d="M10 2.5L17 6.5V13.5L10 17.5L3 13.5V6.5L10 2.5Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path
        d="M10 2.5V10M10 10L17 6.5M10 10L3 6.5M10 10V17.5"
        stroke="currentColor"
        strokeWidth="1.25"
        strokeOpacity="0.65"
        strokeLinejoin="round"
      />
      <circle cx="10" cy="10" r="1.75" fill="#4c8dff" />
    </svg>
  );
}

export function TopToolbar() {
  const { state, openFilePicker, fitToScreen, zoomToActualSize, zoomIn, zoomOut } = useEditor();

  const hasImage = state.image !== null;
  const zoomLabel = hasImage ? formatZoomPercent(state.transform.zoom) : '—';

  const blurAfterAction = (event: React.MouseEvent<HTMLButtonElement>, action: () => void) => {
    event.currentTarget.blur();
    action();
  };

  return (
    <header className="top-toolbar" aria-label="Editor toolbar">
      <div className="toolbar-group toolbar-brand-group">
        <div className="brand-identity">
          <InkrayMark />
          <span className="brand-name">Inkray</span>
          <span className="brand-tagline">Paint beyond pixels</span>
        </div>

        <div className="toolbar-divider" aria-hidden="true" />

        <button
          type="button"
          className="toolbar-button toolbar-button-primary"
          onClick={(e) => blurAfterAction(e, openFilePicker)}
          title={hasImage ? 'Replace current image (O)' : 'Import JPEG or PNG image (O)'}
          disabled={state.isLoading}
        >
          <FolderOpen size={14} strokeWidth={1.75} />
          <span>{hasImage ? 'Replace Image' : 'Import Image'}</span>
        </button>
      </div>

      <div className="toolbar-group toolbar-viewport-controls">
        <div className="segmented-control" role="group" aria-label="Zoom controls">
          <button
            type="button"
            className="toolbar-icon-button"
            onClick={(e) => blurAfterAction(e, zoomOut)}
            disabled={!hasImage}
            aria-label="Zoom out"
            title="Zoom out (-)"
          >
            <Minus size={14} strokeWidth={1.75} />
          </button>

          <span className="zoom-readout" aria-label="Current zoom level">
            {zoomLabel}
          </span>

          <button
            type="button"
            className="toolbar-icon-button"
            onClick={(e) => blurAfterAction(e, zoomIn)}
            disabled={!hasImage}
            aria-label="Zoom in"
            title="Zoom in (+)"
          >
            <Plus size={14} strokeWidth={1.75} />
          </button>
        </div>

        <button
          type="button"
          className={`toolbar-button ${hasImage && state.zoomMode === 'fit' ? 'is-active' : ''}`}
          onClick={(e) => blurAfterAction(e, fitToScreen)}
          disabled={!hasImage}
          aria-label="Fit to screen"
          title="Fit image to viewport (F or 0)"
        >
          <Maximize2 size={13} strokeWidth={1.75} />
          <span>Fit</span>
        </button>

        <button
          type="button"
          className={`toolbar-button ${
            hasImage && Math.abs(state.transform.zoom - 1) < 0.0005 ? 'is-active' : ''
          }`}
          onClick={(e) => blurAfterAction(e, zoomToActualSize)}
          disabled={!hasImage}
          aria-label="100% zoom"
          title="100% actual pixels (1)"
        >
          <span>100%</span>
        </button>
      </div>
    </header>
  );
}
