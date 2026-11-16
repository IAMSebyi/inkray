import {
  Box,
  Download,
  FolderOpen,
  Maximize2,
  Minus,
  Plus,
  Redo2,
  SunMedium,
  Undo2,
  X,
} from 'lucide-react';
import { useEditor } from '../hooks/useEditor';
import { PaintDocument } from '../painting/paintDocument';
import type { ViewMode } from '../types/editor';
import { formatZoomPercent } from '../utils/format';

const VIEW_MODE_OPTIONS: Array<{ mode: ViewMode; label: string; title: string }> = [
  { mode: 'rgb', label: 'RGB', title: 'Original color image' },
  { mode: 'depth', label: 'Depth', title: 'Inferred depth map' },
  { mode: 'normals', label: 'Normals', title: 'Inferred surface normals' },
  { mode: 'mask', label: 'Mask', title: 'Valid geometry mask' },
  { mode: 'pointcloud', label: '3D', title: 'Interactive 3D point cloud' },
  { mode: 'albedo', label: 'Albedo', title: 'Intrinsic surface reflectance' },
  { mode: 'shading', label: 'Shading', title: 'Diffuse shading map (preview-calibrated)' },
  { mode: 'residual', label: 'Residual', title: 'Non-diffuse appearance residual' },
];

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
  const {
    state,
    openFilePicker,
    fitToScreen,
    zoomToActualSize,
    zoomIn,
    zoomOut,
    setViewMode,
    generateGeometry,
    cancelGeometry,
    generateAppearance,
    cancelAppearance,
    undoPaint,
    redoPaint,
    exportPng,
  } = useEditor();

  const hasImage = state.image !== null;
  const geometryData = state.geometry.data;
  const hasGeometry = geometryData !== null;
  const isGeneratingGeometry = state.geometry.status === 'loading';
  const hasAppearance = state.appearance.data !== null;
  const isGeneratingAppearance = state.appearance.status === 'loading';
  const isPointCloudMode = state.viewMode === 'pointcloud';
  const zoomDisabled = !hasImage || isPointCloudMode;
  const canUndo = PaintDocument.canUndo(state.paint);
  const canRedo = PaintDocument.canRedo(state.paint);
  const canExport = hasImage && hasGeometry && !state.isExporting;
  const zoomLabel = !hasImage
    ? '—'
    : isPointCloudMode
      ? '3D'
      : formatZoomPercent(state.transform.zoom);

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

        {isGeneratingGeometry ? (
          <button
            type="button"
            className="toolbar-button toolbar-button-busy"
            onClick={(e) => blurAfterAction(e, cancelGeometry)}
            title="Cancel running geometry inference"
          >
            <X size={14} strokeWidth={1.75} />
            <span>Cancel Inference</span>
          </button>
        ) : (
          <button
            type="button"
            className={`toolbar-button ${hasGeometry ? 'is-completed' : ''}`}
            onClick={(e) =>
              blurAfterAction(e, () => {
                void generateGeometry();
              })
            }
            disabled={!hasImage || state.isLoading}
            title="Infer 3D geometry with MoGe-3"
          >
            <Box size={14} strokeWidth={1.75} />
            <span>
              {state.geometry.status === 'error' ? 'Retry Geometry' : 'Generate Geometry'}
            </span>
          </button>
        )}

        {isGeneratingAppearance ? (
          <button
            type="button"
            className="toolbar-button toolbar-button-busy"
            onClick={(e) => blurAfterAction(e, cancelAppearance)}
            title="Cancel running appearance inference"
          >
            <X size={14} strokeWidth={1.75} />
            <span>Cancel Appearance</span>
          </button>
        ) : (
          <button
            type="button"
            className={`toolbar-button ${hasAppearance ? 'is-completed' : ''}`}
            onClick={(e) =>
              blurAfterAction(e, () => {
                void generateAppearance();
              })
            }
            disabled={!hasImage || state.isLoading}
            title="Estimate albedo, diffuse shading, and residual with Marigold IID"
          >
            <SunMedium size={14} strokeWidth={1.75} />
            <span>
              {state.appearance.status === 'error' ? 'Retry Appearance' : 'Generate Appearance'}
            </span>
          </button>
        )}

        <div className="toolbar-divider" aria-hidden="true" />

        <div className="segmented-control" role="group" aria-label="Paint history">
          <button
            type="button"
            className="toolbar-icon-button"
            onClick={(e) => blurAfterAction(e, undoPaint)}
            disabled={!canUndo}
            aria-label="Undo paint stroke"
            title="Undo paint stroke (Ctrl/Cmd+Z)"
          >
            <Undo2 size={14} strokeWidth={1.75} />
          </button>

          <button
            type="button"
            className="toolbar-icon-button"
            onClick={(e) => blurAfterAction(e, redoPaint)}
            disabled={!canRedo}
            aria-label="Redo paint stroke"
            title="Redo paint stroke (Ctrl/Cmd+Shift+Z)"
          >
            <Redo2 size={14} strokeWidth={1.75} />
          </button>
        </div>

        <button
          type="button"
          className="toolbar-button"
          onClick={(e) =>
            blurAfterAction(e, () => {
              void exportPng();
            })
          }
          disabled={!canExport}
          aria-label="Export composited PNG"
          title="Export composited PNG with geometry-aware paint"
        >
          <Download size={14} strokeWidth={1.75} />
          <span>{state.isExporting ? 'Exporting...' : 'Export PNG'}</span>
        </button>
      </div>

      <div className="toolbar-group toolbar-viewport-controls">
        <div className="segmented-control view-mode-control" role="group" aria-label="View mode">
          {VIEW_MODE_OPTIONS.map((option) => {
            const isAppearanceMode =
              option.mode === 'albedo' || option.mode === 'shading' || option.mode === 'residual';
            const optionDisabled =
              !hasImage ||
              (isAppearanceMode ? !hasAppearance : option.mode !== 'rgb' && !hasGeometry) ||
              (option.mode === 'normals' && !geometryData?.previews.normals);

            return (
              <button
                key={option.mode}
                type="button"
                className={`view-mode-button ${state.viewMode === option.mode ? 'is-active' : ''}`}
                onClick={(e) => blurAfterAction(e, () => setViewMode(option.mode))}
                disabled={optionDisabled}
                aria-pressed={state.viewMode === option.mode}
                title={option.title}
              >
                {option.label}
              </button>
            );
          })}
        </div>

        <div className="toolbar-divider" aria-hidden="true" />

        <div className="segmented-control" role="group" aria-label="Zoom controls">
          <button
            type="button"
            className="toolbar-icon-button"
            onClick={(e) => blurAfterAction(e, zoomOut)}
            disabled={zoomDisabled}
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
            disabled={zoomDisabled}
            aria-label="Zoom in"
            title="Zoom in (+)"
          >
            <Plus size={14} strokeWidth={1.75} />
          </button>
        </div>

        <button
          type="button"
          className={`toolbar-button ${hasImage && !isPointCloudMode && state.zoomMode === 'fit' ? 'is-active' : ''}`}
          onClick={(e) => blurAfterAction(e, fitToScreen)}
          disabled={zoomDisabled}
          aria-label="Fit to screen"
          title="Fit image to viewport (F or 0)"
        >
          <Maximize2 size={13} strokeWidth={1.75} />
          <span>Fit</span>
        </button>

        <button
          type="button"
          className={`toolbar-button ${
            hasImage && !isPointCloudMode && Math.abs(state.transform.zoom - 1) < 0.0005
              ? 'is-active'
              : ''
          }`}
          onClick={(e) => blurAfterAction(e, zoomToActualSize)}
          disabled={zoomDisabled}
          aria-label="100% zoom"
          title="100% actual pixels (1)"
        >
          <span>100%</span>
        </button>
      </div>
    </header>
  );
}
