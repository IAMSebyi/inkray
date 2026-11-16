import { Crosshair, Eraser, Hand, Lasso, MousePointer2, Paintbrush } from 'lucide-react';
import { useEditor } from '../hooks/useEditor';
import type { EditorTool } from '../types/editor';

export function LeftToolbar() {
  const { state, setActiveTool } = useEditor();

  const hasGeometry = state.geometry.status === 'ready' && state.geometry.data !== null;
  const isPointCloud = state.viewMode === 'pointcloud' && state.geometry.data !== null;
  const hasSurfaces = state.paint.surfaces.length > 0;
  const canSelectSurface = hasGeometry && !isPointCloud;
  const canPaint = canSelectSurface && hasSurfaces;

  const handleSelectTool = (event: React.MouseEvent<HTMLButtonElement>, tool: EditorTool) => {
    event.currentTarget.blur();
    setActiveTool(tool);
  };

  return (
    <aside className="left-toolbar" aria-label="Tools">
      <div className="tool-rail-group">
        <button
          type="button"
          className={`tool-button ${state.activeTool === 'pointer' ? 'is-active' : ''}`}
          onClick={(e) => handleSelectTool(e, 'pointer')}
          aria-label="Inspect / Pointer tool"
          aria-pressed={state.activeTool === 'pointer'}
          title="Pointer & Coordinate Inspect (V)"
        >
          <MousePointer2 size={16} strokeWidth={1.75} />
        </button>

        <button
          type="button"
          className={`tool-button ${state.activeTool === 'hand' ? 'is-active' : ''}`}
          onClick={(e) => handleSelectTool(e, 'hand')}
          aria-label="Pan / Hand tool"
          aria-pressed={state.activeTool === 'hand'}
          title="Hand / Pan Viewport (H, or hold Space)"
        >
          <Hand size={16} strokeWidth={1.75} />
        </button>
      </div>

      <div className="tool-rail-divider" aria-hidden="true" />

      <div className="tool-rail-group" aria-label="Surface painting tools">
        <button
          type="button"
          className={`tool-button ${!canSelectSurface ? 'is-disabled' : ''} ${
            state.activeTool === 'select-surface' ? 'is-active' : ''
          }`}
          disabled={!canSelectSurface}
          onClick={(e) => handleSelectTool(e, 'select-surface')}
          aria-label="Select Planar Surface tool"
          aria-pressed={state.activeTool === 'select-surface'}
          title={
            !hasGeometry
              ? 'Select Planar Surface (S) - generate geometry first'
              : isPointCloud
                ? 'Select Planar Surface (S) - switch to a 2D view mode first'
                : 'Select Planar Surface (S)'
          }
        >
          <Crosshair size={16} strokeWidth={1.75} />
        </button>

        <button
          type="button"
          className={`tool-button ${!canPaint ? 'is-disabled' : ''} ${
            state.activeTool === 'brush' ? 'is-active' : ''
          }`}
          disabled={!canPaint}
          onClick={(e) => handleSelectTool(e, 'brush')}
          aria-label="Surface Brush tool"
          aria-pressed={state.activeTool === 'brush'}
          title={
            !hasGeometry
              ? 'Surface Brush (B) - generate geometry first'
              : !hasSurfaces
                ? 'Surface Brush (B) - select a planar surface first'
                : 'Surface Brush (B)'
          }
        >
          <Paintbrush size={16} strokeWidth={1.75} />
        </button>

        <button
          type="button"
          className={`tool-button ${!canPaint ? 'is-disabled' : ''} ${
            state.activeTool === 'eraser' ? 'is-active' : ''
          }`}
          disabled={!canPaint}
          onClick={(e) => handleSelectTool(e, 'eraser')}
          aria-label="Surface Eraser tool"
          aria-pressed={state.activeTool === 'eraser'}
          title={
            !hasGeometry
              ? 'Surface Eraser (E) - generate geometry first'
              : !hasSurfaces
                ? 'Surface Eraser (E) - select a planar surface first'
                : 'Surface Eraser (E)'
          }
        >
          <Eraser size={16} strokeWidth={1.75} />
        </button>
      </div>

      <div className="tool-rail-divider" aria-hidden="true" />

      <div className="tool-rail-group" aria-label="Planned tools">
        <button
          type="button"
          className="tool-button is-disabled"
          disabled
          aria-label="Surface Segmentation (planned)"
          title="Surface Segmentation - SAM 2 (planned)"
        >
          <Lasso size={16} strokeWidth={1.75} />
        </button>
      </div>
    </aside>
  );
}
