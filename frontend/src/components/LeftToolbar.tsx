import { Eraser, Hand, MousePointer2, Paintbrush } from 'lucide-react';
import { useEditor } from '../hooks/useEditor';
import { PaintDocument } from '../painting/paintDocument';
import type { EditorTool } from '../types/editor';

export function LeftToolbar() {
  const { state, setActiveTool } = useEditor();

  const isArtworkMode = state.workspaceMode === 'artwork';
  const activeArtwork = PaintDocument.getActiveArtwork(state.paint);
  const hasImage = state.image !== null;
  const isPointCloud =
    (state.viewMode === 'pointcloud' || state.viewMode === 'mesh') && state.geometry.data !== null;
  const canPaint = isArtworkMode ? activeArtwork?.kind === 'drawn' : hasImage && !isPointCloud;
  const canHand = !isArtworkMode && hasImage && !isPointCloud;

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
          title={
            isArtworkMode ? 'Pointer (V)' : 'Pointer, Coordinate Inspect & Artwork Handles (V)'
          }
        >
          <MousePointer2 size={16} strokeWidth={1.75} />
        </button>

        <button
          type="button"
          className={`tool-button ${!canHand ? 'is-disabled' : ''} ${
            state.activeTool === 'hand' ? 'is-active' : ''
          }`}
          disabled={!canHand}
          onClick={(e) => handleSelectTool(e, 'hand')}
          aria-label="Pan / Hand tool"
          aria-pressed={state.activeTool === 'hand'}
          title="Hand / Pan Viewport (H, or hold Space)"
        >
          <Hand size={16} strokeWidth={1.75} />
        </button>
      </div>

      <div className="tool-rail-divider" aria-hidden="true" />

      <div className="tool-rail-group" aria-label="Painting tools">
        <button
          type="button"
          className={`tool-button ${!canPaint ? 'is-disabled' : ''} ${
            state.activeTool === 'brush' ? 'is-active' : ''
          }`}
          disabled={!canPaint}
          onClick={(e) => handleSelectTool(e, 'brush')}
          aria-label="Brush tool"
          aria-pressed={state.activeTool === 'brush'}
          title={
            isArtworkMode
              ? activeArtwork?.kind === 'drawn'
                ? 'Artwork Brush (B)'
                : 'Imported artwork is read-only'
              : !hasImage
                ? 'Brush (B) - import an image first'
                : isPointCloud
                  ? 'Brush (B) - switch to RGB view first'
                  : 'Brush (B)'
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
          aria-label="Eraser tool"
          aria-pressed={state.activeTool === 'eraser'}
          title={
            isArtworkMode
              ? activeArtwork?.kind === 'drawn'
                ? 'Artwork Eraser (E)'
                : 'Imported artwork is read-only'
              : !hasImage
                ? 'Eraser (E) - import an image first'
                : isPointCloud
                  ? 'Eraser (E) - switch to RGB view first'
                  : 'Eraser (E)'
          }
        >
          <Eraser size={16} strokeWidth={1.75} />
        </button>
      </div>
    </aside>
  );
}
