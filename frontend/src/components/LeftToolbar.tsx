import { Crosshair, Hand, Lasso, MousePointer2, Paintbrush } from 'lucide-react';
import { useEditor } from '../hooks/useEditor';

export function LeftToolbar() {
  const { state, setActiveTool } = useEditor();

  const handleSelectTool = (
    event: React.MouseEvent<HTMLButtonElement>,
    tool: 'pointer' | 'hand',
  ) => {
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

      <div className="tool-rail-group" aria-label="Planned tools">
        <button
          type="button"
          className="tool-button is-disabled"
          disabled
          aria-label="Surface Brush (planned)"
          title="Surface Brush (planned)"
        >
          <Paintbrush size={16} strokeWidth={1.75} />
        </button>

        <button
          type="button"
          className="tool-button is-disabled"
          disabled
          aria-label="Surface Selection (planned)"
          title="Surface Selection - SAM 2 (planned)"
        >
          <Lasso size={16} strokeWidth={1.75} />
        </button>

        <button
          type="button"
          className="tool-button is-disabled"
          disabled
          aria-label="Depth Probe (planned)"
          title="3D Depth Probe - MoGe-3 (planned)"
        >
          <Crosshair size={16} strokeWidth={1.75} />
        </button>
      </div>
    </aside>
  );
}
