import { ImagePlus } from 'lucide-react';
import { useEditor } from '../hooks/useEditor';

export function EmptyViewportState() {
  const { openFilePicker } = useEditor();

  return (
    <div className="empty-viewport-state">
      <div className="empty-state-panel">
        <div className="empty-state-icon" aria-hidden="true">
          <ImagePlus size={22} strokeWidth={1.6} />
        </div>
        <h1 className="empty-state-title">Open a photograph to begin</h1>
        <p className="empty-state-description">
          Drag and drop a JPEG or PNG image into the viewport, or browse from your computer.
        </p>
        <div className="empty-state-actions">
          <button
            type="button"
            className="toolbar-button toolbar-button-primary"
            onClick={(e) => {
              e.currentTarget.blur();
              openFilePicker();
            }}
          >
            Import Image
          </button>
          <span className="empty-state-formats">JPEG, PNG</span>
        </div>
      </div>
    </div>
  );
}
