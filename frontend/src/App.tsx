import { ArtworkCanvasView } from './components/ArtworkCanvasView';
import { LeftToolbar } from './components/LeftToolbar';
import { RightSidebar } from './components/RightSidebar';
import { StatusBar } from './components/StatusBar';
import { TopToolbar } from './components/TopToolbar';
import { EditorViewport } from './editor/EditorViewport';
import { ViewportErrorBoundary } from './editor/ViewportErrorBoundary';
import { useEditor } from './hooks/useEditor';
import { useImageImport } from './hooks/useImageImport';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { EditorProvider } from './state/EditorContext';
import { IMAGE_FILE_ACCEPT } from './utils/fileValidation';

function InkrayEditorShell() {
  const { state, fileInputRef, artworkFileInputRef, importArtworkFile } = useEditor();
  const { isSpacePressed } = useKeyboardShortcuts();
  const { isDragActive, handleFileInputChange, dropZoneProps } = useImageImport();

  const handleArtworkFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (file) {
      void importArtworkFile(file);
    }
  };

  return (
    <div className="editor-shell">
      <input
        ref={fileInputRef}
        type="file"
        accept={IMAGE_FILE_ACCEPT}
        className="visually-hidden-input"
        aria-hidden="true"
        tabIndex={-1}
        onChange={handleFileInputChange}
      />

      <input
        ref={artworkFileInputRef}
        type="file"
        accept={IMAGE_FILE_ACCEPT}
        className="visually-hidden-input"
        aria-hidden="true"
        tabIndex={-1}
        onChange={handleArtworkFileInputChange}
      />

      <TopToolbar />

      <div className="editor-workspace">
        <LeftToolbar />
        {state.workspaceMode === 'artwork' && <ArtworkCanvasView />}
        <ViewportErrorBoundary hidden={state.workspaceMode === 'artwork'}>
          <EditorViewport
            isSpacePressed={isSpacePressed}
            isDragActive={isDragActive}
            dropZoneProps={dropZoneProps}
          />
        </ViewportErrorBoundary>
        <RightSidebar />
      </div>

      <StatusBar />
    </div>
  );
}

export function App() {
  return (
    <EditorProvider>
      <InkrayEditorShell />
    </EditorProvider>
  );
}

export default App;
