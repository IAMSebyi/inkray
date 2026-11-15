import { LeftToolbar } from './components/LeftToolbar';
import { RightSidebar } from './components/RightSidebar';
import { StatusBar } from './components/StatusBar';
import { TopToolbar } from './components/TopToolbar';
import { EditorViewport } from './editor/EditorViewport';
import { useEditor } from './hooks/useEditor';
import { useImageImport } from './hooks/useImageImport';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { EditorProvider } from './state/EditorContext';
import { IMAGE_FILE_ACCEPT } from './utils/fileValidation';

function InkrayEditorShell() {
  const { fileInputRef } = useEditor();
  const { isSpacePressed } = useKeyboardShortcuts();
  const { isDragActive, handleFileInputChange, dropZoneProps } = useImageImport();

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

      <TopToolbar />

      <div className="editor-workspace">
        <LeftToolbar />
        <EditorViewport
          isSpacePressed={isSpacePressed}
          isDragActive={isDragActive}
          dropZoneProps={dropZoneProps}
        />
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
