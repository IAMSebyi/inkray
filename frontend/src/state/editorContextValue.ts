import { createContext } from 'react';
import type { EditorTool, Size2D, ViewMode, ViewportTransform, ZoomMode } from '../types/editor';
import type { CursorStore } from './cursorStore';
import type { EditorState } from './editorReducer';

export interface EditorContextValue {
  state: EditorState;
  cursorStore: CursorStore;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  openFilePicker: () => void;
  importFile: (file: File) => Promise<void>;
  clearError: () => void;
  setViewportSize: (size: Size2D) => void;
  setTransform: (transform: ViewportTransform, zoomMode?: ZoomMode) => void;
  fitToScreen: () => void;
  zoomToActualSize: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  setActiveTool: (tool: EditorTool) => void;
  setViewMode: (mode: ViewMode) => void;
  generateGeometry: () => Promise<void>;
  cancelGeometry: () => void;
}

export const EditorContext = createContext<EditorContextValue | null>(null);
