import { describe, expect, it } from 'vitest';
import type { LoadedImage } from '../types/editor';
import { editorReducer, initialEditorState } from './editorReducer';

function createStubImage(id: string, width: number, height: number): LoadedImage {
  return {
    id,
    metadata: {
      name: `${id}.png`,
      width,
      height,
      fileSize: 1024,
      mimeType: 'image/png',
      formatLabel: 'PNG',
    },
    element: {} as HTMLImageElement,
    sourceFile: {} as File,
    objectUrl: `blob:${id}`,
    getPixelData: () => ({}) as ImageData,
    dispose: () => {},
  };
}

describe('editorReducer', () => {
  it('fits and centers newly imported images and replaces cleanly', () => {
    let state = editorReducer(initialEditorState, {
      type: 'SET_VIEWPORT_SIZE',
      size: { width: 1000, height: 800 },
    });

    const first = createStubImage('first', 2000, 1000);
    state = editorReducer(state, { type: 'IMPORT_START' });
    expect(state.isLoading).toBe(true);

    state = editorReducer(state, { type: 'IMPORT_SUCCESS', image: first });
    expect(state.image).toBe(first);
    expect(state.isLoading).toBe(false);
    expect(state.zoomMode).toBe('fit');
    expect(state.transform.zoom).toBeCloseTo((1000 - 64) / 2000, 8);

    // Zoom in manually, then replace with a second image
    state = editorReducer(state, { type: 'ZOOM_ACTUAL_SIZE' });
    expect(state.zoomMode).toBe('custom');
    expect(state.transform.zoom).toBe(1);

    const second = createStubImage('second', 1000, 2000);
    state = editorReducer(state, { type: 'IMPORT_SUCCESS', image: second });
    expect(state.image).toBe(second);
    expect(state.zoomMode).toBe('fit');
    expect(state.transform.zoom).toBeCloseTo((800 - 64) / 2000, 8);
  });

  it('preserves existing loaded image when a subsequent import fails', () => {
    let state = editorReducer(initialEditorState, {
      type: 'SET_VIEWPORT_SIZE',
      size: { width: 800, height: 600 },
    });
    const first = createStubImage('valid', 800, 600);
    state = editorReducer(state, { type: 'IMPORT_SUCCESS', image: first });
    const prevTransform = state.transform;

    state = editorReducer(state, { type: 'IMPORT_START' });
    state = editorReducer(state, { type: 'IMPORT_ERROR', message: 'Corrupted image' });

    expect(state.image).toBe(first);
    expect(state.transform).toBe(prevTransform);
    expect(state.isLoading).toBe(false);
    expect(state.error).toBe('Corrupted image');
  });

  it('updates transform appropriately on resize in fit and custom modes', () => {
    let state = editorReducer(initialEditorState, {
      type: 'SET_VIEWPORT_SIZE',
      size: { width: 1000, height: 800 },
    });
    const img = createStubImage('photo', 4000, 3000);
    state = editorReducer(state, { type: 'IMPORT_SUCCESS', image: img });

    state = editorReducer(state, {
      type: 'SET_VIEWPORT_SIZE',
      size: { width: 500, height: 400 },
    });
    expect(state.zoomMode).toBe('fit');
    expect(state.transform.zoom).toBeCloseTo(Math.min((500 - 64) / 4000, (400 - 64) / 3000), 8);

    state = editorReducer(state, { type: 'ZOOM_STEP', direction: 'in' });
    const zoomedValue = state.transform.zoom;
    expect(state.zoomMode).toBe('custom');

    state = editorReducer(state, {
      type: 'SET_VIEWPORT_SIZE',
      size: { width: 900, height: 700 },
    });
    expect(state.transform.zoom).toBe(zoomedValue);
  });
});
