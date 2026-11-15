import { describe, expect, it } from 'vitest';
import type { GeometryData, LoadedImage } from '../types/editor';
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

function createStubGeometry(imageId: string, withNormals = true): GeometryData {
  return {
    imageId,
    metadata: {
      id: 'abc123',
      cache_version: 1,
      image_sha256: 'deadbeef',
      model_id: 'Ruicheng/moge-3-vitl',
      resolution_level: 9,
      refine_steps: 3,
      use_fp16: true,
      max_image_size: 1024,
      original_width: 100,
      original_height: 100,
      inference_width: 100,
      inference_height: 100,
      coordinate_system: 'opencv',
      intrinsics_convention: 'normalized',
      intrinsics: [
        [0.9, 0, 0.5],
        [0, 0.9, 0.5],
        [0, 0, 1],
      ],
      depth_range: [1, 5],
      valid_pixel_count: 10000,
      has_normals: withNormals,
      cached: false,
      arrays: {},
    },
    depth: new Float32Array(0),
    points: new Float32Array(0),
    mask: new Uint8Array(0),
    normals: withNormals ? new Float32Array(0) : null,
    colors: new Uint8ClampedArray(0),
    previews: {
      depth: {} as HTMLCanvasElement,
      normals: withNormals ? ({} as HTMLCanvasElement) : null,
      mask: {} as HTMLCanvasElement,
    },
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

  it('manages geometry lifecycle, view modes, and drops stale results on image replace', () => {
    const img1 = createStubImage('img-1', 800, 600);
    const img2 = createStubImage('img-2', 800, 600);

    let state = editorReducer(initialEditorState, { type: 'IMPORT_SUCCESS', image: img1 });
    // Cannot enter geometry view modes before geometry is ready.
    state = editorReducer(state, { type: 'SET_VIEW_MODE', mode: 'depth' });
    expect(state.viewMode).toBe('rgb');

    state = editorReducer(state, { type: 'GEOMETRY_START' });
    expect(state.geometry.status).toBe('loading');

    // Replace image mid-inference, then deliver stale img-1 result.
    state = editorReducer(state, { type: 'IMPORT_SUCCESS', image: img2 });
    expect(state.geometry.status).toBe('idle');
    expect(state.viewMode).toBe('rgb');

    state = editorReducer(state, {
      type: 'GEOMETRY_SUCCESS',
      data: createStubGeometry('img-1'),
    });
    expect(state.geometry.status).toBe('idle');
    expect(state.geometry.data).toBeNull();

    // Deliver matching img-2 geometry.
    const geom2 = createStubGeometry('img-2', false);
    state = editorReducer(state, { type: 'GEOMETRY_START' });
    state = editorReducer(state, { type: 'GEOMETRY_SUCCESS', data: geom2 });
    expect(state.geometry.status).toBe('ready');

    state = editorReducer(state, { type: 'SET_VIEW_MODE', mode: 'pointcloud' });
    expect(state.viewMode).toBe('pointcloud');

    // Normals mode stays blocked if normals preview is absent.
    state = editorReducer(state, { type: 'SET_VIEW_MODE', mode: 'normals' });
    expect(state.viewMode).toBe('pointcloud');

    state = editorReducer(state, { type: 'SET_VIEW_MODE', mode: 'rgb' });
    expect(state.viewMode).toBe('rgb');
  });
});
