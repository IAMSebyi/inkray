import { describe, expect, it } from 'vitest';
import {
  buildSyntheticSurfaceSelection,
  createSyntheticPlanarScene,
} from '../geometry/testFixtures';
import type { BrushStroke } from '../painting/brushEngine';
import type { AppearanceData, GeometryData, LoadedImage } from '../types/editor';
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

function createStubAppearance(imageId: string): AppearanceData {
  return {
    imageId,
    metadata: {
      id: 'app123',
      cache_version: 1,
      image_sha256: 'deadbeef',
      model_id: 'prs-eth/marigold-iid-lighting-v1-1',
      denoising_steps: 4,
      processing_resolution: 768,
      ensemble_size: 1,
      seed: 42,
      use_fp16: true,
      max_image_size: 1024,
      original_width: 100,
      original_height: 100,
      appearance_width: 100,
      appearance_height: 100,
      prediction_space: 'linear',
      scale_conventions: {
        albedo: 'bounded_0_1',
        shading: 'up_to_scale',
        residual: 'up_to_scale',
      },
      albedo_range: [0.1, 0.9],
      shading_range: [0.2, 0.95],
      residual_range: [0, 0.2],
      cached: false,
      arrays: {},
    },
    albedo: new Float32Array(0),
    shading: new Float32Array(0),
    residual: new Float32Array(0),
    linearReference: new Float32Array(0),
    previews: {
      albedo: {} as HTMLCanvasElement,
      shading: {} as HTMLCanvasElement,
      residual: {} as HTMLCanvasElement,
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

  it('gates surface/brush tools cleanly, tracks paint undo/redo/export, and resets on image replacement', () => {
    const img1 = createStubImage('wall-1', 24, 24);
    const geom1 = createStubGeometry('wall-1');
    const scene = createSyntheticPlanarScene({
      width: 24,
      height: 24,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [0, 0, -1],
    });
    const selection1 = buildSyntheticSurfaceSelection(scene, 10, 12);
    const selection2 = buildSyntheticSurfaceSelection(scene, 14, 12);

    // Painting tools stay disabled before geometry is ready.
    let state = editorReducer(initialEditorState, { type: 'IMPORT_SUCCESS', image: img1 });
    state = editorReducer(state, { type: 'SET_ACTIVE_TOOL', tool: 'select-surface' });
    expect(state.activeTool).toBe('pointer');

    state = editorReducer(state, { type: 'GEOMETRY_START' });
    state = editorReducer(state, { type: 'GEOMETRY_SUCCESS', data: geom1 });

    // Brush and eraser stay disabled until at least one planar surface is selected.
    state = editorReducer(state, { type: 'SET_ACTIVE_TOOL', tool: 'brush' });
    expect(state.activeTool).toBe('pointer');
    state = editorReducer(state, { type: 'SET_ACTIVE_TOOL', tool: 'eraser' });
    expect(state.activeTool).toBe('pointer');

    // Selecting surfaces keeps select-surface tool active for multi-surface picking.
    state = editorReducer(state, { type: 'SET_ACTIVE_TOOL', tool: 'select-surface' });
    state = editorReducer(state, { type: 'PAINT_SELECT_SURFACE_START' });
    expect(state.paint.isSelectingSurface).toBe(true);

    state = editorReducer(state, { type: 'PAINT_SELECT_SURFACE', selection: selection1 });
    expect(state.paint.isSelectingSurface).toBe(false);
    expect(state.activeTool).toBe('select-surface');
    state = editorReducer(state, { type: 'PAINT_SELECT_SURFACE', selection: selection2 });
    expect(state.paint.surfaces.length).toBe(2);
    expect(state.activeTool).toBe('select-surface');

    const surface1Id = state.paint.surfaces[0]?.id;
    const surface2Id = state.paint.surfaces[1]?.id;
    if (!surface1Id || !surface2Id) {
      throw new Error('Expected two selected surfaces');
    }

    // Manual switch to brush/eraser transitions viewMode to rgb.
    state = editorReducer(state, { type: 'SET_VIEW_MODE', mode: 'depth' });
    state = editorReducer(state, { type: 'SET_ACTIVE_TOOL', tool: 'brush' });
    expect(state.activeTool).toBe('brush');
    expect(state.viewMode).toBe('rgb');
    state = editorReducer(state, { type: 'SET_ACTIVE_TOOL', tool: 'eraser' });
    expect(state.activeTool).toBe('eraser');

    // Commit multi-surface stroke batch + priority reorder + undo/redo + export state flags.
    const stroke: BrushStroke = {
      id: 'test-stroke-1',
      surfaceId: surface1Id,
      mode: 'paint',
      brush: { diameter: 0.2, color: '#ff4d4d', opacity: 0.9, hardness: 0.8 },
      points: [{ u: 0, v: 0 }],
    };
    state = editorReducer(state, { type: 'PAINT_COMMIT_STROKES', strokes: [stroke] });
    expect(state.paint.strokes.length).toBe(1);

    state = editorReducer(state, { type: 'PAINT_REORDER_SURFACES', fromIndex: 1, toIndex: 0 });
    expect(state.paint.surfaces[0]?.id).toBe(surface2Id);

    state = editorReducer(state, { type: 'PAINT_UNDO' });
    expect(state.paint.surfaces[0]?.id).toBe(surface1Id);
    state = editorReducer(state, { type: 'PAINT_UNDO' });
    expect(state.paint.strokes.length).toBe(0);

    state = editorReducer(state, { type: 'PAINT_REDO' });
    expect(state.paint.strokes.length).toBe(1);

    state = editorReducer(state, { type: 'PAINT_EXPORT_START' });
    expect(state.isExporting).toBe(true);
    state = editorReducer(state, { type: 'PAINT_EXPORT_DONE', error: 'Export failed' });
    expect(state.isExporting).toBe(false);
    expect(state.error).toBe('Export failed');

    // Replacing image clears paint document and resets tool cleanly.
    const img2 = createStubImage('wall-2', 24, 24);
    state = editorReducer(state, { type: 'IMPORT_SUCCESS', image: img2 });
    expect(state.paint.surfaces.length).toBe(0);
    expect(state.paint.activeSurfaceId).toBeNull();
    expect(state.activeTool).toBe('pointer');
  });

  it('manages appearance lifecycle, ignores stale completions, and supports lighting paint modes across undo/redo', () => {
    const img1 = createStubImage('scene-1', 24, 24);
    const img2 = createStubImage('scene-2', 24, 24);
    const geom1 = createStubGeometry('scene-1');
    const app1 = createStubAppearance('scene-1');

    let state = editorReducer(initialEditorState, { type: 'IMPORT_SUCCESS', image: img1 });
    state = editorReducer(state, { type: 'GEOMETRY_START' });
    state = editorReducer(state, { type: 'GEOMETRY_SUCCESS', data: geom1 });

    // Appearance views stay gated before inference completes.
    state = editorReducer(state, { type: 'SET_VIEW_MODE', mode: 'albedo' });
    expect(state.viewMode).toBe('rgb');

    state = editorReducer(state, { type: 'APPEARANCE_START' });
    expect(state.appearance.status).toBe('loading');

    // Stale completion for scene-1 after switching to scene-2 must be ignored.
    let staleState = editorReducer(state, { type: 'IMPORT_SUCCESS', image: img2 });
    staleState = editorReducer(staleState, { type: 'APPEARANCE_SUCCESS', data: app1 });
    expect(staleState.appearance.status).toBe('idle');
    expect(staleState.appearance.data).toBeNull();

    // Valid completion unlocks albedo/shading/residual view modes while preserving geometry.
    state = editorReducer(state, { type: 'APPEARANCE_SUCCESS', data: app1 });
    expect(state.appearance.status).toBe('ready');
    expect(state.geometry.status).toBe('ready');

    state = editorReducer(state, { type: 'SET_VIEW_MODE', mode: 'shading' });
    expect(state.viewMode).toBe('shading');
    state = editorReducer(state, { type: 'SET_VIEW_MODE', mode: 'residual' });
    expect(state.viewMode).toBe('residual');

    // Switching compositing modes & texture strength works independently of stroke undo history.
    state = editorReducer(state, { type: 'PAINT_SET_MODE', mode: 'texture' });
    state = editorReducer(state, { type: 'PAINT_SET_TEXTURE_PRESERVATION', value: 0.4 });
    expect(state.paint.paintMode).toBe('texture');
    expect(state.paint.texturePreservation).toBeCloseTo(0.4, 5);

    const scene = createSyntheticPlanarScene({
      width: 24,
      height: 24,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [0, 0, -1],
    });
    state = editorReducer(state, {
      type: 'PAINT_SELECT_SURFACE',
      selection: buildSyntheticSurfaceSelection(scene, 12, 12),
    });
    const surfId = state.paint.surfaces[0]?.id;
    if (!surfId) throw new Error('Expected surface');

    state = editorReducer(state, {
      type: 'PAINT_COMMIT_STROKE',
      stroke: {
        id: 'lit-stroke-1',
        surfaceId: surfId,
        mode: 'paint',
        brush: { diameter: 0.2, color: '#22aaee', opacity: 0.9, hardness: 0.8 },
        points: [{ u: 0, v: 0 }],
      },
    });
    expect(state.paint.strokes.length).toBe(1);

    state = editorReducer(state, { type: 'PAINT_SET_MODE', mode: 'flat' });
    expect(state.paint.paintMode).toBe('flat');
    state = editorReducer(state, { type: 'PAINT_UNDO' });
    expect(state.paint.strokes.length).toBe(0);
    state = editorReducer(state, { type: 'PAINT_REDO' });
    expect(state.paint.strokes.length).toBe(1);

    // Cancelling appearance resets appearance status and returns view to RGB cleanly.
    state = editorReducer(state, { type: 'SET_VIEW_MODE', mode: 'albedo' });
    state = editorReducer(state, { type: 'APPEARANCE_CANCEL' });
    expect(state.appearance.status).toBe('idle');
    expect(state.viewMode).toBe('rgb');
  });
});
