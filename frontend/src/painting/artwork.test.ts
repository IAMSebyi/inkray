import { describe, expect, it } from 'vitest';
import type { LoadedImage } from '../types/editor';
import {
  ArtworkModel,
  getArtworkFootprint,
  rasterizeArtworkRgba,
  rasterizeArtworkWithLiveStroke,
  toPremultipliedRgba,
} from './artwork';
import { BrushEngine } from './brushEngine';

function createStubImportedImage(
  width: number,
  height: number,
  rgba: Uint8ClampedArray,
): LoadedImage {
  return {
    id: `stub-${width}x${height}`,
    metadata: {
      name: 'logo.png',
      width,
      height,
      fileSize: rgba.byteLength,
      mimeType: 'image/png',
      formatLabel: 'PNG',
    },
    element: {} as HTMLImageElement,
    sourceFile: {} as File,
    objectUrl: 'blob:stub',
    getPixelData: () => ({
      width,
      height,
      data: rgba,
      colorSpace: 'srgb',
    }),
    dispose: () => {},
  };
}

describe('Artwork & independent 2D asset model', () => {
  it('creates transparent blank artwork with configurable dimensions and retains editable paint/erase strokes with undo/redo', () => {
    let art = ArtworkModel.createBlank({ name: 'Badge', width: 64, height: 48 });
    expect(art.kind).toBe('drawn');
    expect(art.width).toBe(64);
    expect(art.height).toBe(48);

    const initialRgba = rasterizeArtworkRgba(art);
    expect(initialRgba.length).toBe(64 * 48 * 4);
    expect(initialRgba.every((v) => v === 0)).toBe(true);

    // Paint with the existing BrushEngine mapped directly to pixel coordinates.
    const engine = new BrushEngine();
    const fp = getArtworkFootprint(art.width, art.height);
    const liveStroke = engine.beginStroke(
      art.id,
      fp,
      { diameter: 16, color: '#ff2244', opacity: 0.9, hardness: 0.9 },
      { u: 32, v: 24 },
      'paint',
    );
    expect(liveStroke).not.toBeNull();
    engine.extendStroke({ u: 44, v: 24 }, true);

    const liveRgba = rasterizeArtworkWithLiveStroke(art, engine.getActiveStroke());
    expect(liveRgba[(24 * 64 + 32) * 4]).toBeGreaterThan(240);
    expect(liveRgba[(24 * 64 + 32) * 4 + 3]).toBeGreaterThan(200);
    // Unpainted corner remains completely transparent.
    expect(liveRgba[3]).toBe(0);

    const paintStroke = engine.commitStroke();
    if (!paintStroke) throw new Error('Expected committed paint stroke');
    art = ArtworkModel.commitDrawnStroke(art, paintStroke);
    expect(art.strokes.length).toBe(1);
    expect(ArtworkModel.canUndo(art)).toBe(true);

    // Erase the center of the stroke back to transparent alpha.
    engine.beginStroke(
      art.id,
      fp,
      { diameter: 12, color: '#000000', opacity: 1, hardness: 1 },
      { u: 32, v: 24 },
      'erase',
    );
    const eraseStroke = engine.commitStroke();
    if (!eraseStroke) throw new Error('Expected committed erase stroke');
    art = ArtworkModel.commitDrawnStroke(art, eraseStroke);

    const afterErase = rasterizeArtworkRgba(art);
    expect(afterErase[(24 * 64 + 32) * 4 + 3]).toBe(0);
    expect(afterErase[(24 * 64 + 44) * 4 + 3]).toBeGreaterThan(200);

    // Undo / redo / clear operate deterministically without raster snapshots.
    art = ArtworkModel.undoDrawn(art);
    expect(rasterizeArtworkRgba(art)[(24 * 64 + 32) * 4 + 3]).toBeGreaterThan(200);
    art = ArtworkModel.redoDrawn(art);
    expect(rasterizeArtworkRgba(art)[(24 * 64 + 32) * 4 + 3]).toBe(0);

    art = ArtworkModel.clearDrawn(art);
    expect(art.strokes.length).toBe(0);
    expect(rasterizeArtworkRgba(art).every((v) => v === 0)).toBe(true);
    art = ArtworkModel.undoDrawn(art);
    expect(art.strokes.length).toBe(2);
  });

  it('preserves native dimensions and alpha for imported images and premultiplies without boundary halos', () => {
    const rgba = new Uint8ClampedArray([
      // Bright opaque red pixel next to a transparent pixel with nonzero dirty RGB
      255, 0, 0, 255, 0, 255, 0, 0,
      // Semi-transparent blue edge pixel next to transparent black
      0, 64, 255, 128, 0, 0, 0, 0,
    ]);
    const stubImg = createStubImportedImage(2, 2, rgba);
    const imported = ArtworkModel.createFromImage(stubImg);
    expect(imported.kind).toBe('imported');
    expect(imported.width).toBe(2);
    expect(imported.height).toBe(2);

    const extracted = rasterizeArtworkRgba(imported);
    expect(extracted).toBe(rgba);

    // Premultiplication zeroes dirty RGB where alpha == 0 and scales RGB by alpha so bilinear
    // GPU interpolation followed by straight unpremultiplication stays halo-free.
    const premul = toPremultipliedRgba(extracted);
    expect([premul[0], premul[1], premul[2], premul[3]]).toEqual([255, 0, 0, 255]);
    expect([premul[4], premul[5], premul[6], premul[7]]).toEqual([0, 0, 0, 0]);

    // Simulate 50% bilinear filtering across texel (0,0) red [255,0,0,255] and (1,0) transparent [0,0,0,0].
    const filtR = 0.5 * (premul[0] ?? 0) + 0.5 * (premul[4] ?? 0);
    const filtG = 0.5 * (premul[1] ?? 0) + 0.5 * (premul[5] ?? 0);
    const filtB = 0.5 * (premul[2] ?? 0) + 0.5 * (premul[6] ?? 0);
    const filtA = 0.5 * (premul[3] ?? 0) + 0.5 * (premul[7] ?? 0);
    // Unpremultiplied RGB remains pure red [255, 0, 0] with 0 dark fringe!
    expect(filtR / filtA).toBeCloseTo(1.0, 5);
    expect(filtG / filtA).toBeCloseTo(0.0, 5);
    expect(filtB / filtA).toBeCloseTo(0.0, 5);
  });
});
