import { describe, expect, it } from 'vitest';
import type { Size2D, ViewportRect, ViewportTransform } from '../types/editor';
import {
  clampPanToViewport,
  clientToImage,
  clientToViewport,
  computeActualSizeTransform,
  computeFitTransform,
  computeWheelZoomTarget,
  getEffectiveZoomBounds,
  getImageBoundsInViewport,
  imagePointToPixel,
  imageToClient,
  imageToViewport,
  panTransform,
  reconcileTransformOnResize,
  sampleImageAtClientPoint,
  viewportToClient,
  viewportToImage,
  zoomAtPoint,
} from './viewportTransforms';

describe('viewportTransforms', () => {
  const nonSquareRect: ViewportRect = {
    left: 44,
    top: 44,
    width: 1124,
    height: 730,
  };

  describe('screen <-> viewport <-> image round trips', () => {
    it('round-trips client -> image -> client across offset non-square viewports', () => {
      const transform: ViewportTransform = { zoom: 1.75, panX: 123.5, panY: -48.25 };
      const clientPoints = [
        { x: 44, y: 44 },
        { x: 600.25, y: 412.75 },
        { x: 1168, y: 774 },
        { x: 250, y: 190 },
      ];

      for (const clientPt of clientPoints) {
        const imagePt = clientToImage(clientPt, nonSquareRect, transform);
        const roundTripped = imageToClient(imagePt, nonSquareRect, transform);
        expect(roundTripped.x).toBeCloseTo(clientPt.x, 10);
        expect(roundTripped.y).toBeCloseTo(clientPt.y, 10);
      }
    });

    it('round-trips image -> viewport -> image with high-resolution coordinates', () => {
      const transform: ViewportTransform = { zoom: 0.125, panX: 80, panY: 42 };
      const imagePoints = [
        { x: 0, y: 0 },
        { x: 4032.5, y: 3024.25 },
        { x: 7999.9, y: 5999.1 },
      ];

      for (const imagePt of imagePoints) {
        const viewportPt = imageToViewport(imagePt, transform);
        const recovered = viewportToImage(viewportPt, transform);
        expect(recovered.x).toBeCloseTo(imagePt.x, 10);
        expect(recovered.y).toBeCloseTo(imagePt.y, 10);
      }
    });

    it('throws on invalid non-positive or non-finite zoom', () => {
      expect(() => viewportToImage({ x: 10, y: 10 }, { zoom: 0, panX: 0, panY: 0 })).toThrow(
        RangeError,
      );
      expect(() => imageToViewport({ x: 10, y: 10 }, { zoom: -1, panX: 0, panY: 0 })).toThrow(
        RangeError,
      );
      expect(() =>
        viewportToImage({ x: 10, y: 10 }, { zoom: Number.NaN, panX: 0, panY: 0 }),
      ).toThrow(RangeError);
    });
  });

  describe('coordinate mappings after zooming and panning', () => {
    it('maps image coordinates accurately after zooming', () => {
      const base: ViewportTransform = { zoom: 1, panX: 100, panY: 50 };
      const anchor = { x: 300, y: 250 };
      const zoomed = zoomAtPoint(base, anchor, 4);

      const imagePt = { x: 250, y: 180 };
      const vpPt = imageToViewport(imagePt, zoomed);
      const backToImage = viewportToImage(vpPt, zoomed);

      expect(zoomed.zoom).toBe(4);
      expect(backToImage.x).toBeCloseTo(imagePt.x, 10);
      expect(backToImage.y).toBeCloseTo(imagePt.y, 10);
      // Verify 1 image pixel delta equals 4 CSS pixels
      const nextPixelVp = imageToViewport({ x: imagePt.x + 1, y: imagePt.y + 1 }, zoomed);
      expect(nextPixelVp.x - vpPt.x).toBeCloseTo(4, 10);
      expect(nextPixelVp.y - vpPt.y).toBeCloseTo(4, 10);
    });

    it('maps coordinates accurately after panning', () => {
      const initial: ViewportTransform = { zoom: 2, panX: 120, panY: 80 };
      const panned = panTransform(initial, -45, 70);

      const sampleViewport = { x: 400, y: 300 };
      const beforeImage = viewportToImage(sampleViewport, initial);
      const afterImage = viewportToImage(
        { x: sampleViewport.x - 45, y: sampleViewport.y + 70 },
        panned,
      );

      expect(afterImage.x).toBeCloseTo(beforeImage.x, 10);
      expect(afterImage.y).toBeCloseTo(beforeImage.y, 10);
    });

    it('handles client <-> viewport offset consistently after zoom and pan', () => {
      const transform = panTransform(
        zoomAtPoint({ zoom: 0.5, panX: 200, panY: 120 }, { x: 500, y: 350 }, 2.5),
        -30,
        15,
      );
      const clientPt = { x: nonSquareRect.left + 512.5, y: nonSquareRect.top + 289.25 };
      const vpPt = clientToViewport(clientPt, nonSquareRect);
      expect(viewportToClient(vpPt, nonSquareRect)).toEqual(clientPt);

      const imgPt = clientToImage(clientPt, nonSquareRect, transform);
      expect(imgPt.x).toBeCloseTo((vpPt.x - transform.panX) / transform.zoom, 10);
      expect(imgPt.y).toBeCloseTo((vpPt.y - transform.panY) / transform.zoom, 10);
    });
  });

  describe('zoom-to-cursor invariance', () => {
    it('keeps the exact image coordinate pinned under the cursor across zoom steps', () => {
      let transform: ViewportTransform = { zoom: 0.4, panX: 180, panY: 95 };
      const cursorViewport = { x: 742.5, y: 213.75 };
      const expectedImagePoint = viewportToImage(cursorViewport, transform);

      const targetZooms = [0.75, 1.25, 2.5, 8.0, 16.0, 3.2, 0.55, 0.25];
      for (const nextZoom of targetZooms) {
        transform = zoomAtPoint(transform, cursorViewport, nextZoom);
        const actualImagePoint = viewportToImage(cursorViewport, transform);
        expect(actualImagePoint.x).toBeCloseTo(expectedImagePoint.x, 9);
        expect(actualImagePoint.y).toBeCloseTo(expectedImagePoint.y, 9);
      }
    });

    it('preserves cursor invariance with wheel zoom deltas and high-res images', () => {
      const viewportSize: Size2D = { width: 1280, height: 720 };
      const highResImage: Size2D = { width: 8000, height: 6000 };
      let transform = computeFitTransform(viewportSize, highResImage);

      const cursorClient = { x: 910, y: 340 };
      const cursorViewport = clientToViewport(cursorClient, nonSquareRect);
      const pinnedImagePt = clientToImage(cursorClient, nonSquareRect, transform);
      const bounds = getEffectiveZoomBounds(viewportSize, highResImage);

      const deltas = [-120, -120, -18, 60, -240, 120];
      for (const deltaY of deltas) {
        const nextZoom = computeWheelZoomTarget(
          transform.zoom,
          deltaY,
          0,
          false,
          viewportSize.height,
          bounds,
        );
        transform = zoomAtPoint(transform, cursorViewport, nextZoom, bounds);

        const currentImagePt = clientToImage(cursorClient, nonSquareRect, transform);
        expect(currentImagePt.x).toBeCloseTo(pinnedImagePt.x, 8);
        expect(currentImagePt.y).toBeCloseTo(pinnedImagePt.y, 8);
      }
    });
  });

  describe('fit-to-screen and 100% zoom calculations', () => {
    it('fits and centers landscape image inside portrait viewport', () => {
      const viewport: Size2D = { width: 600, height: 1000 };
      const image: Size2D = { width: 4000, height: 2000 };
      const t = computeFitTransform(viewport, image, { padding: 20 });

      const bounds = getImageBoundsInViewport(image, t);
      expect(bounds.width).toBeCloseTo(560, 8);
      expect(bounds.height).toBeCloseTo(280, 8);
      expect(bounds.width / bounds.height).toBeCloseTo(image.width / image.height, 10);
      expect(bounds.x).toBeCloseTo(20, 8);
      expect(bounds.y).toBeCloseTo((1000 - 280) / 2, 8);

      // Image center maps to viewport center
      const centerVp = imageToViewport({ x: image.width / 2, y: image.height / 2 }, t);
      expect(centerVp.x).toBeCloseTo(viewport.width / 2, 8);
      expect(centerVp.y).toBeCloseTo(viewport.height / 2, 8);
    });

    it('fits and centers portrait image inside wide non-square viewport', () => {
      const viewport: Size2D = { width: 1440, height: 640 };
      const image: Size2D = { width: 2400, height: 3600 };
      const t = computeFitTransform(viewport, image, { padding: 32 });

      const bounds = getImageBoundsInViewport(image, t);
      expect(bounds.height).toBeCloseTo(640 - 64, 8);
      expect(bounds.width / bounds.height).toBeCloseTo(2400 / 3600, 10);
      expect(bounds.y).toBeCloseTo(32, 8);
      expect(bounds.x).toBeCloseTo((1440 - bounds.width) / 2, 8);
    });

    it('fits ultra-high-resolution images without hitting minimum zoom clamp', () => {
      const viewport: Size2D = { width: 800, height: 500 };
      const megaImage: Size2D = { width: 24000, height: 18000 };
      const t = computeFitTransform(viewport, megaImage, { padding: 0 });

      const bounds = getImageBoundsInViewport(megaImage, t);
      expect(bounds.width).toBeLessThanOrEqual(viewport.width + 1e-6);
      expect(bounds.height).toBeLessThanOrEqual(viewport.height + 1e-6);
      expect(bounds.height).toBeCloseTo(500, 8);
      expect(bounds.width).toBeCloseTo(500 * (4 / 3), 8);
    });

    it('scales small images to fit and returns identity for empty viewports', () => {
      const smallFit = computeFitTransform(
        { width: 1000, height: 800 },
        { width: 200, height: 100 },
        { padding: 50 },
      );
      expect(smallFit.zoom).toBeCloseTo(4.5, 8);

      const emptyVp = computeFitTransform({ width: 0, height: 0 }, { width: 800, height: 600 });
      expect(emptyVp).toEqual({ zoom: 1, panX: 0, panY: 0 });
    });

    it('sets 100% zoom where 1 source pixel equals 1 CSS pixel', () => {
      const viewport: Size2D = { width: 1000, height: 700 };
      const image: Size2D = { width: 4000, height: 3000 };
      const fit = computeFitTransform(viewport, image);

      const actual = computeActualSizeTransform(viewport, image, fit);
      expect(actual.zoom).toBe(1);

      const p1 = imageToViewport({ x: 100, y: 100 }, actual);
      const p2 = imageToViewport({ x: 101, y: 101 }, actual);
      expect(p2.x - p1.x).toBeCloseTo(1, 10);
      expect(p2.y - p1.y).toBeCloseTo(1, 10);

      // Centered image stays centered at 100%
      const center = imageToViewport({ x: 2000, y: 1500 }, actual);
      expect(center.x).toBeCloseTo(500, 8);
      expect(center.y).toBeCloseTo(350, 8);
    });
  });

  describe('resize reconciliation and pixel sampling', () => {
    it('re-fits in fit mode and preserves center + zoom in custom mode on resize', () => {
      const image: Size2D = { width: 3000, height: 2000 };
      const initialVp: Size2D = { width: 1000, height: 800 };
      const nextVp: Size2D = { width: 1400, height: 600 };

      const fitInitial = computeFitTransform(initialVp, image);
      const resizedFit = reconcileTransformOnResize(fitInitial, initialVp, nextVp, image, 'fit');
      const expectedFit = computeFitTransform(nextVp, image);
      expect(resizedFit).toEqual(expectedFit);

      const customInitial = zoomAtPoint(fitInitial, { x: 400, y: 300 }, 2.2);
      const centerImageBefore = viewportToImage(
        { x: initialVp.width / 2, y: initialVp.height / 2 },
        customInitial,
      );
      const resizedCustom = reconcileTransformOnResize(
        customInitial,
        initialVp,
        nextVp,
        image,
        'custom',
      );
      const centerImageAfter = viewportToImage(
        { x: nextVp.width / 2, y: nextVp.height / 2 },
        resizedCustom,
      );

      expect(resizedCustom.zoom).toBe(2.2);
      expect(centerImageAfter.x).toBeCloseTo(centerImageBefore.x, 9);
      expect(centerImageAfter.y).toBeCloseTo(centerImageBefore.y, 9);
    });

    it('samples discrete pixel indices only within image bounds', () => {
      const imageSize: Size2D = { width: 100, height: 50 };
      expect(imagePointToPixel({ x: 0, y: 0 }, imageSize)).toEqual({ x: 0, y: 0 });
      expect(imagePointToPixel({ x: 99.99, y: 49.99 }, imageSize)).toEqual({ x: 99, y: 49 });
      expect(imagePointToPixel({ x: 100, y: 10 }, imageSize)).toBeNull();
      expect(imagePointToPixel({ x: -0.01, y: 10 }, imageSize)).toBeNull();

      const transform: ViewportTransform = { zoom: 2, panX: 10, panY: 20 };
      const sample = sampleImageAtClientPoint(
        { x: nonSquareRect.left + 15, y: nonSquareRect.top + 25 },
        nonSquareRect,
        transform,
        imageSize,
      );
      expect(sample.inBounds).toBe(true);
      expect(sample.imagePoint).toEqual({ x: 2.5, y: 2.5 });
      expect(sample.pixel).toEqual({ x: 2, y: 2 });
    });

    it('soft-clamps pan so at least a margin of the image remains visible', () => {
      const viewport: Size2D = { width: 800, height: 600 };
      const image: Size2D = { width: 1000, height: 800 };
      const extremePan: ViewportTransform = { zoom: 1, panX: 5000, panY: -5000 };

      const clamped = clampPanToViewport(extremePan, viewport, image, 40);
      const bounds = getImageBoundsInViewport(image, clamped);
      expect(bounds.x).toBeLessThanOrEqual(viewport.width - 40);
      expect(bounds.y + bounds.height).toBeGreaterThanOrEqual(40);
    });
  });
});
