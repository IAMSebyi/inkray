import { ArrowLeft, Eraser, FolderOpen, Paintbrush, Plus, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { getCheckerboardTileCanvas, syncCanvasBackingSize } from '../editor/canvasRenderer';
import { computeFitTransform } from '../editor/viewportTransforms';
import { useDevicePixelRatio } from '../hooks/useDevicePixelRatio';
import { useEditor } from '../hooks/useEditor';
import { useElementSize } from '../hooks/useElementSize';
import {
  ARTWORK_DRAG_MIME,
  clampArtworkDimension,
  getArtworkFootprint,
  rasterizeArtworkRgba,
  rasterizeArtworkWithLiveStroke,
} from '../painting/artwork';
import { PaintDocument } from '../painting/paintDocument';
import type { Point2D } from '../types/editor';

export function ArtworkCanvasView() {
  const {
    state,
    brushEngineRef,
    returnToPhoto,
    createBlankArtwork,
    openArtworkFilePicker,
    resizeActiveArtwork,
    commitArtworkStroke,
    clearActiveArtwork,
    placeArtworkAtImagePoint,
  } = useEditor();

  const dpr = useDevicePixelRatio();
  const stageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });
  useElementSize(stageRef, setStageSize);

  const activeArtwork = PaintDocument.getActiveArtwork(state.paint);
  const isDrawn = activeArtwork?.kind === 'drawn';

  const [dimWidthInput, setDimWidthInput] = useState(String(activeArtwork?.width ?? 512));
  const [dimHeightInput, setDimHeightInput] = useState(String(activeArtwork?.height ?? 512));

  useEffect(() => {
    if (activeArtwork) {
      setDimWidthInput(String(activeArtwork.width));
      setDimHeightInput(String(activeArtwork.height));
    }
  }, [activeArtwork]);

  const drawingSessionRef = useRef<{
    active: boolean;
    pointerId: number | null;
  }>({
    active: false,
    pointerId: null,
  });
  const cursorStagePtRef = useRef<Point2D | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  const displayTransform = useMemo(() => {
    if (!activeArtwork || stageSize.width <= 0 || stageSize.height <= 0) {
      return { zoom: 1, panX: 0, panY: 0 };
    }
    return computeFitTransform(stageSize, {
      width: activeArtwork.width,
      height: activeArtwork.height,
    });
  }, [activeArtwork, stageSize]);

  const drawArtworkCanvasNow = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !activeArtwork || stageSize.width <= 0 || stageSize.height <= 0) {
      return;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    const safeDpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
    syncCanvasBackingSize(canvas, stageSize, safeDpr);

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.setTransform(safeDpr, 0, 0, safeDpr, 0, 0);

    const drawW = activeArtwork.width * displayTransform.zoom;
    const drawH = activeArtwork.height * displayTransform.zoom;
    const drawX = displayTransform.panX;
    const drawY = displayTransform.panY;

    // Transparency checkerboard is purely visual and never enters the stored artwork RGBA.
    const pattern = ctx.createPattern(getCheckerboardTileCanvas(), 'repeat');
    if (pattern) {
      ctx.fillStyle = pattern;
      ctx.fillRect(drawX, drawY, drawW, drawH);
    }

    const liveStroke =
      activeArtwork.kind === 'drawn' ? brushEngineRef.current.getActiveStroke() : null;
    const rgba =
      activeArtwork.kind === 'drawn'
        ? rasterizeArtworkWithLiveStroke(activeArtwork, liveStroke)
        : rasterizeArtworkRgba(activeArtwork);

    let offscreen = offscreenCanvasRef.current;
    if (!offscreen) {
      offscreen = document.createElement('canvas');
      offscreenCanvasRef.current = offscreen;
    }
    if (offscreen.width !== activeArtwork.width) {
      offscreen.width = activeArtwork.width;
    }
    if (offscreen.height !== activeArtwork.height) {
      offscreen.height = activeArtwork.height;
    }
    const offCtx = offscreen.getContext('2d');
    if (offCtx) {
      offCtx.putImageData(
        new ImageData(new Uint8ClampedArray(rgba), activeArtwork.width, activeArtwork.height),
        0,
        0,
      );
      ctx.imageSmoothingEnabled = displayTransform.zoom < 3;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(offscreen, drawX, drawY, drawW, drawH);
    }

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.24)';
    ctx.lineWidth = 1;
    ctx.strokeRect(drawX, drawY, drawW, drawH);

    if (
      isDrawn &&
      (state.activeTool === 'brush' || state.activeTool === 'eraser') &&
      cursorStagePtRef.current
    ) {
      const rCss = Math.max(2, state.paint.brushConfig.pixelSize * 0.5 * displayTransform.zoom);
      const { x, y } = cursorStagePtRef.current;
      ctx.beginPath();
      ctx.arc(x, y, rCss, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.72)';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(x, y, rCss, 0, Math.PI * 2);
      ctx.strokeStyle =
        state.activeTool === 'eraser' ? 'rgba(255, 196, 140, 0.96)' : 'rgba(255, 255, 255, 0.96)';
      ctx.lineWidth = 1.25;
      ctx.stroke();
    }

    ctx.restore();
  }, [
    activeArtwork,
    brushEngineRef,
    displayTransform,
    dpr,
    isDrawn,
    stageSize,
    state.activeTool,
    state.paint.brushConfig.pixelSize,
  ]);

  const scheduleRedraw = useCallback(() => {
    if (rafRef.current !== null) {
      return;
    }
    rafRef.current = window.requestAnimationFrame(() => {
      rafRef.current = null;
      drawArtworkCanvasNow();
    });
  }, [drawArtworkCanvasNow]);

  useLayoutEffect(() => {
    drawArtworkCanvasNow();
  }, [drawArtworkCanvasNow, activeArtwork?.revision, state.paint.brushConfig]);

  // Update the compact transparent artwork preview thumbnail.
  useEffect(() => {
    const thumb = previewCanvasRef.current;
    if (!thumb || !activeArtwork) {
      return;
    }
    const ctx = thumb.getContext('2d');
    if (!ctx) {
      return;
    }
    const size = 36;
    if (thumb.width !== size) thumb.width = size;
    if (thumb.height !== size) thumb.height = size;
    ctx.clearRect(0, 0, size, size);
    const pattern = ctx.createPattern(getCheckerboardTileCanvas(), 'repeat');
    if (pattern) {
      ctx.fillStyle = pattern;
      ctx.fillRect(0, 0, size, size);
    }

    const rgba = rasterizeArtworkRgba(activeArtwork);
    const tmp = document.createElement('canvas');
    tmp.width = activeArtwork.width;
    tmp.height = activeArtwork.height;
    const tmpCtx = tmp.getContext('2d');
    if (tmpCtx) {
      tmpCtx.putImageData(
        new ImageData(new Uint8ClampedArray(rgba), activeArtwork.width, activeArtwork.height),
        0,
        0,
      );
      const scale = Math.min((size - 4) / activeArtwork.width, (size - 4) / activeArtwork.height);
      const w = activeArtwork.width * scale;
      const h = activeArtwork.height * scale;
      ctx.drawImage(tmp, (size - w) * 0.5, (size - h) * 0.5, w, h);
    }
  }, [activeArtwork]);

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  const stagePointToArtworkPixel = (clientX: number, clientY: number): Point2D | null => {
    const canvas = canvasRef.current;
    if (!canvas || !activeArtwork) {
      return null;
    }
    const rect = canvas.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    cursorStagePtRef.current = { x: sx, y: sy };
    const artX = (sx - displayTransform.panX) / Math.max(1e-6, displayTransform.zoom);
    const artY = (sy - displayTransform.panY) / Math.max(1e-6, displayTransform.zoom);
    return {
      x: artX,
      y: artY,
    };
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (
      event.button !== 0 ||
      !activeArtwork ||
      activeArtwork.kind !== 'drawn' ||
      (state.activeTool !== 'brush' && state.activeTool !== 'eraser')
    ) {
      return;
    }
    const pt = stagePointToArtworkPixel(event.clientX, event.clientY);
    if (
      !pt ||
      pt.x < 0 ||
      pt.y < 0 ||
      pt.x >= activeArtwork.width ||
      pt.y >= activeArtwork.height
    ) {
      return;
    }
    const footprint = getArtworkFootprint(activeArtwork.width, activeArtwork.height);
    const mode = state.activeTool === 'eraser' ? 'erase' : 'paint';
    const started = brushEngineRef.current.beginStroke(
      activeArtwork.id,
      footprint,
      {
        diameter: state.paint.brushConfig.pixelSize,
        color: state.paint.brushConfig.color,
        opacity: state.paint.brushConfig.opacity,
        hardness: state.paint.brushConfig.hardness,
      },
      { u: pt.x, v: pt.y },
      mode,
    );
    if (!started) {
      return;
    }
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drawingSessionRef.current = { active: true, pointerId: event.pointerId };
    scheduleRedraw();
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const pt = stagePointToArtworkPixel(event.clientX, event.clientY);
    if (
      drawingSessionRef.current.active &&
      drawingSessionRef.current.pointerId === event.pointerId &&
      pt
    ) {
      brushEngineRef.current.extendStroke({ u: pt.x, v: pt.y });
      scheduleRedraw();
      return;
    }
    if (isDrawn && (state.activeTool === 'brush' || state.activeTool === 'eraser')) {
      scheduleRedraw();
    }
  };

  const finishStroke = (event: React.PointerEvent<HTMLCanvasElement>, cancelled: boolean) => {
    if (
      !drawingSessionRef.current.active ||
      drawingSessionRef.current.pointerId !== event.pointerId
    ) {
      return;
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    drawingSessionRef.current = { active: false, pointerId: null };
    if (cancelled) {
      brushEngineRef.current.cancelStroke();
      scheduleRedraw();
      return;
    }
    const committed = brushEngineRef.current.commitStroke();
    if (committed) {
      commitArtworkStroke(committed);
    } else {
      scheduleRedraw();
    }
  };

  const handleApplyDimensions = (event: React.SyntheticEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!activeArtwork || activeArtwork.kind !== 'drawn') {
      return;
    }
    const w = clampArtworkDimension(parseInt(dimWidthInput, 10));
    const h = clampArtworkDimension(parseInt(dimHeightInput, 10));
    setDimWidthInput(String(w));
    setDimHeightInput(String(h));
    resizeActiveArtwork(w, h);
  };

  return (
    <section className="artwork-workspace" aria-label="Artwork canvas workspace">
      <div className="artwork-header-bar">
        <div className="artwork-bar-group">
          <button
            type="button"
            className="toolbar-button"
            onClick={() => returnToPhoto()}
            title="Return to photograph viewport"
          >
            <ArrowLeft size={14} strokeWidth={1.75} />
            <span>Return to Photo</span>
          </button>

          {activeArtwork && (
            <div
              className="artwork-preview-pill"
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData(ARTWORK_DRAG_MIME, activeArtwork.id);
                e.dataTransfer.effectAllowed = 'copy';
              }}
              title="Live transparent artwork preview (drag to place or use Place on Photo)"
            >
              <canvas ref={previewCanvasRef} className="artwork-preview-thumb" />
              <div className="artwork-preview-meta">
                <span className="artwork-preview-name">{activeArtwork.name}</span>
                <span className="mono artwork-preview-dims">
                  {activeArtwork.width} x {activeArtwork.height}
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="artwork-bar-group">
          {activeArtwork?.kind === 'drawn' && (
            <form className="artwork-dims-form" onSubmit={handleApplyDimensions}>
              <label className="artwork-dim-field">
                <span>W</span>
                <input
                  type="number"
                  min={16}
                  max={4096}
                  value={dimWidthInput}
                  onChange={(e) => setDimWidthInput(e.target.value)}
                  aria-label="Canvas width"
                />
              </label>
              <span className="artwork-dim-x">x</span>
              <label className="artwork-dim-field">
                <span>H</span>
                <input
                  type="number"
                  min={16}
                  max={4096}
                  value={dimHeightInput}
                  onChange={(e) => setDimHeightInput(e.target.value)}
                  aria-label="Canvas height"
                />
              </label>
              <button type="submit" className="toolbar-button" title="Resize drawn artwork canvas">
                Resize
              </button>
            </form>
          )}

          <button
            type="button"
            className="toolbar-button"
            onClick={() => createBlankArtwork()}
            title="Create a new blank transparent 512x512 artwork"
          >
            <Plus size={14} strokeWidth={1.75} />
            <span>New Blank</span>
          </button>

          <button
            type="button"
            className="toolbar-button"
            onClick={() => openArtworkFilePicker()}
            title="Import PNG or JPEG artwork preserving transparency"
          >
            <FolderOpen size={14} strokeWidth={1.75} />
            <span>Import Image</span>
          </button>

          {activeArtwork?.kind === 'drawn' && (
            <button
              type="button"
              className="toolbar-button"
              onClick={() => clearActiveArtwork()}
              disabled={activeArtwork.strokes.length === 0}
              title="Clear drawn strokes cleanly to transparent canvas"
            >
              <Trash2 size={14} strokeWidth={1.75} />
              <span>Clear</span>
            </button>
          )}

          {activeArtwork && (
            <button
              type="button"
              className="toolbar-button toolbar-button-primary"
              disabled={!state.image}
              onClick={() => placeArtworkAtImagePoint(activeArtwork.id)}
              title={
                state.image
                  ? 'Place artwork automatically onto the photograph'
                  : 'Import a photograph first to place artwork'
              }
            >
              {state.activeTool === 'eraser' ? (
                <Eraser size={14} strokeWidth={1.75} />
              ) : (
                <Paintbrush size={14} strokeWidth={1.75} />
              )}
              <span>Place on Photo</span>
            </button>
          )}
        </div>
      </div>

      <div
        ref={stageRef}
        className={`artwork-canvas-stage ${
          isDrawn && (state.activeTool === 'brush' || state.activeTool === 'eraser')
            ? 'cursor-brush'
            : 'cursor-default'
        }`}
      >
        <canvas
          ref={canvasRef}
          className="artwork-stage-canvas"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={(e) => finishStroke(e, false)}
          onPointerCancel={(e) => finishStroke(e, true)}
          onPointerLeave={() => {
            cursorStagePtRef.current = null;
            scheduleRedraw();
          }}
        />
      </div>
    </section>
  );
}
