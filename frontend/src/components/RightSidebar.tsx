import {
  Anchor,
  Box,
  Copy,
  Eraser,
  Eye,
  EyeOff,
  FolderOpen,
  GripVertical,
  ImagePlus,
  Layers,
  Paintbrush,
  Pencil,
  Plus,
  SunMedium,
  Trash2,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { getCheckerboardTileCanvas } from '../editor/canvasRenderer';
import { computeFovYDegrees } from '../editor/pointCloudBuilder';
import { useEditor } from '../hooks/useEditor';
import { ARTWORK_DRAG_MIME, type Artwork, rasterizeArtworkRgba } from '../painting/artwork';
import { type ArtworkPlacement, normalizeAngleRad } from '../painting/artworkPlacement';
import { calibrateShadingScale } from '../painting/colorSpace';
import { PaintDocument } from '../painting/paintDocument';
import type { PaintMode } from '../types/editor';
import { formatAspectRatio, formatDimensions, formatFileSize } from '../utils/format';

const SCENE_OBJECT_DRAG_MIME = 'application/x-inkray-scene-object-index';

function ArtworkMiniThumb({ artwork }: { artwork: Artwork }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    const size = 28;
    if (canvas.width !== size) canvas.width = size;
    if (canvas.height !== size) canvas.height = size;
    ctx.clearRect(0, 0, size, size);
    const pattern = ctx.createPattern(getCheckerboardTileCanvas(), 'repeat');
    if (pattern) {
      ctx.fillStyle = pattern;
      ctx.fillRect(0, 0, size, size);
    }
    const rgba = rasterizeArtworkRgba(artwork);
    const tmp = document.createElement('canvas');
    tmp.width = artwork.width;
    tmp.height = artwork.height;
    const tmpCtx = tmp.getContext('2d');
    if (tmpCtx) {
      tmpCtx.putImageData(
        new ImageData(new Uint8ClampedArray(rgba), artwork.width, artwork.height),
        0,
        0,
      );
      const scale = Math.min((size - 2) / artwork.width, (size - 2) / artwork.height);
      const w = artwork.width * scale;
      const h = artwork.height * scale;
      ctx.drawImage(tmp, (size - w) * 0.5, (size - h) * 0.5, w, h);
    }
  }, [artwork]);

  return <canvas ref={canvasRef} className="artwork-thumb-canvas" />;
}

const PAINT_MODE_OPTIONS: Array<{ mode: PaintMode; label: string; title: string }> = [
  { mode: 'flat', label: 'Flat', title: 'Classic paint without lighting modulation' },
  {
    mode: 'shading',
    label: 'Shading',
    title: 'Blend paint using estimated surface diffuse illumination',
  },
  {
    mode: 'texture',
    label: 'Texture',
    title: 'Preserve both diffuse illumination and fine surface texture',
  },
];

export function RightSidebar() {
  const {
    state,
    openFilePicker,
    openArtworkFilePicker,
    openArtworkEditor,
    createBlankArtwork,
    removeArtwork,
    placeArtworkAtImagePoint,
    updatePlacement,
    setActivePlacement,
    togglePlacementVisibility,
    removePlacement,
    reanchorPlacement,
    createBrushObject,
    setActiveBrushObject,
    toggleBrushObjectVisibility,
    removeBrushObject,
    reorderSceneObject,
    generateGeometry,
    cancelGeometry,
    generateAppearance,
    cancelAppearance,
    setActiveTool,
    setBrushConfig,
    setPaintMode,
    setTexturePreservation,
  } = useEditor();

  const placementEditRef = useRef<string | null>(null);
  const finishPlacementEdit = () => {
    placementEditRef.current = null;
  };
  const updatePlacementControl = (placement: ArtworkPlacement) => {
    updatePlacement(placement, placementEditRef.current !== placement.id);
    placementEditRef.current = placement.id;
  };

  const [draggedSceneIndex, setDraggedSceneIndex] = useState<number | null>(null);
  const [dropTargetIndex, setDropTargetIndex] = useState<number | null>(null);

  const metadata = state.image?.metadata ?? null;
  const geometry = state.geometry.data;
  const geometryStatus = state.geometry.status;
  const appearance = state.appearance.data;
  const appearanceStatus = state.appearance.status;

  const isArtworkMode = state.workspaceMode === 'artwork';
  const activeArtwork = PaintDocument.getActiveArtwork(state.paint);
  const artworks = state.paint.artworks;
  const placements = state.paint.artworkPlacements;
  const activePlacement = PaintDocument.getActivePlacement(state.paint);
  const brushObjects = state.paint.brushObjects;
  const activeBrushObject = PaintDocument.getActiveBrushObject(state.paint);
  const sceneOrder = state.paint.sceneOrder;
  const showBrushControls = isArtworkMode ? activeArtwork?.kind === 'drawn' : metadata !== null;

  const brushConfig = state.paint.brushConfig;
  const isEraser = state.activeTool === 'eraser';
  const lastAnchor = state.paint.lastAnchorMode;

  let anchorBadge = 'Idle';
  let anchorBadgeClass = 'inspector-badge inspector-badge-muted';
  if (isArtworkMode) {
    anchorBadge = activeArtwork?.kind === 'drawn' ? '2D Canvas' : 'Imported';
    anchorBadgeClass = 'inspector-badge';
  } else if (metadata) {
    if (lastAnchor === 'planar') {
      anchorBadge = 'Planar';
      anchorBadgeClass = 'inspector-badge';
    } else if (lastAnchor === 'curved') {
      anchorBadge = 'Curved';
      anchorBadgeClass = 'inspector-badge';
    } else if (lastAnchor === 'fallback-2d') {
      anchorBadge = '2D Fallback';
      anchorBadgeClass = 'inspector-badge inspector-badge-muted';
    } else if (geometry) {
      anchorBadge = 'Auto 3D';
      anchorBadgeClass = 'inspector-badge';
    } else {
      anchorBadge = '2D Mode';
    }
  }

  let geometryBadge = 'Idle';
  let geometryBadgeClass = 'inspector-badge inspector-badge-muted';
  if (geometryStatus === 'loading') {
    geometryBadge = 'Running';
    geometryBadgeClass = 'inspector-badge inspector-badge-accent';
  } else if (geometryStatus === 'ready' && geometry) {
    geometryBadge = geometry.metadata.cached ? 'Cached' : 'Ready';
    geometryBadgeClass = 'inspector-badge';
  } else if (geometryStatus === 'error') {
    geometryBadge = 'Error';
    geometryBadgeClass = 'inspector-badge inspector-badge-danger';
  }

  let appearanceBadge = 'Idle';
  let appearanceBadgeClass = 'inspector-badge inspector-badge-muted';
  if (appearanceStatus === 'loading') {
    appearanceBadge = 'Running';
    appearanceBadgeClass = 'inspector-badge inspector-badge-accent';
  } else if (appearanceStatus === 'ready' && appearance) {
    appearanceBadge = appearance.metadata.cached ? 'Cached' : 'Ready';
    appearanceBadgeClass = 'inspector-badge';
  } else if (appearanceStatus === 'error') {
    appearanceBadge = 'Error';
    appearanceBadgeClass = 'inspector-badge inspector-badge-danger';
  }

  const sceneCalibration = useMemo(() => {
    if (!appearance) {
      return null;
    }
    return calibrateShadingScale({
      linearReference: appearance.linearReference,
      albedo: appearance.albedo,
      shading: appearance.shading,
      residual: appearance.residual,
      appearanceWidth: appearance.metadata.appearance_width,
      appearanceHeight: appearance.metadata.appearance_height,
    });
  }, [appearance]);

  const totalInferencePixels = geometry
    ? geometry.metadata.inference_width * geometry.metadata.inference_height
    : 0;
  const validCoveragePct =
    geometry && totalInferencePixels > 0
      ? ((geometry.metadata.valid_pixel_count / totalInferencePixels) * 100).toFixed(1)
      : null;
  const fovYDegrees = geometry ? computeFovYDegrees(geometry.metadata.intrinsics) : null;
  const depthRangeLabel =
    geometry?.metadata.depth_range !== null && geometry?.metadata.depth_range !== undefined
      ? `${geometry.metadata.depth_range[0].toFixed(2)} - ${geometry.metadata.depth_range[1].toFixed(2)}`
      : 'Unavailable';

  return (
    <aside className="right-sidebar" aria-label="Inspector">
      <section className="inspector-section">
        <div className="inspector-header">
          <h2 className="inspector-title">Image</h2>
          {metadata && <span className="inspector-badge">{metadata.formatLabel}</span>}
        </div>

        {metadata ? (
          <div className="inspector-content">
            <dl className="metadata-grid">
              <dt>Filename</dt>
              <dd className="metadata-filename" title={metadata.name}>
                {metadata.name}
              </dd>

              <dt>Dimensions</dt>
              <dd className="mono">{formatDimensions(metadata.width, metadata.height)}</dd>

              <dt>Aspect Ratio</dt>
              <dd className="mono">{formatAspectRatio(metadata.width, metadata.height)}</dd>

              <dt>Format</dt>
              <dd className="mono">{metadata.formatLabel}</dd>

              <dt>File Size</dt>
              <dd className="mono">{formatFileSize(metadata.fileSize)}</dd>
            </dl>

            <button
              type="button"
              className="sidebar-action-button"
              onClick={(e) => {
                e.currentTarget.blur();
                openFilePicker();
              }}
              disabled={state.isLoading}
            >
              <FolderOpen size={13} strokeWidth={1.75} />
              <span>Replace Image</span>
            </button>
          </div>
        ) : (
          <p className="inspector-empty-note">
            No image loaded. Import a JPEG or PNG file to start painting.
          </p>
        )}
      </section>

      <section className="inspector-section">
        <div className="inspector-header">
          <h2 className="inspector-title">
            {isEraser ? (
              <Eraser size={13} strokeWidth={1.75} />
            ) : (
              <Paintbrush size={13} strokeWidth={1.75} />
            )}
            <span>{isEraser ? 'Eraser' : 'Brush'}</span>
          </h2>
          <span className={anchorBadgeClass}>{anchorBadge}</span>
        </div>

        {!showBrushControls ? (
          <p className="inspector-empty-note">
            {isArtworkMode
              ? 'Imported artwork is read-only. Create a blank canvas to draw new artwork.'
              : 'Import an image to enable painting tools.'}
          </p>
        ) : (
          <div className="inspector-controls-stack">
            <div className="inspector-action-row">
              <button
                type="button"
                className={`sidebar-action-button ${state.activeTool === 'brush' ? 'is-active' : ''}`}
                onClick={(e) => {
                  e.currentTarget.blur();
                  setActiveTool('brush');
                }}
              >
                <Paintbrush size={13} strokeWidth={1.75} />
                <span>Brush (B)</span>
              </button>
              <button
                type="button"
                className={`sidebar-action-button ${state.activeTool === 'eraser' ? 'is-active' : ''}`}
                onClick={(e) => {
                  e.currentTarget.blur();
                  setActiveTool('eraser');
                }}
              >
                <Eraser size={13} strokeWidth={1.75} />
                <span>Eraser (E)</span>
              </button>
            </div>

            {!isArtworkMode && (
              <div className="brush-object-bar">
                <span className="brush-object-name" title={activeBrushObject?.label ?? 'Brush 1'}>
                  Target: <strong>{activeBrushObject?.label ?? 'Auto (Brush 1)'}</strong>
                </span>
                <button
                  type="button"
                  className="sidebar-action-button brush-object-new-btn"
                  onClick={(e) => {
                    e.currentTarget.blur();
                    createBrushObject();
                  }}
                  title="Create a new separate Brush Object at top display priority"
                >
                  <Plus size={12} strokeWidth={1.75} />
                  <span>New Brush</span>
                </button>
              </div>
            )}

            <label className="inspector-control-row">
              <span className="inspector-control-header">
                <span>Size</span>
                <span className="mono">{Math.round(brushConfig.pixelSize)} px</span>
              </span>
              <input
                type="range"
                className="inspector-range"
                min={4}
                max={240}
                step={1}
                value={Math.round(brushConfig.pixelSize)}
                onChange={(e) => setBrushConfig({ pixelSize: parseFloat(e.target.value) })}
              />
            </label>

            {isEraser ? (
              <p className="inspector-empty-note">
                {isArtworkMode
                  ? 'Erases painted pixels back to transparent alpha.'
                  : 'Erases paint down to the underlying photograph inside the brush cursor.'}
              </p>
            ) : (
              <>
                <div className="inspector-color-row">
                  <span className="inspector-color-label">Color</span>
                  <div className="inspector-color-picker-wrap">
                    <input
                      type="color"
                      className="inspector-color-input"
                      aria-label="Brush color"
                      value={brushConfig.color}
                      onChange={(e) => setBrushConfig({ color: e.target.value })}
                    />
                    <span className="mono inspector-color-hex">{brushConfig.color}</span>
                  </div>
                </div>

                <label className="inspector-control-row">
                  <span className="inspector-control-header">
                    <span>Opacity</span>
                    <span className="mono">{Math.round(brushConfig.opacity * 100)}%</span>
                  </span>
                  <input
                    type="range"
                    className="inspector-range"
                    min={0.05}
                    max={1.0}
                    step={0.05}
                    value={brushConfig.opacity}
                    onChange={(e) => setBrushConfig({ opacity: parseFloat(e.target.value) })}
                  />
                </label>

                <label className="inspector-control-row">
                  <span className="inspector-control-header">
                    <span>Hardness</span>
                    <span className="mono">{Math.round(brushConfig.hardness * 100)}%</span>
                  </span>
                  <input
                    type="range"
                    className="inspector-range"
                    min={0.0}
                    max={1.0}
                    step={0.05}
                    value={brushConfig.hardness}
                    onChange={(e) => setBrushConfig({ hardness: parseFloat(e.target.value) })}
                  />
                </label>
              </>
            )}

            <p className="inspector-empty-note">
              {isArtworkMode
                ? 'Drawing directly onto transparent artwork RGBA.'
                : geometry
                  ? 'Brush automatically anchors to the surface under your cursor.'
                  : 'Painting in 2D mode. Generate geometry to follow scene surfaces.'}
            </p>
          </div>
        )}
      </section>

      <section className="inspector-section">
        <div className="inspector-header">
          <h2 className="inspector-title">
            <ImagePlus size={13} strokeWidth={1.75} />
            <span>Artwork</span>
          </h2>
          <span className="inspector-badge mono">{artworks.length}</span>
        </div>

        <div className="inspector-content">
          <div className="inspector-action-row">
            <button
              type="button"
              className="sidebar-action-button"
              onClick={(e) => {
                e.currentTarget.blur();
                createBlankArtwork();
              }}
              title="Open a blank transparent 2D canvas to draw artwork"
            >
              <Plus size={13} strokeWidth={1.75} />
              <span>New Canvas</span>
            </button>
            <button
              type="button"
              className="sidebar-action-button"
              onClick={(e) => {
                e.currentTarget.blur();
                openArtworkFilePicker();
              }}
              title="Import PNG or JPEG artwork preserving alpha"
            >
              <FolderOpen size={13} strokeWidth={1.75} />
              <span>Import Art</span>
            </button>
          </div>

          {artworks.length === 0 ? (
            <p className="inspector-empty-note">
              Create a transparent 2D drawing or import a PNG/JPEG to drag onto the photograph.
            </p>
          ) : (
            <div className="artwork-asset-list">
              {artworks.map((art) => (
                <div
                  key={art.id}
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData(ARTWORK_DRAG_MIME, art.id);
                    e.dataTransfer.effectAllowed = 'copy';
                  }}
                  className={`artwork-asset-item ${art.id === activeArtwork?.id ? 'is-selected' : ''}`}
                  title="Drag onto photograph to place automatically"
                >
                  <ArtworkMiniThumb artwork={art} />
                  <div className="artwork-asset-meta">
                    <span className="artwork-asset-name">{art.name}</span>
                    <span className="mono artwork-asset-dims">
                      {art.width}x{art.height} •{' '}
                      {art.kind === 'drawn' ? `${art.strokes.length} str` : 'PNG/JPG'}
                    </span>
                  </div>
                  <div className="artwork-asset-actions">
                    <button
                      type="button"
                      className="surface-icon-button"
                      onClick={() => openArtworkEditor(art.id)}
                      title={art.kind === 'drawn' ? 'Edit artwork canvas' : 'Inspect artwork'}
                      aria-label={`Open ${art.name}`}
                    >
                      <Pencil size={12} strokeWidth={1.75} />
                    </button>
                    <button
                      type="button"
                      className="surface-icon-button"
                      disabled={!metadata}
                      onClick={() => placeArtworkAtImagePoint(art.id)}
                      title={metadata ? 'Place artwork on photo center' : 'Import a photo first'}
                      aria-label={`Place ${art.name} on photo`}
                    >
                      <Plus size={12} strokeWidth={1.75} />
                    </button>
                    <button
                      type="button"
                      className="surface-icon-button surface-delete-button"
                      onClick={() => removeArtwork(art.id)}
                      title="Delete artwork"
                      aria-label={`Delete ${art.name}`}
                    >
                      <Trash2 size={12} strokeWidth={1.75} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {!isArtworkMode && (
        <section className="inspector-section">
          <div className="inspector-header">
            <h2 className="inspector-title">
              <Layers size={13} strokeWidth={1.75} />
              <span>Scene Objects</span>
            </h2>
            <span className="inspector-badge mono">{sceneOrder.length}</span>
          </div>

          {!metadata ? (
            <p className="inspector-empty-note">
              Import an image to stack and reorder Brush and Artwork objects.
            </p>
          ) : (
            <div className="inspector-content">
              <div className="inspector-action-row">
                <button
                  type="button"
                  className="sidebar-action-button"
                  onClick={(e) => {
                    e.currentTarget.blur();
                    createBrushObject();
                  }}
                  title="Create a new Brush Object at top display priority"
                >
                  <Plus size={13} strokeWidth={1.75} />
                  <span>New Brush Object</span>
                </button>
              </div>

              {sceneOrder.length === 0 ? (
                <p className="inspector-empty-note">
                  Draw with the Brush or place an Artwork onto the photo to control display order.
                </p>
              ) : (
                <>
                  {sceneOrder.length > 1 && (
                    <p className="inspector-empty-note">
                      Top renders in front. Drag rows to change display order.
                    </p>
                  )}

                  <div
                    className="surface-priority-list"
                    role="list"
                    aria-label="Scene objects display order"
                  >
                    {sceneOrder.map((ref, index) => {
                      const isDragging = draggedSceneIndex === index;
                      const isDropTarget =
                        dropTargetIndex === index &&
                        draggedSceneIndex !== null &&
                        draggedSceneIndex !== index;

                      if (ref.kind === 'brush') {
                        const brushObj = brushObjects.find((b) => b.id === ref.id);
                        if (!brushObj) return null;
                        const strokeCount = PaintDocument.getBrushObjectStrokeCount(
                          state.paint,
                          brushObj.id,
                        );
                        const isTargetBrush = brushObj.id === activeBrushObject?.id;
                        const isSelected = activePlacement === null && isTargetBrush;

                        return (
                          <div
                            key={`brush:${brushObj.id}`}
                            role="listitem"
                            draggable
                            onDragStart={(e) => {
                              e.dataTransfer.setData(SCENE_OBJECT_DRAG_MIME, String(index));
                              e.dataTransfer.effectAllowed = 'move';
                              setDraggedSceneIndex(index);
                            }}
                            onDragOver={(e) => {
                              if (
                                draggedSceneIndex === null &&
                                !Array.from(e.dataTransfer.types).includes(SCENE_OBJECT_DRAG_MIME)
                              ) {
                                return;
                              }
                              e.preventDefault();
                              e.dataTransfer.dropEffect = 'move';
                              if (dropTargetIndex !== index) {
                                setDropTargetIndex(index);
                              }
                            }}
                            onDrop={(e) => {
                              if (
                                draggedSceneIndex === null &&
                                !Array.from(e.dataTransfer.types).includes(SCENE_OBJECT_DRAG_MIME)
                              ) {
                                return;
                              }
                              e.preventDefault();
                              const rawIdx = e.dataTransfer.getData(SCENE_OBJECT_DRAG_MIME);
                              const fromIdx = rawIdx
                                ? Number.parseInt(rawIdx, 10)
                                : draggedSceneIndex;
                              if (
                                fromIdx !== null &&
                                Number.isFinite(fromIdx) &&
                                fromIdx !== index
                              ) {
                                reorderSceneObject(fromIdx, index);
                              }
                              setDraggedSceneIndex(null);
                              setDropTargetIndex(null);
                            }}
                            onDragEnd={() => {
                              setDraggedSceneIndex(null);
                              setDropTargetIndex(null);
                            }}
                            onClick={() => setActiveBrushObject(brushObj.id)}
                            className={`surface-priority-item ${isSelected || isTargetBrush ? 'is-active' : ''} ${isDragging ? 'is-dragging' : ''} ${isDropTarget ? 'is-drop-target' : ''} ${!brushObj.visible ? 'is-hidden' : ''}`}
                            title="Click to paint inside this Brush Object; drag to change display order"
                          >
                            <span className="surface-drag-grip" aria-hidden="true">
                              <GripVertical size={12} strokeWidth={1.75} />
                            </span>
                            <span className="scene-object-icon" aria-hidden="true">
                              <Paintbrush size={12} strokeWidth={1.75} />
                            </span>
                            <div className="surface-priority-meta">
                              <div className="surface-priority-title-row">
                                <span className="surface-priority-rank mono">#{index + 1}</span>
                                <span className="surface-priority-label">{brushObj.label}</span>
                              </div>
                              <span className="surface-priority-sub mono">
                                Brush {isTargetBrush ? '(active) ' : ''}- {strokeCount} str
                              </span>
                            </div>
                            <div className="surface-item-actions">
                              <button
                                type="button"
                                className="surface-icon-button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  toggleBrushObjectVisibility(brushObj.id);
                                }}
                                title={brushObj.visible ? 'Hide brush object' : 'Show brush object'}
                                aria-label={
                                  brushObj.visible
                                    ? `Hide ${brushObj.label}`
                                    : `Show ${brushObj.label}`
                                }
                              >
                                {brushObj.visible ? (
                                  <Eye size={12} strokeWidth={1.75} />
                                ) : (
                                  <EyeOff size={12} strokeWidth={1.75} />
                                )}
                              </button>
                              <button
                                type="button"
                                className="surface-icon-button surface-delete-button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  removeBrushObject(brushObj.id);
                                }}
                                title="Delete brush object"
                                aria-label={`Delete ${brushObj.label}`}
                              >
                                <Trash2 size={12} strokeWidth={1.75} />
                              </button>
                            </div>
                          </div>
                        );
                      }

                      const pl = placements.find((p) => p.id === ref.id);
                      if (!pl) return null;
                      const isSel = pl.id === activePlacement?.id;
                      const modeTag =
                        pl.mapping.type === 'curved'
                          ? 'Curved'
                          : pl.mapping.type === 'planar'
                            ? pl.mapping.isApproximatelyPlanar
                              ? 'Approx'
                              : 'Planar'
                            : '2D';

                      return (
                        <div
                          key={`artwork:${pl.id}`}
                          role="listitem"
                          draggable
                          onDragStart={(e) => {
                            e.dataTransfer.setData(SCENE_OBJECT_DRAG_MIME, String(index));
                            e.dataTransfer.effectAllowed = 'move';
                            setDraggedSceneIndex(index);
                          }}
                          onDragOver={(e) => {
                            if (
                              draggedSceneIndex === null &&
                              !Array.from(e.dataTransfer.types).includes(SCENE_OBJECT_DRAG_MIME)
                            ) {
                              return;
                            }
                            e.preventDefault();
                            e.dataTransfer.dropEffect = 'move';
                            if (dropTargetIndex !== index) {
                              setDropTargetIndex(index);
                            }
                          }}
                          onDrop={(e) => {
                            if (
                              draggedSceneIndex === null &&
                              !Array.from(e.dataTransfer.types).includes(SCENE_OBJECT_DRAG_MIME)
                            ) {
                              return;
                            }
                            e.preventDefault();
                            const rawIdx = e.dataTransfer.getData(SCENE_OBJECT_DRAG_MIME);
                            const fromIdx = rawIdx
                              ? Number.parseInt(rawIdx, 10)
                              : draggedSceneIndex;
                            if (fromIdx !== null && Number.isFinite(fromIdx) && fromIdx !== index) {
                              reorderSceneObject(fromIdx, index);
                            }
                            setDraggedSceneIndex(null);
                            setDropTargetIndex(null);
                          }}
                          onDragEnd={() => {
                            setDraggedSceneIndex(null);
                            setDropTargetIndex(null);
                          }}
                          onClick={() => setActivePlacement(pl.id)}
                          className={`surface-priority-item ${isSel ? 'is-active' : ''} ${isDragging ? 'is-dragging' : ''} ${isDropTarget ? 'is-drop-target' : ''} ${!pl.visible ? 'is-hidden' : ''}`}
                          title="Click to adjust transform; drag to change display order"
                        >
                          <span className="surface-drag-grip" aria-hidden="true">
                            <GripVertical size={12} strokeWidth={1.75} />
                          </span>
                          <span className="scene-object-icon" aria-hidden="true">
                            <ImagePlus size={12} strokeWidth={1.75} />
                          </span>
                          <div className="surface-priority-meta">
                            <div className="surface-priority-title-row">
                              <span className="surface-priority-rank mono">#{index + 1}</span>
                              <span className="surface-priority-label">{pl.label}</span>
                            </div>
                            <span className="surface-priority-sub mono">Artwork - {modeTag}</span>
                          </div>
                          <div className="surface-item-actions">
                            <button
                              type="button"
                              className="surface-icon-button"
                              onClick={(e) => {
                                e.stopPropagation();
                                togglePlacementVisibility(pl.id);
                              }}
                              title={
                                pl.visible ? 'Hide artwork placement' : 'Show artwork placement'
                              }
                              aria-label={pl.visible ? `Hide ${pl.label}` : `Show ${pl.label}`}
                            >
                              {pl.visible ? (
                                <Eye size={12} strokeWidth={1.75} />
                              ) : (
                                <EyeOff size={12} strokeWidth={1.75} />
                              )}
                            </button>
                            <button
                              type="button"
                              className="surface-icon-button surface-delete-button"
                              onClick={(e) => {
                                e.stopPropagation();
                                removePlacement(pl.id);
                              }}
                              title="Remove placement"
                              aria-label={`Remove ${pl.label}`}
                            >
                              <Trash2 size={12} strokeWidth={1.75} />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}

              {activePlacement && (
                <div className="placement-inspector">
                  <label className="inspector-control-row">
                    <span className="inspector-control-header">
                      <span>Scale</span>
                      <span className="mono">{Math.round(activePlacement.scale * 100)}%</span>
                    </span>
                    <input
                      type="range"
                      className="inspector-range"
                      min={0.1}
                      max={4.0}
                      step={0.05}
                      onPointerUp={finishPlacementEdit}
                      onPointerCancel={finishPlacementEdit}
                      onKeyUp={finishPlacementEdit}
                      onBlur={finishPlacementEdit}
                      value={Math.min(4.0, Math.max(0.1, activePlacement.scale))}
                      onChange={(e) =>
                        updatePlacementControl({
                          ...activePlacement,
                          scale: parseFloat(e.target.value),
                        })
                      }
                    />
                  </label>

                  <label className="inspector-control-row">
                    <span className="inspector-control-header">
                      <span>Rotation</span>
                      <span className="mono">
                        {Math.round((activePlacement.rotationRad * 180) / Math.PI)} deg
                      </span>
                    </span>
                    <input
                      type="range"
                      className="inspector-range"
                      min={-180}
                      max={180}
                      step={1}
                      onPointerUp={finishPlacementEdit}
                      onPointerCancel={finishPlacementEdit}
                      onKeyUp={finishPlacementEdit}
                      onBlur={finishPlacementEdit}
                      value={Math.round((activePlacement.rotationRad * 180) / Math.PI)}
                      onChange={(e) =>
                        updatePlacementControl({
                          ...activePlacement,
                          rotationRad: normalizeAngleRad(
                            (parseFloat(e.target.value) * Math.PI) / 180,
                          ),
                        })
                      }
                    />
                  </label>

                  <label className="inspector-control-row">
                    <span className="inspector-control-header">
                      <span>Opacity</span>
                      <span className="mono">{Math.round(activePlacement.opacity * 100)}%</span>
                    </span>
                    <input
                      type="range"
                      className="inspector-range"
                      min={0.05}
                      max={1.0}
                      step={0.05}
                      onPointerUp={finishPlacementEdit}
                      onPointerCancel={finishPlacementEdit}
                      onKeyUp={finishPlacementEdit}
                      onBlur={finishPlacementEdit}
                      value={activePlacement.opacity}
                      onChange={(e) =>
                        updatePlacementControl({
                          ...activePlacement,
                          opacity: parseFloat(e.target.value),
                        })
                      }
                    />
                  </label>

                  {activePlacement.mapping.type === 'image-2d' &&
                    activePlacement.mapping.fallbackReason && (
                      <p className="inspector-empty-note" role="status">
                        {activePlacement.mapping.fallbackReason}
                      </p>
                    )}

                  <div className="inspector-action-row">
                    <button
                      type="button"
                      className="sidebar-action-button"
                      disabled={!metadata}
                      onClick={() => reanchorPlacement(activePlacement.id)}
                      title="Re-fit local planar or curved surface mapping at placement center"
                    >
                      <Anchor size={12} strokeWidth={1.75} />
                      <span>Re-anchor</span>
                    </button>
                    <button
                      type="button"
                      className="sidebar-action-button"
                      disabled={!metadata}
                      onClick={() => placeArtworkAtImagePoint(activePlacement.artworkId)}
                      title="Create another placement using this same artwork"
                    >
                      <Copy size={12} strokeWidth={1.75} />
                      <span>Place Copy</span>
                    </button>
                  </div>

                  <div className="inspector-action-row">
                    <button
                      type="button"
                      className="sidebar-action-button"
                      onClick={() => openArtworkEditor(activePlacement.artworkId)}
                      title="Edit or inspect source artwork"
                    >
                      <Pencil size={12} strokeWidth={1.75} />
                      <span>Edit Art</span>
                    </button>
                    <button
                      type="button"
                      className="sidebar-action-button"
                      onClick={() => removePlacement(activePlacement.id)}
                      title="Remove placement from photograph"
                    >
                      <Trash2 size={12} strokeWidth={1.75} />
                      <span>Remove</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      )}

      <section className="inspector-section">
        <div className="inspector-header">
          <h2 className="inspector-title">
            <Box size={13} strokeWidth={1.75} />
            <span>Scene Geometry</span>
          </h2>
          <span className={geometryBadgeClass}>{geometryBadge}</span>
        </div>

        {!metadata ? (
          <p className="inspector-empty-note">
            Import an image first to infer monocular depth, surface normals, and 3D points.
          </p>
        ) : (
          <div className="inspector-content">
            {geometryStatus === 'error' && state.geometry.error && (
              <p className="inspector-error-note" role="alert">
                {state.geometry.error}
              </p>
            )}

            {geometryStatus === 'loading' && (
              <p className="inspector-empty-note">
                Running MoGe-3 inference on the backend. You can cancel or switch images safely.
              </p>
            )}

            {geometry && (
              <dl className="metadata-grid">
                <dt>Model</dt>
                <dd className="mono" title={geometry.metadata.model_id}>
                  {geometry.metadata.model_id}
                </dd>

                <dt>Source</dt>
                <dd className="mono">{geometry.metadata.cached ? 'Disk cache' : 'Inferred'}</dd>

                <dt>Inference</dt>
                <dd className="mono">
                  {formatDimensions(
                    geometry.metadata.inference_width,
                    geometry.metadata.inference_height,
                  )}
                </dd>

                <dt>Valid Pixels</dt>
                <dd className="mono">
                  {geometry.metadata.valid_pixel_count.toLocaleString()}
                  {validCoveragePct !== null ? ` (${validCoveragePct}%)` : ''}
                </dd>

                <dt>Depth Z</dt>
                <dd className="mono">{depthRangeLabel}</dd>

                <dt>Est. FOV Y</dt>
                <dd className="mono">
                  {fovYDegrees !== null ? `${fovYDegrees.toFixed(1)} deg` : '—'}
                </dd>

                <dt>Normals</dt>
                <dd className="mono">{geometry.metadata.has_normals ? 'Available' : 'None'}</dd>
              </dl>
            )}

            {geometryStatus === 'loading' ? (
              <button
                type="button"
                className="sidebar-action-button toolbar-button-busy"
                onClick={(e) => {
                  e.currentTarget.blur();
                  cancelGeometry();
                }}
              >
                <X size={13} strokeWidth={1.75} />
                <span>Stop Waiting</span>
              </button>
            ) : (
              <button
                type="button"
                className="sidebar-action-button"
                onClick={(e) => {
                  e.currentTarget.blur();
                  void generateGeometry();
                }}
                disabled={state.isLoading}
              >
                <Box size={13} strokeWidth={1.75} />
                <span>
                  {geometryStatus === 'error'
                    ? 'Retry Geometry'
                    : geometry
                      ? 'Regenerate Geometry'
                      : 'Generate Geometry'}
                </span>
              </button>
            )}
          </div>
        )}
      </section>

      <section className="inspector-section">
        <div className="inspector-header">
          <h2 className="inspector-title">
            <SunMedium size={13} strokeWidth={1.75} />
            <span>Intrinsic Lighting</span>
          </h2>
          <span className={appearanceBadgeClass}>{appearanceBadge}</span>
        </div>

        {!metadata ? (
          <p className="inspector-empty-note">
            Import an image first to estimate intrinsic albedo, diffuse shading, and residual layers
            for lighting-aware compositing.
          </p>
        ) : (
          <div className="inspector-content">
            {appearanceStatus === 'error' && state.appearance.error && (
              <p className="inspector-error-note" role="alert">
                {state.appearance.error}
              </p>
            )}

            {appearanceStatus === 'loading' && (
              <p className="inspector-empty-note">
                Running Marigold IID Lighting on the backend. This may briefly offload other GPU
                models to stay within VRAM limits.
              </p>
            )}

            {appearance && (
              <dl className="metadata-grid">
                <dt>Model</dt>
                <dd className="mono" title={appearance.metadata.model_id}>
                  {appearance.metadata.model_id}
                </dd>

                <dt>Source</dt>
                <dd className="mono">{appearance.metadata.cached ? 'Disk cache' : 'Inferred'}</dd>

                <dt>Resolution</dt>
                <dd className="mono">
                  {formatDimensions(
                    appearance.metadata.appearance_width,
                    appearance.metadata.appearance_height,
                  )}
                </dd>

                <dt>Space</dt>
                <dd className="mono">{appearance.metadata.prediction_space}</dd>

                <dt>Shading Scale</dt>
                <dd className="mono">
                  {sceneCalibration
                    ? sceneCalibration.isReliable
                      ? `${sceneCalibration.scale.toFixed(2)}x`
                      : 'Fallback (Flat)'
                    : '—'}
                </dd>
              </dl>
            )}

            <div className="inspector-controls-stack">
              <div className="inspector-control-row">
                <span className="inspector-control-header">
                  <span>Paint Compositing</span>
                  <span className="mono">
                    {!appearance && state.paint.paintMode !== 'flat'
                      ? `${state.paint.paintMode} (awaiting maps)`
                      : state.paint.paintMode}
                  </span>
                </span>
                <div
                  className="segmented-control sidebar-mode-control"
                  role="group"
                  aria-label="Paint compositing mode"
                >
                  {PAINT_MODE_OPTIONS.map((option) => (
                    <button
                      key={option.mode}
                      type="button"
                      className={`sidebar-mode-button ${
                        state.paint.paintMode === option.mode ? 'is-active' : ''
                      }`}
                      onClick={(e) => {
                        e.currentTarget.blur();
                        setPaintMode(option.mode);
                      }}
                      title={option.title}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>

              {state.paint.paintMode === 'texture' && (
                <label className="inspector-control-row">
                  <span className="inspector-control-header">
                    <span>Texture Detail</span>
                    <span className="mono">
                      {Math.round(state.paint.texturePreservation * 100)}%
                    </span>
                  </span>
                  <input
                    type="range"
                    className="inspector-range"
                    min={0.0}
                    max={1.0}
                    step={0.05}
                    value={state.paint.texturePreservation}
                    onChange={(e) => setTexturePreservation(parseFloat(e.target.value))}
                  />
                </label>
              )}

              {sceneCalibration &&
                !sceneCalibration.isReliable &&
                state.paint.paintMode !== 'flat' && (
                  <p className="inspector-empty-note">
                    Scene has low diffuse signal; compositing automatically falls back to Flat
                    Paint.
                  </p>
                )}
            </div>

            {appearanceStatus === 'loading' ? (
              <button
                type="button"
                className="sidebar-action-button toolbar-button-busy"
                onClick={(e) => {
                  e.currentTarget.blur();
                  cancelAppearance();
                }}
              >
                <X size={13} strokeWidth={1.75} />
                <span>Stop Waiting</span>
              </button>
            ) : (
              <button
                type="button"
                className="sidebar-action-button"
                onClick={(e) => {
                  e.currentTarget.blur();
                  void generateAppearance();
                }}
                disabled={state.isLoading}
              >
                <SunMedium size={13} strokeWidth={1.75} />
                <span>
                  {appearanceStatus === 'error'
                    ? 'Retry Appearance'
                    : appearance
                      ? 'Regenerate Appearance'
                      : 'Generate Appearance'}
                </span>
              </button>
            )}
          </div>
        )}
      </section>
    </aside>
  );
}
