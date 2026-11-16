import {
  Box,
  ChevronDown,
  ChevronUp,
  Crosshair,
  Eraser,
  FolderOpen,
  GripVertical,
  Layers,
  Paintbrush,
  SunMedium,
  Trash2,
  X,
} from 'lucide-react';
import { useMemo, useState } from 'react';
import { computeFovYDegrees } from '../editor/pointCloudBuilder';
import { useEditor } from '../hooks/useEditor';
import { calibrateShadingScale } from '../painting/colorSpace';
import { PaintDocument } from '../painting/paintDocument';
import { getEffectiveFootprint, getSurfaceAccentColor } from '../painting/planarDecal';
import type { PaintMode } from '../types/editor';
import { formatAspectRatio, formatDimensions, formatFileSize } from '../utils/format';

type SidebarTab = 'inspector' | 'surfaces';

const PAINT_MODE_OPTIONS: Array<{ mode: PaintMode; label: string; title: string }> = [
  { mode: 'flat', label: 'Flat', title: 'Classic perspective paint without lighting modulation' },
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
    generateGeometry,
    cancelGeometry,
    generateAppearance,
    cancelAppearance,
    setActiveTool,
    setActiveSurface,
    removeSurface,
    reorderSurfaces,
    setFootprintExtent,
    setBrushConfig,
    setPaintMode,
    setTexturePreservation,
    setPaintOverlayOptions,
  } = useEditor();

  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('inspector');
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [dropTargetIndex, setDropTargetIndex] = useState<number | null>(null);

  const metadata = state.image?.metadata ?? null;
  const geometry = state.geometry.data;
  const geometryStatus = state.geometry.status;
  const appearance = state.appearance.data;
  const appearanceStatus = state.appearance.status;

  const surfaces = state.paint.surfaces;
  const activeSurface = PaintDocument.getActiveSurface(state.paint);
  const activePriority = activeSurface
    ? surfaces.findIndex((s) => s.id === activeSurface.id) + 1
    : 0;
  const effectiveFootprint = activeSurface ? getEffectiveFootprint(activeSurface) : null;
  const footprintSpanU = effectiveFootprint ? effectiveFootprint.maxU - effectiveFootprint.minU : 0;
  const footprintSpanV = effectiveFootprint ? effectiveFootprint.maxV - effectiveFootprint.minV : 0;
  const activeStrokeCount = activeSurface
    ? PaintDocument.getStrokesForSurface(state.paint, activeSurface.id).length
    : 0;
  const referenceSurface = activeSurface ?? surfaces[0] ?? null;
  const effectiveBrush = referenceSurface
    ? PaintDocument.getEffectiveBrushSettingsForSurface(state.paint, referenceSurface)
    : null;
  const brushConfig = state.paint.brushConfig;
  const isEraser = state.activeTool === 'eraser';

  let surfaceBadge = 'None';
  let surfaceBadgeClass = 'inspector-badge inspector-badge-muted';
  if (state.paint.isSelectingSurface) {
    surfaceBadge = 'Fitting';
    surfaceBadgeClass = 'inspector-badge inspector-badge-accent';
  } else if (activeSurface) {
    surfaceBadge = `#${activePriority} Ready`;
    surfaceBadgeClass = 'inspector-badge';
  } else if (state.paint.selectionError) {
    surfaceBadge = 'Error';
    surfaceBadgeClass = 'inspector-badge inspector-badge-danger';
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

  const activeSurfaceCalibration = useMemo(() => {
    if (!appearance || !geometry || !activeSurface) {
      return null;
    }
    return calibrateShadingScale({
      linearReference: appearance.linearReference,
      albedo: appearance.albedo,
      shading: appearance.shading,
      residual: appearance.residual,
      appearanceWidth: appearance.metadata.appearance_width,
      appearanceHeight: appearance.metadata.appearance_height,
      supportMask: activeSurface.supportMask,
      supportWidth: geometry.metadata.inference_width,
      supportHeight: geometry.metadata.inference_height,
    });
  }, [appearance, geometry, activeSurface]);

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
    <aside className="right-sidebar" aria-label="Inspector and Surfaces">
      <div className="sidebar-tabs" role="tablist" aria-label="Inspector sections">
        <button
          type="button"
          role="tab"
          aria-selected={sidebarTab === 'inspector'}
          className={`sidebar-tab-button ${sidebarTab === 'inspector' ? 'is-active' : ''}`}
          onClick={(e) => {
            e.currentTarget.blur();
            setSidebarTab('inspector');
          }}
        >
          Inspector
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={sidebarTab === 'surfaces'}
          className={`sidebar-tab-button ${sidebarTab === 'surfaces' ? 'is-active' : ''}`}
          onClick={(e) => {
            e.currentTarget.blur();
            setSidebarTab('surfaces');
          }}
        >
          <span>Surfaces</span>
          {surfaces.length > 0 && <span className="sidebar-tab-count">{surfaces.length}</span>}
        </button>
      </div>

      {sidebarTab === 'surfaces' ? (
        <section className="inspector-section sidebar-surfaces-panel">
          <div className="inspector-header">
            <h2 className="inspector-title">
              <Layers size={13} strokeWidth={1.75} />
              <span>Plane Priority</span>
            </h2>
            <span className="inspector-badge mono">{surfaces.length}</span>
          </div>

          {!geometry ? (
            <p className="inspector-empty-note">
              Generate scene geometry first to select multiple planar surfaces.
            </p>
          ) : (
            <div className="inspector-content">
              <button
                type="button"
                className={`sidebar-action-button ${
                  state.activeTool === 'select-surface' ? 'is-active' : ''
                }`}
                onClick={(e) => {
                  e.currentTarget.blur();
                  setActiveTool('select-surface');
                }}
              >
                <Crosshair size={13} strokeWidth={1.75} />
                <span>
                  {state.paint.isSelectingSurface ? 'Fitting surface...' : 'Add Surface (S)'}
                </span>
              </button>

              {surfaces.length === 0 ? (
                <p className="inspector-empty-note">
                  No surfaces selected yet. Click planes in the image to add them, then drag items
                  here to resolve overlapping intersections.
                </p>
              ) : (
                <>
                  <p className="inspector-empty-note">
                    Drag to reorder priority. Where planes overlap, higher entries (#1) own the
                    pixel.
                  </p>

                  <div className="surface-priority-list" role="list">
                    {surfaces.map((surface, index) => {
                      const accent = getSurfaceAccentColor(surface.colorIndex);
                      const isActive = surface.id === state.paint.activeSurfaceId;
                      const strokeCount = PaintDocument.getStrokesForSurface(
                        state.paint,
                        surface.id,
                      ).length;
                      const isDragging = dragIndex === index;
                      const isDropTarget =
                        dropTargetIndex === index && dragIndex !== null && dragIndex !== index;

                      return (
                        <div
                          key={surface.id}
                          role="listitem"
                          draggable
                          className={`surface-priority-item ${isActive ? 'is-active' : ''} ${
                            isDragging ? 'is-dragging' : ''
                          } ${isDropTarget ? 'is-drop-target' : ''}`}
                          onClick={() => setActiveSurface(surface.id)}
                          onDragStart={(e) => {
                            e.dataTransfer.effectAllowed = 'move';
                            e.dataTransfer.setData('text/plain', surface.id);
                            setDragIndex(index);
                            setDropTargetIndex(index);
                          }}
                          onDragOver={(e) => {
                            e.preventDefault();
                            e.dataTransfer.dropEffect = 'move';
                            if (dropTargetIndex !== index) {
                              setDropTargetIndex(index);
                            }
                          }}
                          onDrop={(e) => {
                            e.preventDefault();
                            if (dragIndex !== null && dragIndex !== index) {
                              reorderSurfaces(dragIndex, index);
                            }
                            setDragIndex(null);
                            setDropTargetIndex(null);
                          }}
                          onDragEnd={() => {
                            setDragIndex(null);
                            setDropTargetIndex(null);
                          }}
                        >
                          <span className="surface-drag-grip" aria-hidden="true">
                            <GripVertical size={13} strokeWidth={1.75} />
                          </span>

                          <span
                            className="surface-color-swatch"
                            style={{ backgroundColor: accent.hex }}
                            aria-hidden="true"
                          />

                          <div className="surface-priority-meta">
                            <div className="surface-priority-title-row">
                              <span className="mono surface-priority-rank">#{index + 1}</span>
                              <span className="surface-priority-label">{surface.label}</span>
                            </div>
                            <span className="mono surface-priority-sub">
                              {surface.inlierCount.toLocaleString()} pts
                              {strokeCount > 0 ? ` • ${strokeCount} strokes` : ''}
                            </span>
                          </div>

                          <div className="surface-item-actions">
                            <button
                              type="button"
                              className="surface-icon-button"
                              disabled={index === 0}
                              aria-label={`Move ${surface.label} up`}
                              title="Increase priority"
                              onClick={(e) => {
                                e.stopPropagation();
                                e.currentTarget.blur();
                                reorderSurfaces(index, index - 1);
                              }}
                            >
                              <ChevronUp size={13} strokeWidth={1.75} />
                            </button>
                            <button
                              type="button"
                              className="surface-icon-button"
                              disabled={index === surfaces.length - 1}
                              aria-label={`Move ${surface.label} down`}
                              title="Decrease priority"
                              onClick={(e) => {
                                e.stopPropagation();
                                e.currentTarget.blur();
                                reorderSurfaces(index, index + 1);
                              }}
                            >
                              <ChevronDown size={13} strokeWidth={1.75} />
                            </button>
                            <button
                              type="button"
                              className="surface-icon-button surface-delete-button"
                              aria-label={`Remove ${surface.label}`}
                              title="Remove surface"
                              onClick={(e) => {
                                e.stopPropagation();
                                e.currentTarget.blur();
                                removeSurface(surface.id);
                              }}
                            >
                              <Trash2 size={13} strokeWidth={1.75} />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <label className="inspector-checkbox-row">
                    <input
                      type="checkbox"
                      checked={state.paint.showSupportOverlay}
                      onChange={(e) =>
                        setPaintOverlayOptions({ showSupportOverlay: e.target.checked })
                      }
                    />
                    <span>Show plane ownership tint</span>
                  </label>
                </>
              )}
            </div>
          )}
        </section>
      ) : (
        <>
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
                No image loaded. Import a JPEG or PNG file to inspect its properties.
              </p>
            )}
          </section>

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
                    <span>Cancel Inference</span>
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
                <Crosshair size={13} strokeWidth={1.75} />
                <span>Planar Surface</span>
              </h2>
              <span className={surfaceBadgeClass}>{surfaceBadge}</span>
            </div>

            {!geometry ? (
              <p className="inspector-empty-note">
                Generate scene geometry first to select a planar surface for painting.
              </p>
            ) : (
              <div className="inspector-content">
                {state.paint.selectionError && (
                  <p className="inspector-error-note" role="alert">
                    {state.paint.selectionError}
                  </p>
                )}

                <div className="inspector-action-row">
                  <button
                    type="button"
                    className={`sidebar-action-button ${
                      state.activeTool === 'select-surface' ? 'is-active' : ''
                    }`}
                    onClick={(e) => {
                      e.currentTarget.blur();
                      setActiveTool('select-surface');
                    }}
                  >
                    <Crosshair size={13} strokeWidth={1.75} />
                    <span>
                      {state.paint.isSelectingSurface
                        ? 'Fitting...'
                        : surfaces.length > 0
                          ? 'Add Surface'
                          : 'Select Surface'}
                    </span>
                  </button>

                  {surfaces.length > 0 && (
                    <button
                      type="button"
                      className="sidebar-action-button"
                      onClick={(e) => {
                        e.currentTarget.blur();
                        setSidebarTab('surfaces');
                      }}
                    >
                      <Layers size={13} strokeWidth={1.75} />
                      <span>Priorities ({surfaces.length})</span>
                    </button>
                  )}
                </div>

                {activeSurface && effectiveFootprint ? (
                  <div className="inspector-controls-stack">
                    <dl className="metadata-grid">
                      <dt>Active Plane</dt>
                      <dd className="mono">
                        #{activePriority} {activeSurface.label}
                      </dd>

                      <dt>Support Pts</dt>
                      <dd className="mono">{activeSurface.inlierCount.toLocaleString()}</dd>

                      <dt>Fit RMS</dt>
                      <dd className="mono">{activeSurface.rmsError.toFixed(4)}</dd>

                      <dt>Footprint</dt>
                      <dd className="mono">
                        {footprintSpanU.toFixed(2)} x {footprintSpanV.toFixed(2)}
                      </dd>

                      <dt>Strokes</dt>
                      <dd className="mono">{activeStrokeCount}</dd>
                    </dl>

                    <label className="inspector-control-row">
                      <span className="inspector-control-header">
                        <span>Footprint Extent</span>
                        <span className="mono">{Math.round(activeSurface.extentScale * 100)}%</span>
                      </span>
                      <input
                        type="range"
                        className="inspector-range"
                        min={0.5}
                        max={2.5}
                        step={0.05}
                        value={activeSurface.extentScale}
                        onChange={(e) =>
                          setFootprintExtent(activeSurface.id, parseFloat(e.target.value))
                        }
                      />
                    </label>

                    <label className="inspector-checkbox-row">
                      <input
                        type="checkbox"
                        checked={state.paint.cleanPreview}
                        onChange={(e) => setPaintOverlayOptions({ cleanPreview: e.target.checked })}
                      />
                      <span>Clean final view (hide overlays)</span>
                    </label>

                    <label className="inspector-checkbox-row">
                      <input
                        type="checkbox"
                        checked={state.paint.showSupportOverlay}
                        onChange={(e) =>
                          setPaintOverlayOptions({ showSupportOverlay: e.target.checked })
                        }
                      />
                      <span>Show plane ownership tint</span>
                    </label>

                    <label className="inspector-checkbox-row">
                      <input
                        type="checkbox"
                        checked={state.paint.showFootprintPreview}
                        onChange={(e) =>
                          setPaintOverlayOptions({ showFootprintPreview: e.target.checked })
                        }
                      />
                      <span>Show paintable footprint bounds</span>
                    </label>
                  </div>
                ) : (
                  !state.paint.selectionError && (
                    <p className="inspector-empty-note">
                      Choose Select Surface (S), click one or more planes in the image, then pick
                      Brush (B) or Eraser (E) to paint.
                    </p>
                  )
                )}
              </div>
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
                <span>{isEraser ? 'Surface Eraser' : 'Surface Brush'}</span>
              </h2>
              {effectiveBrush && (
                <span className="inspector-badge mono">
                  {isEraser ? 'Erase' : brushConfig.color}
                </span>
              )}
            </div>

            {surfaces.length === 0 || !effectiveBrush ? (
              <p className="inspector-empty-note">
                Select at least one planar surface to enable perspective brush and eraser controls.
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

                <label className="inspector-control-row">
                  <span className="inspector-control-header">
                    <span>Size</span>
                    <span className="mono">
                      {Math.round(brushConfig.sizeMultiplier * 100)}% (
                      {effectiveBrush.diameter.toFixed(3)})
                    </span>
                  </span>
                  <input
                    type="range"
                    className="inspector-range"
                    min={0.2}
                    max={5.0}
                    step={0.05}
                    value={brushConfig.sizeMultiplier}
                    onChange={(e) => setBrushConfig({ sizeMultiplier: parseFloat(e.target.value) })}
                  />
                </label>

                {isEraser ? (
                  <p className="inspector-empty-note">
                    Erases paint completely down to the underlying photograph inside the brush
                    circle.
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
                Import an image first to estimate intrinsic albedo, diffuse shading, and residual
                layers for lighting-aware compositing.
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
                    <dd className="mono">
                      {appearance.metadata.cached ? 'Disk cache' : 'Inferred'}
                    </dd>

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
                      {activeSurfaceCalibration
                        ? activeSurfaceCalibration.isReliable
                          ? `${activeSurfaceCalibration.scale.toFixed(2)}x`
                          : 'Fallback (Flat)'
                        : 'Select surface'}
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

                  {activeSurfaceCalibration &&
                    !activeSurfaceCalibration.isReliable &&
                    state.paint.paintMode !== 'flat' && (
                      <p className="inspector-empty-note">
                        Active surface has low diffuse signal; compositing automatically falls back
                        to Flat Paint.
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
                    <span>Cancel Appearance</span>
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

          <section className="inspector-section inspector-section-planned">
            <div className="inspector-header">
              <h2 className="inspector-title">
                <Layers size={13} strokeWidth={1.75} />
                <span>Layers</span>
              </h2>
              <span className="inspector-badge inspector-badge-muted">Planned</span>
            </div>
            <p className="inspector-planned-note">
              Non-destructive paint layers and surface masks will be managed here.
            </p>
          </section>
        </>
      )}
    </aside>
  );
}
