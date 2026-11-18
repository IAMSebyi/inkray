import { Box, X } from 'lucide-react';
import { useEditor } from '../hooks/useEditor';
import { computeFovYDegrees } from '../editor/pointCloudBuilder';
import { hasGeometryDependentEdits } from '../state/editorReducer';
import { GEOMETRY_MODELS, NORMAL_MODELS, type GeometryData } from '../types/editor';
import { formatDimensions } from '../utils/format';
import { ModelSelect } from './ModelSelect';

function geometryLabel(result: GeometryData): string {
  const { model, model_id, normal_model_id, smooth_iterations } = result.metadata;
  const base = model ? GEOMETRY_MODELS[model] : model_id;
  let label = base;
  if (normal_model_id) {
    if (normal_model_id.includes('marigold-v2')) label = `${base} + V2 normals`;
    else if (normal_model_id.includes('marigold-normals')) label = `${base} + normals v1.1`;
    else label = `${base} + normals`;
  }
  return smooth_iterations && smooth_iterations > 0
    ? `${label} · smooth ${smooth_iterations}x`
    : label;
}

export function SceneGeometrySection() {
  const {
    state,
    geometryModel,
    setGeometryModel,
    modelAvailability,
    refineGeometryNormals,
    setRefineGeometryNormals,
    normalModel,
    setNormalModel,
    smoothGeometry,
    setSmoothGeometry,
    smoothingPasses,
    setSmoothingPasses,
    generateGeometry,
    cancelGeometry,
    geometryComparison,
    setGeometryComparison,
    activateGeometry,
    setViewMode,
  } = useEditor();
  const { data: active, status, error } = state.geometry;
  const { comparisonGeometryId } = state;
  const candidates = state.geometryResults.filter(
    (result) => result.metadata.id !== active?.metadata.id,
  );
  const running = status === 'loading';
  const viewingDifference = state.viewMode === 'depth-difference';
  const locked = candidates.length > 0 && hasGeometryDependentEdits(state.paint);
  const comparison = geometryComparison;
  const activeMetadata = active?.metadata;
  const fov = activeMetadata ? computeFovYDegrees(activeMetadata.intrinsics) : null;

  return (
    <section className="inspector-section" aria-label="Scene geometry">
      <div className="inspector-header">
        <h2 className="inspector-title">
          <Box size={13} strokeWidth={1.75} />
          Scene Geometry
        </h2>
      </div>
      {state.image ? (
        <div className="inspector-content">
          <ModelSelect
            label="Geometry Model"
            models={GEOMETRY_MODELS}
            value={geometryModel}
            unavailable={modelAvailability?.unavailable}
            disabled={running}
            onChange={setGeometryModel}
          />
          <label className="inspector-checkbox-row">
            <input
              type="checkbox"
              checked={refineGeometryNormals}
              disabled={running}
              onChange={(event) => setRefineGeometryNormals(event.target.checked)}
            />
            Refine depth with normals
          </label>
          {refineGeometryNormals && (
            <ModelSelect
              label="Normal Model"
              models={NORMAL_MODELS}
              value={normalModel}
              unavailable={modelAvailability?.unavailable}
              disabled={running}
              onChange={setNormalModel}
            />
          )}
          <label className="inspector-checkbox-row">
            <input
              type="checkbox"
              checked={smoothGeometry}
              disabled={running}
              onChange={(event) => setSmoothGeometry(event.target.checked)}
            />
            Smooth depth &amp; normals
          </label>
          {smoothGeometry && (
            <label className="inspector-control-row">
              <span className="inspector-control-header">
                <span>Smoothing Passes</span>
                <span className="mono">{smoothingPasses}</span>
              </span>
              <input
                type="range"
                className="inspector-range"
                min={1}
                max={6}
                step={1}
                value={smoothingPasses}
                disabled={running}
                onChange={(event) => setSmoothingPasses(parseInt(event.target.value, 10))}
              />
            </label>
          )}
          {running ? (
            <button
              type="button"
              className="sidebar-action-button toolbar-button-busy"
              onClick={cancelGeometry}
            >
              <X size={13} strokeWidth={1.75} /> Stop Waiting
            </button>
          ) : (
            <button
              type="button"
              className="sidebar-action-button"
              disabled={state.isLoading}
              onClick={() => void generateGeometry()}
            >
              <Box size={13} strokeWidth={1.75} /> Run Geometry Model
            </button>
          )}
          {running && (
            <p className="inspector-empty-note" role="status">
              Running {GEOMETRY_MODELS[geometryModel]}
              {refineGeometryNormals ? ` + ${NORMAL_MODELS[normalModel]}` : ''}
              {smoothGeometry ? ` + smooth (${smoothingPasses}x)` : ''}.
            </p>
          )}
          {error && (
            <p className="inspector-error-note" role="alert">
              {error}
            </p>
          )}

          {active && activeMetadata && (
            <div className="inspector-result">
              <dl className="metadata-grid">
                <dt>Active</dt>
                <dd title={activeMetadata.model_id}>{geometryLabel(active)}</dd>
              </dl>
              <details className="inspector-details">
                <summary>Reconstruction details</summary>
                <dl className="metadata-grid">
                  <dt>Source</dt>
                  <dd>{activeMetadata.cached ? 'Disk cache' : 'Inferred'}</dd>
                  <dt>Resolution</dt>
                  <dd>
                    {formatDimensions(
                      activeMetadata.inference_width,
                      activeMetadata.inference_height,
                    )}
                  </dd>
                  <dt>Valid Pixels</dt>
                  <dd>{activeMetadata.valid_pixel_count.toLocaleString()}</dd>
                  <dt>Depth Z</dt>
                  <dd>
                    {activeMetadata.depth_range?.map((depth) => depth.toFixed(2)).join(' to ') ??
                      'Unavailable'}
                  </dd>
                  <dt>Est. FOV Y</dt>
                  <dd>{fov === null ? 'Unavailable' : `${fov.toFixed(1)}°`}</dd>
                  {activeMetadata.smooth_iterations ? (
                    <>
                      <dt>Smoothing</dt>
                      <dd>{activeMetadata.smooth_iterations}x</dd>
                    </>
                  ) : null}
                </dl>
              </details>
            </div>
          )}

          {comparisonGeometryId && candidates.length > 0 && (
            <div className="inspector-result">
              <label className="inspector-control-row">
                <span>Compare against active</span>
                <select
                  className="inspector-select"
                  value={comparisonGeometryId}
                  onChange={(event) => setGeometryComparison(event.target.value)}
                >
                  {candidates.map((result) => (
                    <option key={result.metadata.id} value={result.metadata.id}>
                      {geometryLabel(result)}
                    </option>
                  ))}
                </select>
              </label>
              {comparison ? (
                <>
                  <dl className="metadata-grid">
                    <dt>Scale Applied</dt>
                    <dd className="mono">{comparison.scale.toPrecision(3)}×</dd>
                    <dt>Overlap</dt>
                    <dd className="mono">{(100 * comparison.overlap).toFixed(1)}%</dd>
                    <dt title="Shared pixels whose larger depth exceeds the smaller by 25%">
                      Gap &gt; 25%
                    </dt>
                    <dd className="mono">{(100 * comparison.differingFraction).toFixed(1)}%</dd>
                  </dl>
                  <button
                    type="button"
                    className={`sidebar-action-button ${viewingDifference ? 'is-active' : ''}`}
                    aria-pressed={viewingDifference}
                    onClick={() => setViewMode(viewingDifference ? 'rgb' : 'depth-difference')}
                  >
                    View Depth Difference
                  </button>
                  {viewingDifference && (
                    <div
                      className="depth-difference-legend"
                      aria-label="Aligned depth ratio for compared model"
                    >
                      <span>0.5× (nearer)</span>
                      <span>Equal</span>
                      <span>2× (farther)</span>
                    </div>
                  )}
                </>
              ) : (
                <p className="inspector-empty-note">No shared valid depth samples.</p>
              )}
              <button
                type="button"
                className="sidebar-action-button"
                disabled={running || locked}
                onClick={() => activateGeometry(comparisonGeometryId)}
              >
                Use Compared Geometry
              </button>
              {locked && (
                <p className="inspector-empty-note">
                  Existing surface edits use the active reconstruction.
                </p>
              )}
            </div>
          )}
        </div>
      ) : (
        <p className="inspector-empty-note">Import a photo to generate geometry.</p>
      )}
    </section>
  );
}
