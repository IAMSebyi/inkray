import { RotateCcw } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ImageMeshBuilder } from '../geometry/imageMeshBuilder';
import type { GeometryData } from '../types/editor';
import {
  DEFAULT_POINT_BUDGET,
  buildPointCloudAttributes,
  computeInitialCameraPose,
  computePointCloudStride,
} from './pointCloudBuilder';
import { pickPointInObject3D } from './pointCloudPicking';

interface PivotMarkerHandle {
  group: THREE.Group;
  dispose: () => void;
}

function createPivotMarker(): PivotMarkerHandle {
  const group = new THREE.Group();
  group.visible = false;
  group.renderOrder = 10;

  const sphereGeom = new THREE.SphereGeometry(0.18, 12, 10);
  const sphereMat = new THREE.MeshBasicMaterial({ color: 0xfef3c7, depthTest: false });
  const centerDot = new THREE.Mesh(sphereGeom, sphereMat);
  centerDot.renderOrder = 10;

  const axesGeom = new THREE.BufferGeometry();
  axesGeom.setAttribute(
    'position',
    new THREE.Float32BufferAttribute([-1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1], 3),
  );
  axesGeom.setAttribute(
    'color',
    new THREE.Float32BufferAttribute(
      [
        0.94, 0.27, 0.27, 0.94, 0.27, 0.27, 0.13, 0.77, 0.37, 0.13, 0.77, 0.37, 0.23, 0.51, 0.96,
        0.23, 0.51, 0.96,
      ],
      3,
    ),
  );
  const axesMat = new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false });
  const axes = new THREE.LineSegments(axesGeom, axesMat);
  axes.renderOrder = 10;

  group.add(axes, centerDot);
  return {
    group,
    dispose: () => {
      sphereGeom.dispose();
      sphereMat.dispose();
      axesGeom.dispose();
      axesMat.dispose();
    },
  };
}

interface PointCloudViewerProps {
  geometry: GeometryData;
  mode?: ViewMode3D;
  onModeChange?: (mode: ViewMode3D) => void;
}

type DensityMode = 'auto' | 'full';
type ViewMode3D = 'points' | 'mesh';

export function PointCloudViewer({
  geometry,
  mode = 'points',
  onModeChange,
}: PointCloudViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const materialRef = useRef<THREE.PointsMaterial | null>(null);
  const activeObjectRef = useRef<THREE.Object3D | null>(null);
  const pivotMarkerRef = useRef<PivotMarkerHandle | null>(null);
  const scheduleRenderRef = useRef<(() => void) | null>(null);

  const [densityMode, setDensityMode] = useState<DensityMode>('auto');
  const [pivotGeometryId, setPivotGeometryId] = useState<string | null>(null);
  const geometryIdRef = useRef(geometry.metadata.id);
  geometryIdRef.current = geometry.metadata.id;
  const hasCustomPivot = pivotGeometryId === geometry.metadata.id;
  const [internalMode, setInternalMode] = useState<ViewMode3D>(mode);
  const viewMode3d = onModeChange ? mode : internalMode;
  const setViewMode3d = (next: ViewMode3D) => {
    if (onModeChange) {
      onModeChange(next);
    } else {
      setInternalMode(next);
    }
  };
  const [pointSize, setPointSize] = useState<number>(2);
  const pointSizeRef = useRef(pointSize);
  pointSizeRef.current = pointSize;

  const { inference_width: width, inference_height: height, intrinsics } = geometry.metadata;
  const autoStride = useMemo(
    () => computePointCloudStride(width, height, DEFAULT_POINT_BUDGET),
    [width, height],
  );
  const stride = densityMode === 'full' ? 1 : autoStride;

  // Keep camera bounds stable across mode/density toggles using a single auto-strided pass.
  const sceneBounds = useMemo(() => {
    const sample = buildPointCloudAttributes({
      points: geometry.points,
      colors: geometry.colors,
      mask: geometry.mask,
      width,
      height,
      stride: autoStride,
    });
    return {
      center: sample.center,
      boundingRadius: sample.boundingRadius,
      autoSample: sample,
    };
  }, [geometry.points, geometry.colors, geometry.mask, width, height, autoStride]);

  const cloudData = useMemo(() => {
    if (viewMode3d !== 'points') {
      return null;
    }
    if (stride === autoStride) {
      return sceneBounds.autoSample;
    }
    return buildPointCloudAttributes({
      points: geometry.points,
      colors: geometry.colors,
      mask: geometry.mask,
      width,
      height,
      stride,
    });
  }, [
    viewMode3d,
    stride,
    autoStride,
    sceneBounds.autoSample,
    geometry.points,
    geometry.colors,
    geometry.mask,
    width,
    height,
  ]);

  const meshData = useMemo(() => {
    if (viewMode3d !== 'mesh') {
      return null;
    }
    const built = ImageMeshBuilder.build(geometry, {
      coordinateSystem: 'threejs',
      colors: geometry.colors,
      stride,
    });
    const index = built.getIndex();
    const triangleCount = index ? Math.floor(index.count / 3) : 0;
    return { geometry: built, triangleCount };
  }, [viewMode3d, geometry, stride]);

  useEffect(() => {
    const built = meshData?.geometry;
    return () => {
      built?.dispose();
    };
  }, [meshData]);

  const initialPose = useMemo(
    () => computeInitialCameraPose(sceneBounds.center, sceneBounds.boundingRadius, intrinsics),
    [sceneBounds.center, sceneBounds.boundingRadius, intrinsics],
  );

  const resetCamera = useCallback(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) {
      return;
    }
    camera.position.set(...initialPose.position);
    camera.up.set(0, 1, 0);
    controls.target.set(...initialPose.target);
    if (pivotMarkerRef.current) {
      pivotMarkerRef.current.group.visible = false;
    }
    setPivotGeometryId(null);
    controls.update();
    scheduleRenderRef.current?.();
  }, [initialPose]);

  // Persistent WebGLRenderer + OrbitControls lifecycle (survives Points <-> Mesh and density toggles).
  useEffect(() => {
    const host = hostRef.current;
    if (!host) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#121316');
    sceneRef.current = scene;

    const pivotMarker = createPivotMarker();
    scene.add(pivotMarker.group);
    pivotMarkerRef.current = pivotMarker;

    const initialWidth = Math.max(1, host.clientWidth);
    const initialHeight = Math.max(1, host.clientHeight);
    const nearClip = Math.max(0.01, sceneBounds.boundingRadius * 0.002);
    const farClip = Math.max(200, sceneBounds.boundingRadius * 20);

    const camera = new THREE.PerspectiveCamera(
      initialPose.fovDeg,
      initialWidth / initialHeight,
      nearClip,
      farClip,
    );
    camera.position.set(...initialPose.position);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(initialWidth, initialHeight, false);
    renderer.domElement.className = 'pointcloud-canvas';
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(...initialPose.target);
    controls.update();
    controlsRef.current = controls;

    let rafId = 0;
    let isUpdatingControls = false;

    const scheduleRender = () => {
      if (rafId !== 0) {
        return;
      }
      rafId = window.requestAnimationFrame(() => {
        rafId = 0;
        isUpdatingControls = true;
        const moved = controls.update();
        isUpdatingControls = false;
        if (pivotMarker.group.visible) {
          const markerScale = Math.max(
            sceneBounds.boundingRadius * 0.005,
            camera.position.distanceTo(pivotMarker.group.position) * 0.02,
          );
          pivotMarker.group.scale.setScalar(markerScale);
        }
        renderer.render(scene, camera);
        if (moved) {
          scheduleRender();
        }
      });
    };
    scheduleRenderRef.current = scheduleRender;

    const handleCanvasDoubleClick = (event: MouseEvent) => {
      const targetObject = activeObjectRef.current;
      if (!targetObject) {
        return;
      }
      const rect = renderer.domElement.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        return;
      }
      const ndcX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
      const hit = pickPointInObject3D({
        object: targetObject,
        camera,
        ndcX,
        ndcY,
        viewportWidth: rect.width,
        viewportHeight: rect.height,
        pointSize: pointSizeRef.current,
      });
      if (!hit) {
        return;
      }
      controls.target.copy(hit);
      pivotMarker.group.position.copy(hit);
      pivotMarker.group.visible = true;
      setPivotGeometryId(geometryIdRef.current);
      controls.update();
      scheduleRender();
    };
    renderer.domElement.addEventListener('dblclick', handleCanvasDoubleClick);

    const onControlsChange = () => {
      if (!isUpdatingControls) {
        scheduleRender();
      }
    };
    controls.addEventListener('change', onControlsChange);

    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      const nextWidth = Math.max(1, Math.round(entry ? entry.contentRect.width : host.clientWidth));
      const nextHeight = Math.max(
        1,
        Math.round(entry ? entry.contentRect.height : host.clientHeight),
      );
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(nextWidth, nextHeight, false);
      scheduleRender();
    });
    resizeObserver.observe(host);

    scheduleRender();

    return () => {
      window.cancelAnimationFrame(rafId);
      rafId = 0;
      scheduleRenderRef.current = null;
      renderer.domElement.removeEventListener('dblclick', handleCanvasDoubleClick);
      controls.removeEventListener('change', onControlsChange);
      resizeObserver.disconnect();
      controls.dispose();
      pivotMarker.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === host) {
        host.removeChild(renderer.domElement);
      }
      sceneRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
      materialRef.current = null;
      activeObjectRef.current = null;
      pivotMarkerRef.current = null;
    };
  }, [sceneBounds.boundingRadius, initialPose]);

  // Swap the active Points or Mesh object without recreating the WebGL context or resetting camera.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) {
      return;
    }

    let ownedGeometry: THREE.BufferGeometry | null = null;
    let activeMaterial: THREE.Material;
    let nextObject: THREE.Object3D;

    if (viewMode3d === 'mesh' && meshData) {
      const meshMaterial = new THREE.MeshBasicMaterial({
        vertexColors: meshData.geometry.hasAttribute('color'),
        color: 0xffffff,
        side: THREE.DoubleSide,
      });
      nextObject = new THREE.Mesh(meshData.geometry, meshMaterial);
      activeMaterial = meshMaterial;
      materialRef.current = null;
    } else if (cloudData) {
      const bufferGeometry = new THREE.BufferGeometry();
      bufferGeometry.setAttribute('position', new THREE.BufferAttribute(cloudData.positions, 3));
      bufferGeometry.setAttribute('color', new THREE.BufferAttribute(cloudData.colors, 3));
      bufferGeometry.computeBoundingSphere();
      ownedGeometry = bufferGeometry;

      const pointsMaterial = new THREE.PointsMaterial({
        size: pointSizeRef.current,
        sizeAttenuation: false,
        vertexColors: true,
      });
      materialRef.current = pointsMaterial;
      nextObject = new THREE.Points(bufferGeometry, pointsMaterial);
      activeMaterial = pointsMaterial;
    } else {
      return;
    }

    scene.add(nextObject);
    activeObjectRef.current = nextObject;
    scheduleRenderRef.current?.();

    return () => {
      scene.remove(nextObject);
      if (activeObjectRef.current === nextObject) {
        activeObjectRef.current = null;
      }
      ownedGeometry?.dispose();
      activeMaterial.dispose();
      if (materialRef.current === activeMaterial) {
        materialRef.current = null;
      }
    };
  }, [viewMode3d, meshData, cloudData]);

  useEffect(() => {
    if (materialRef.current && viewMode3d === 'points') {
      materialRef.current.size = pointSize;
      scheduleRenderRef.current?.();
    }
  }, [pointSize, viewMode3d]);

  return (
    <div className="pointcloud-viewer">
      <div
        ref={hostRef}
        className="pointcloud-canvas-host"
        title="Double-click a point or surface to set the orbit origin"
      />

      <div className="pointcloud-overlay-bar">
        <div className="segmented-control" role="group" aria-label="3D representation">
          <button
            type="button"
            className={`view-mode-button ${viewMode3d === 'points' ? 'is-active' : ''}`}
            onClick={() => setViewMode3d('points')}
            aria-pressed={viewMode3d === 'points'}
            title="View inferred 3D points"
          >
            Points
          </button>
          <button
            type="button"
            className={`view-mode-button ${viewMode3d === 'mesh' ? 'is-active' : ''}`}
            onClick={() => setViewMode3d('mesh')}
            aria-pressed={viewMode3d === 'mesh'}
            title="View 2.5D discontinuity-filtered surface mesh"
          >
            2.5D Mesh
          </button>
        </div>

        <span className="pointcloud-stat mono">
          {viewMode3d === 'mesh'
            ? `${(meshData?.triangleCount ?? 0).toLocaleString()} tris${stride > 1 ? ` (stride ${stride}x)` : ''}`
            : `${(cloudData?.pointCount ?? 0).toLocaleString()} pts${stride > 1 ? ` (stride ${stride}x)` : ''}`}
        </span>

        {autoStride > 1 && (
          <button
            type="button"
            className="toolbar-button"
            onClick={() => setDensityMode((prev) => (prev === 'auto' ? 'full' : 'auto'))}
            title="Toggle between interactive budget and full inference resolution"
          >
            <span>{densityMode === 'full' ? 'Full Res' : 'Auto Density'}</span>
          </button>
        )}

        {viewMode3d === 'points' && (
          <div className="segmented-control" role="group" aria-label="Point size">
            <button
              type="button"
              className="toolbar-icon-button"
              onClick={() => setPointSize((s) => Math.max(1, s - 1))}
              aria-label="Decrease point size"
              title="Smaller points"
            >
              -
            </button>
            <span className="point-size-readout mono">{pointSize}px</span>
            <button
              type="button"
              className="toolbar-icon-button"
              onClick={() => setPointSize((s) => Math.min(6, s + 1))}
              aria-label="Increase point size"
              title="Larger points"
            >
              +
            </button>
          </div>
        )}

        <button
          type="button"
          className="toolbar-button"
          onClick={resetCamera}
          title="Reset 3D camera and rotation origin"
        >
          <RotateCcw size={13} strokeWidth={1.75} />
          <span>{hasCustomPivot ? 'Reset Pivot' : 'Reset Camera'}</span>
        </button>
      </div>
    </div>
  );
}
