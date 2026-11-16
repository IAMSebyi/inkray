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
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const materialRef = useRef<THREE.PointsMaterial | null>(null);

  const [densityMode, setDensityMode] = useState<DensityMode>('auto');
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

  const cloudData = useMemo(
    () =>
      buildPointCloudAttributes({
        points: geometry.points,
        colors: geometry.colors,
        mask: geometry.mask,
        width,
        height,
        stride,
      }),
    [geometry.points, geometry.colors, geometry.mask, width, height, stride],
  );

  const initialPose = useMemo(
    () => computeInitialCameraPose(cloudData.center, cloudData.boundingRadius, intrinsics),
    [cloudData.center, cloudData.boundingRadius, intrinsics],
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
    controls.update();
  }, [initialPose]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#121316');

    const initialWidth = Math.max(1, host.clientWidth);
    const initialHeight = Math.max(1, host.clientHeight);
    const nearClip = Math.max(0.01, cloudData.boundingRadius * 0.002);
    const farClip = Math.max(200, cloudData.boundingRadius * 20);

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

    let activeGeometry: THREE.BufferGeometry;
    let activeMaterial: THREE.Material;

    if (viewMode3d === 'mesh') {
      const meshGeometry = ImageMeshBuilder.build(geometry, {
        coordinateSystem: 'threejs',
        colors: geometry.colors,
      });
      const meshMaterial = new THREE.MeshBasicMaterial({
        vertexColors: meshGeometry.hasAttribute('color'),
        color: 0xffffff,
        side: THREE.DoubleSide,
      });
      const meshObject = new THREE.Mesh(meshGeometry, meshMaterial);
      scene.add(meshObject);
      activeGeometry = meshGeometry;
      activeMaterial = meshMaterial;
      materialRef.current = null;
    } else {
      const bufferGeometry = new THREE.BufferGeometry();
      bufferGeometry.setAttribute('position', new THREE.BufferAttribute(cloudData.positions, 3));
      bufferGeometry.setAttribute('color', new THREE.BufferAttribute(cloudData.colors, 3));
      bufferGeometry.computeBoundingSphere();

      const pointsMaterial = new THREE.PointsMaterial({
        size: pointSizeRef.current,
        sizeAttenuation: false,
        vertexColors: true,
      });
      materialRef.current = pointsMaterial;

      const pointsObject = new THREE.Points(bufferGeometry, pointsMaterial);
      scene.add(pointsObject);
      activeGeometry = bufferGeometry;
      activeMaterial = pointsMaterial;
    }

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
    });
    resizeObserver.observe(host);

    let rafId = 0;
    const renderLoop = () => {
      rafId = window.requestAnimationFrame(renderLoop);
      controls.update();
      renderer.render(scene, camera);
    };
    renderLoop();

    return () => {
      window.cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      controls.dispose();
      activeGeometry.dispose();
      activeMaterial.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === host) {
        host.removeChild(renderer.domElement);
      }
      cameraRef.current = null;
      controlsRef.current = null;
      materialRef.current = null;
    };
  }, [geometry, cloudData, initialPose, viewMode3d]);

  useEffect(() => {
    if (materialRef.current && viewMode3d === 'points') {
      materialRef.current.size = pointSize;
    }
  }, [pointSize, viewMode3d]);

  return (
    <div className="pointcloud-viewer">
      <div ref={hostRef} className="pointcloud-canvas-host" />

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
            ? `${geometry.metadata.inference_width}x${geometry.metadata.inference_height} mesh`
            : `${cloudData.pointCount.toLocaleString()} pts${stride > 1 ? ` (stride ${stride}x)` : ''}`}
        </span>

        {viewMode3d === 'points' && autoStride > 1 && (
          <button
            type="button"
            className="toolbar-button"
            onClick={() => setDensityMode((prev) => (prev === 'auto' ? 'full' : 'auto'))}
            title="Toggle between point budget and full inference resolution"
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
          title="Reset 3D camera"
        >
          <RotateCcw size={13} strokeWidth={1.75} />
          <span>Reset Camera</span>
        </button>
      </div>
    </div>
  );
}
