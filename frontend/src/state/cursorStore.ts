import type { ImageCoordinateSample } from '../types/editor';

export type CursorListener = () => void;

export interface CursorStore {
  getSnapshot: () => ImageCoordinateSample | null;
  setSample: (next: ImageCoordinateSample | null) => void;
  subscribe: (listener: CursorListener) => () => void;
}

export function createCursorStore(): CursorStore {
  let current: ImageCoordinateSample | null = null;
  const listeners = new Set<CursorListener>();

  return {
    getSnapshot: () => current,
    setSample: (next) => {
      if (current === null && next === null) {
        return;
      }
      if (
        current !== null &&
        next !== null &&
        current.inBounds === next.inBounds &&
        current.pixel?.x === next.pixel?.x &&
        current.pixel?.y === next.pixel?.y &&
        current.imagePoint.x === next.imagePoint.x &&
        current.imagePoint.y === next.imagePoint.y
      ) {
        return;
      }
      current = next;
      for (const listener of listeners) {
        listener();
      }
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}
