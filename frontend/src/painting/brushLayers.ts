import type { BrushStroke } from './brushEngine';
import type { ImageSpaceStroke } from './imageSpacePaint';

export type BrushLayer =
  | { kind: 'planar'; key: string; erase: boolean; strokes: BrushStroke[] }
  | { kind: 'image'; key: string; erase: boolean; strokes: ImageSpaceStroke[] };

export function buildBrushLayers(
  strokes: readonly BrushStroke[],
  imageStrokes: readonly ImageSpaceStroke[],
): BrushLayer[] {
  const ordered = [
    ...strokes.map((stroke, index) => ({
      kind: 'planar' as const,
      stroke,
      order: stroke.order ?? index,
    })),
    ...imageStrokes.map((stroke, index) => ({
      kind: 'image' as const,
      stroke,
      order: stroke.order ?? strokes.length + index,
    })),
  ].sort((a, b) => a.order - b.order);
  const layers: BrushLayer[] = [];
  for (const item of ordered) {
    const erase = item.stroke.mode === 'erase';
    const previous = layers[layers.length - 1];
    if (item.kind === 'planar') {
      if (
        previous?.kind === 'planar' &&
        previous.erase === erase &&
        previous.strokes[0]?.surfaceId === item.stroke.surfaceId
      ) {
        previous.strokes.push(item.stroke);
      } else {
        layers.push({ kind: 'planar', key: item.stroke.id, erase, strokes: [item.stroke] });
      }
    } else if (
      previous?.kind === 'image' &&
      previous.erase === erase &&
      previous.strokes[0]?.layer === item.stroke.layer
    ) {
      previous.strokes.push(item.stroke);
    } else {
      layers.push({ kind: 'image', key: item.stroke.id, erase, strokes: [item.stroke] });
    }
  }
  return layers;
}

export function nextStrokeOrder(
  strokes: readonly BrushStroke[],
  imageStrokes: readonly ImageSpaceStroke[],
): number {
  let order = strokes.length + imageStrokes.length;
  for (const stroke of [...strokes, ...imageStrokes]) {
    order = Math.max(order, (stroke.order ?? -1) + 1);
  }
  return order;
}
