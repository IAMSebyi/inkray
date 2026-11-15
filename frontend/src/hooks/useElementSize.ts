import { useEffect } from 'react';
import type { Size2D } from '../types/editor';

export function useElementSize(
  elementRef: React.RefObject<HTMLElement | null>,
  onResize: (size: Size2D) => void,
): void {
  useEffect(() => {
    const element = elementRef.current;
    if (!element) {
      return;
    }

    const reportSize = () => {
      const rect = element.getBoundingClientRect();
      onResize({
        width: rect.width,
        height: rect.height,
      });
    };

    reportSize();

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        reportSize();
        return;
      }
      onResize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      });
    });

    observer.observe(element);
    window.addEventListener('resize', reportSize);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', reportSize);
    };
  }, [elementRef, onResize]);
}
