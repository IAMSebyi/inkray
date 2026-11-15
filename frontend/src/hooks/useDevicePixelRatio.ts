import { useEffect, useState } from 'react';

export function useDevicePixelRatio(): number {
  const [dpr, setDpr] = useState(() =>
    typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1,
  );

  useEffect(() => {
    let removeMatchMediaListener: (() => void) | null = null;

    const observeCurrentDpr = () => {
      removeMatchMediaListener?.();
      const currentDpr = window.devicePixelRatio || 1;
      setDpr(currentDpr);

      if (typeof window.matchMedia !== 'function') {
        return;
      }

      const mediaQuery = window.matchMedia(`(resolution: ${currentDpr}dppx)`);
      const handleChange = () => {
        observeCurrentDpr();
      };
      mediaQuery.addEventListener('change', handleChange);
      removeMatchMediaListener = () => {
        mediaQuery.removeEventListener('change', handleChange);
      };
    };

    observeCurrentDpr();
    window.addEventListener('resize', observeCurrentDpr);

    return () => {
      removeMatchMediaListener?.();
      window.removeEventListener('resize', observeCurrentDpr);
    };
  }, []);

  return dpr;
}
