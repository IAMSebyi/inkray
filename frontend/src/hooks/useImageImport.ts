import { useCallback, useEffect, useRef, useState } from 'react';
import { useEditor } from './useEditor';

function hasDraggedFiles(dataTransfer: DataTransfer | null): boolean {
  if (!dataTransfer) {
    return false;
  }
  return Array.from(dataTransfer.types).includes('Files');
}

export function useImageImport() {
  const { importFile } = useEditor();
  const [isDragActive, setIsDragActive] = useState(false);
  const dragDepthRef = useRef(0);

  // Prevent browser file navigation if dropped outside the viewport.
  useEffect(() => {
    const preventWindowDrop = (event: DragEvent) => {
      if (hasDraggedFiles(event.dataTransfer)) {
        event.preventDefault();
      }
    };

    window.addEventListener('dragover', preventWindowDrop);
    window.addEventListener('drop', preventWindowDrop);
    return () => {
      window.removeEventListener('dragover', preventWindowDrop);
      window.removeEventListener('drop', preventWindowDrop);
    };
  }, []);

  const handleFileInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      // Clear input value so picking the same file again still triggers onChange.
      event.target.value = '';
      if (file) {
        void importFile(file);
      }
    },
    [importFile],
  );

  const handleDragEnter = useCallback((event: React.DragEvent<HTMLElement>) => {
    if (!hasDraggedFiles(event.dataTransfer)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    dragDepthRef.current += 1;
    setIsDragActive(true);
  }, []);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLElement>) => {
    if (!hasDraggedFiles(event.dataTransfer)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'copy';
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLElement>) => {
    if (!hasDraggedFiles(event.dataTransfer)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setIsDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLElement>) => {
      if (!hasDraggedFiles(event.dataTransfer)) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      dragDepthRef.current = 0;
      setIsDragActive(false);

      const file = event.dataTransfer.files[0];
      if (file) {
        void importFile(file);
      }
    },
    [importFile],
  );

  return {
    isDragActive,
    handleFileInputChange,
    dropZoneProps: {
      onDragEnter: handleDragEnter,
      onDragOver: handleDragOver,
      onDragLeave: handleDragLeave,
      onDrop: handleDrop,
    },
  };
}
