import { AlertCircle, X } from 'lucide-react';
import { useEditor } from '../hooks/useEditor';

export function ErrorBanner() {
  const { state, clearError } = useEditor();

  if (!state.error) {
    return null;
  }

  return (
    <div className="error-banner" role="alert" aria-live="polite">
      <AlertCircle size={15} strokeWidth={1.75} className="error-banner-icon" />
      <span className="error-banner-text">{state.error}</span>
      <button
        type="button"
        className="error-banner-dismiss"
        onClick={clearError}
        aria-label="Dismiss error"
        title="Dismiss"
      >
        <X size={14} strokeWidth={1.75} />
      </button>
    </div>
  );
}
