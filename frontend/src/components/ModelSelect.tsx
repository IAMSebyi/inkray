import { useId } from 'react';
import type { ModelAvailability } from '../api/backend';
import {
  MODEL_DOWNLOAD_NOTES,
  MODEL_OPTION_NOTES,
  MODEL_USAGE,
  type AnyModelId,
} from '../types/editor';

interface Props<T extends AnyModelId> {
  label: string;
  models: Readonly<Record<T, string>>;
  value: T;
  unavailable?: ModelAvailability['unavailable'];
  disabled: boolean;
  onChange: (value: T) => void;
}

export function ModelSelect<T extends AnyModelId>({
  label,
  models,
  value,
  unavailable,
  disabled,
  onChange,
}: Props<T>) {
  const ids = Object.keys(models) as T[];
  const disabledModels = ids.filter((id) => unavailable?.[id]);
  const usageId = useId();
  const downloadNote = !unavailable?.[value] ? MODEL_DOWNLOAD_NOTES[value] : undefined;
  return (
    <>
      <label className="inspector-control-row">
        <span>{label}</span>
        <select
          className="inspector-select"
          value={value}
          disabled={disabled}
          aria-describedby={usageId}
          onChange={(event) => onChange(event.target.value as T)}
        >
          {ids.map((id) => {
            const optionNote = MODEL_OPTION_NOTES[id];
            return (
              <option key={id} value={id} disabled={Boolean(unavailable?.[id])}>
                {models[id]}
                {optionNote ? ` (${optionNote})` : ''}
                {unavailable?.[id] ? ' (unavailable)' : ''}
              </option>
            );
          })}
        </select>
      </label>
      <p className="inspector-empty-note" id={usageId}>
        {MODEL_USAGE[value]}
      </p>
      {downloadNote && <p className="inspector-empty-note">{downloadNote}</p>}
      {disabledModels.length > 0 && (
        <details className="inspector-details">
          <summary>Unavailable models</summary>
          <ul>
            {disabledModels.map((id) => (
              <li key={id}>
                <strong>{models[id]}</strong> {unavailable?.[id]}
              </li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
}
