/**
 * ConfigPanel — slide-in panel for model/strategy configuration.
 */

import React from 'react';
import { useStore } from '../stores/useStore';

const STRATEGIES = ['adaptive', 'parallel', 'sequential'];

export default function ConfigPanel() {
  const isOpen = useStore((s) => s.isConfigPanelOpen);
  const toggle = useStore((s) => s.toggleConfigPanel);
  const config = useStore((s) => s.config);
  const updateConfig = useStore((s) => s.updateConfig);

  const [models, setModels] = React.useState([]);

  React.useEffect(() => {
    if (isOpen && models.length === 0) {
      fetch('http://localhost:8000/api/models')
        .then((r) => r.json())
        .then((data) => setModels(data))
        .catch((err) => console.error('Failed to load models:', err));
    }
  }, [isOpen, models.length]);

  if (!isOpen) return null;

  const field = (label, key, type = 'text', options = null, datalist = null) => {
    const listId = datalist ? `list-${key}` : undefined;
    return (
      <div style={{ marginBottom: 16 }}>
        <label style={{ display: 'block', fontSize: 12, color: 'var(--text-muted)', marginBottom: 6, fontWeight: 500 }}>
          {label}
        </label>
        {options ? (
          <select
            id={`config-${key}`}
            className="input"
            value={config[key] || ''}
            onChange={(e) => updateConfig({ [key]: e.target.value })}
            style={{ padding: '8px 12px' }}
          >
            {options.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        ) : (
          <>
            <input
              id={`config-${key}`}
              type={type}
              className="input"
              value={config[key] || ''}
              onChange={(e) => updateConfig({ [key]: e.target.value || undefined })}
              placeholder={key}
              list={listId}
              style={{ padding: '8px 12px' }}
            />
            {datalist && (
              <datalist id={listId}>
                {datalist.map((o) => <option key={o} value={o} />)}
              </datalist>
            )}
          </>
        )}
      </div>
    );
  };

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={toggle}
        style={{
          position: 'fixed', inset: 0,
          background: 'var(--bg-overlay)',
          zIndex: 200,
        }}
      />
      {/* Panel */}
      <div
        id="config-panel"
        className="animate-slide-right"
        style={{
          position: 'fixed', right: 0, top: 0, bottom: 0,
          width: 360,
          background: 'var(--bg-secondary)',
          borderLeft: '1px solid var(--border)',
          zIndex: 201,
          display: 'flex',
          flexDirection: 'column',
          boxShadow: 'var(--shadow-lg)',
        }}
      >
        <div style={{ padding: 'var(--sp-md)', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h2>Configuration</h2>
          <button className="btn btn-ghost btn-sm" onClick={toggle}>✕</button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: 'var(--sp-lg)' }}>
          {field('Model', 'model', 'text', models.length > 0 ? models : ['ollama/ornith:latest'])}
          {field('Orchestrator Model (override)', 'orchestrator_model', 'text', models.length > 0 ? [''].concat(models) : [''])}
          {field('Strategy', 'strategy', 'text', STRATEGIES)}
          {field('Max Subtasks', 'max_subtasks', 'number')}
          {field('API Key', 'api_key', 'password')}
          {field('API Base URL', 'api_base')}
        </div>
        <div style={{ padding: 'var(--sp-md)', borderTop: '1px solid var(--border)' }}>
          <button className="btn btn-primary" style={{ width: '100%' }} onClick={toggle}>
            Save & Close
          </button>
        </div>
      </div>
    </>
  );
}
