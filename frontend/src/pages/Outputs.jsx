import { useState, useEffect } from 'react';
import { HiOutlineTableCells, HiOutlineFolderOpen } from 'react-icons/hi2';
import './Outputs.css';

const API = 'http://localhost:5000/api';

const FILE_LABELS = {
  'classified_patents.csv': { name: 'Classified Patents', stage: 'Stage 1', color: 'info' },
  'keyword_filtered_patents.csv': { name: 'Keyword Filtered', stage: 'Stage 2', color: 'accent' },
  'citation_expanded_patents.csv': { name: 'Citation Expanded', stage: 'Stage 3', color: 'warning' },
  'output.csv': { name: 'Final Results', stage: 'Stage 4', color: 'success' },
};

export default function Outputs() {
  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [tableData, setTableData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const res = await fetch(`${API}/outputs`);
      const data = await res.json();
      setFiles(data.files || []);
      if (data.files?.length > 0) {
        loadFile(data.files[data.files.length - 1].filename);
      }
    } catch {
      setError('Cannot connect to backend.');
    }
  };

  const loadFile = async (filename) => {
    setActiveFile(filename);
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API}/outputs/${filename}`);
      const data = await res.json();
      if (res.ok) {
        setTableData(data);
      } else {
        setError(data.error);
      }
    } catch {
      setError('Failed to load file data.');
    }
    setLoading(false);
  };

  const formatCellValue = (value, col) => {
    if (value === null || value === undefined || value === '') return '—';
    if (typeof value === 'number') {
      if (col.includes('score') || col.includes('pagerank')) return value.toFixed(4);
      return value.toString();
    }
    const str = String(value);
    if (str.length > 100) return str.slice(0, 100) + '...';
    return str;
  };

  return (
    <div className="container">
      <div className="animate-in">
        <h1 className="page-title">Pipeline Outputs</h1>
        <p className="page-subtitle">View generated CSV output files from each pipeline stage in tabular format.</p>
      </div>

      {/* File Tabs */}
      {files.length > 0 && (
        <div className="clay-tabs animate-in-delay-1">
          {files.map((file) => {
            const meta = FILE_LABELS[file.filename] || {};
            return (
              <button
                key={file.filename}
                className={`clay-tab ${activeFile === file.filename ? 'active' : ''}`}
                onClick={() => loadFile(file.filename)}
              >
                <span className="tab-stage">{meta.stage}</span>
                <span className="tab-name">{meta.name || file.filename}</span>
                <span className="tab-count">{file.rows} rows</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Table Display */}
      {loading && (
        <div className="clay-card" style={{ textAlign: 'center', padding: 'var(--space-2xl)' }}>
          <div className="spinner" style={{ margin: '0 auto' }} />
          <p style={{ marginTop: 'var(--space-md)', color: 'var(--text-muted)' }}>Loading data...</p>
        </div>
      )}

      {error && (
        <div className="clay-card">
          <div className="empty-state">
            <div className="empty-state-icon">⚠️</div>
            <p className="empty-state-text">{error}</p>
          </div>
        </div>
      )}

      {!loading && !error && tableData && (
        <div className="animate-in-delay-2">
          <div className="table-meta">
            <span className="badge badge-accent">
              <HiOutlineTableCells /> {tableData.total_rows} rows × {tableData.columns.length} columns
            </span>
          </div>

          <div className="clay-table-wrapper" style={{ maxHeight: '600px', overflowY: 'auto' }}>
            <table className="clay-table">
              <thead>
                <tr>
                  <th>#</th>
                  {tableData.columns.map((col) => (
                    <th key={col}>{col.replace(/_/g, ' ')}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableData.data.map((row, idx) => (
                  <tr key={idx}>
                    <td style={{ color: 'var(--text-muted)', fontWeight: 600 }}>{idx + 1}</td>
                    {tableData.columns.map((col) => (
                      <td key={col} title={String(row[col] || '')}>
                        {formatCellValue(row[col], col)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!loading && !error && files.length === 0 && (
        <div className="clay-card animate-in-delay-1">
          <div className="empty-state">
            <div className="empty-state-icon"><HiOutlineFolderOpen /></div>
            <p className="empty-state-text">No output files yet.</p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 'var(--space-sm)' }}>
              Run the pipeline from the Home page to generate outputs.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
