import { useState, useEffect, useRef } from 'react';
import {
  HiOutlineCloudArrowUp,
  HiOutlinePlay,
  HiOutlineDocumentText,
  HiOutlineArrowDownTray,
  HiOutlineCheckCircle,
  HiOutlineCog6Tooth,
  HiOutlineExclamationTriangle,
  HiOutlineBeaker,
  HiOutlineMagnifyingGlass,
  HiOutlineLink,
  HiOutlineCpuChip,
  HiOutlineSparkles,
  HiOutlineDocument,
  HiOutlineXMark,
} from 'react-icons/hi2';
import './Home.css';

const API = 'http://localhost:5000/api';

const STAGES = [
  { num: 1, name: 'Classification Search', icon: <HiOutlineMagnifyingGlass />, desc: 'IPC/CPC prefix matching' },
  { num: 2, name: 'Keyword Search', icon: <HiOutlineBeaker />, desc: 'Weighted keyword scoring' },
  { num: 3, name: 'Citation Search', icon: <HiOutlineLink />, desc: 'PageRank + BFS expansion' },
  { num: 4, name: 'Semantic Search', icon: <HiOutlineCpuChip />, desc: 'SPECTER + cross-encoder' },
  { num: 5, name: 'AI Analysis', icon: <HiOutlineSparkles />, desc: 'Groq / Llama 3.3 70B' },
];

export default function Home() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [csvPreview, setCsvPreview] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [report, setReport] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const pollRef = useRef(null);
  const fileInputRef = useRef(null);

  // Poll pipeline status
  useEffect(() => {
    if (pipelineStatus?.status === 'running') {
      pollRef.current = setInterval(async () => {
        try {
          const res = await fetch(`${API}/status`);
          const data = await res.json();
          setPipelineStatus(data);

          if (data.status === 'completed' || data.status === 'error') {
            clearInterval(pollRef.current);
            if (data.status === 'completed') {
              fetchReport();
            }
          }
        } catch (err) {
          console.error('Poll error:', err);
        }
      }, 1000);
    }

    return () => clearInterval(pollRef.current);
  }, [pipelineStatus?.status]);

  // Check if pipeline already has results on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/status`);
        const data = await res.json();
        setPipelineStatus(data);
        if (data.status === 'running') return; // will be picked up by poll
      } catch {
        // backend not running yet, ignore
      }
      // Always try to load existing report (it may exist from a prior run)
      fetchReport();
    })();
  }, []);

  const fetchReport = async () => {
    try {
      const res = await fetch(`${API}/report`);
      if (res.ok) {
        const data = await res.json();
        setReport(data.report);
      }
    } catch (err) {
      console.error('Failed to fetch report:', err);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      setError('Please upload a .csv file.');
      return;
    }

    setError('');
    setUploadedFile(file);

    // Read and preview the file
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      setCsvPreview(text);
    };
    reader.readAsText(file);
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    setCsvPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file) {
      if (!file.name.endsWith('.csv')) {
        setError('Please upload a .csv file.');
        return;
      }
      setError('');
      setUploadedFile(file);
      const reader = new FileReader();
      reader.onload = (event) => setCsvPreview(event.target.result);
      reader.readAsText(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleSubmit = async () => {
    setError('');

    if (!uploadedFile || !csvPreview) {
      setError('Please upload a CSV file first.');
      return;
    }

    setIsSubmitting(true);
    setReport('');

    try {
      const res = await fetch(`${API}/upload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ csv_content: csvPreview }),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.error || 'Failed to start pipeline');
        setIsSubmitting(false);
        return;
      }

      setPipelineStatus({
        status: 'running',
        current_stage: null,
        stages_completed: [],
      });
    } catch (err) {
      setError('Cannot connect to backend. Make sure Flask server is running on port 5000.');
    }
    setIsSubmitting(false);
  };

  const handleDownloadPdf = () => {
    window.open(`${API}/report/pdf`, '_blank');
  };

  const getStageStatus = (stageNum) => {
    if (!pipelineStatus) return 'pending';
    if (pipelineStatus.stages_completed?.includes(stageNum)) return 'completed';
    if (pipelineStatus.current_stage === stageNum) return 'active';
    return 'pending';
  };

  // Parse CSV preview into a simple table
  const parseCsvPreview = () => {
    if (!csvPreview) return null;
    const lines = csvPreview.trim().split('\n');
    if (lines.length < 2) return null;

    // Simple CSV parse (handles basic cases)
    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
    return { headers, rowCount: lines.length - 1 };
  };

  const previewData = parseCsvPreview();

  // Parse markdown-like report into formatted JSX
  const renderReport = (text) => {
    if (!text) return null;
    const lines = text.split('\n');
    const elements = [];
    let key = 0;

    const formatInline = (str) => {
      // Convert **bold** to <strong>
      const parts = [];
      const regex = /\*\*(.+?)\*\*/g;
      let lastIndex = 0;
      let match;
      while ((match = regex.exec(str)) !== null) {
        if (match.index > lastIndex) {
          parts.push(str.slice(lastIndex, match.index));
        }
        parts.push(<strong key={`b${match.index}`}>{match[1]}</strong>);
        lastIndex = regex.lastIndex;
      }
      if (lastIndex < str.length) {
        parts.push(str.slice(lastIndex));
      }
      return parts.length > 0 ? parts : [str];
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trimEnd();

      // Skip separator lines (===)
      if (/^={3,}$/.test(line.trim())) continue;

      // Skip empty lines
      if (!line.trim()) {
        elements.push(<div key={key++} className="report-spacer" />);
        continue;
      }

      // ## Section heading
      if (line.startsWith('## ')) {
        const heading = line.replace(/^#+\s*/, '').replace(/\*\*/g, '');
        elements.push(
          <h3 key={key++} className="report-heading">{heading}</h3>
        );
        continue;
      }

      // Verdict keywords — color-coded
      const upper = line.trim().toUpperCase();
      if (upper === 'RECOMMENDED' || upper === 'CONDITIONAL' || upper === 'NOT RECOMMENDED') {
        const cls = upper === 'RECOMMENDED' ? 'verdict-recommended'
          : upper === 'CONDITIONAL' ? 'verdict-conditional'
          : 'verdict-not-recommended';
        elements.push(
          <div key={key++} className={`report-verdict ${cls}`}>{line.trim()}</div>
        );
        continue;
      }

      // Bullet point (- or * or •)
      if (/^[-*•]\s+/.test(line.trim())) {
        const content = line.trim().replace(/^[-*•]\s+/, '');
        elements.push(
          <li key={key++} className="report-bullet">{formatInline(content)}</li>
        );
        continue;
      }

      // Numbered list (1. 2. etc.)
      if (/^\d+\.\s+/.test(line.trim())) {
        const content = line.trim().replace(/^\d+\.\s+/, '');
        elements.push(
          <li key={key++} className="report-bullet report-numbered">{formatInline(content)}</li>
        );
        continue;
      }

      // Header-ish line (all caps with colon, like "Query Patent : ...")
      if (/^[A-Z][A-Za-z\s]+:/.test(line.trim()) && line.trim().length < 120) {
        elements.push(
          <p key={key++} className="report-meta">{formatInline(line.trim())}</p>
        );
        continue;
      }

      // Regular paragraph
      elements.push(
        <p key={key++} className="report-paragraph">{formatInline(line)}</p>
      );
    }

    return elements;
  };

  return (
    <div className="container">
      {/* Hero */}
      <section className="hero-section animate-in">
        <div className="clay-card hero-card">
          <div className="hero-content">
            <h1 className="page-title">Prior Art Search</h1>
            <p className="hero-desc">
              A multi-stage patent prior art search pipeline that progressively filters a patent corpus
              using classification codes, keyword scoring, citation network analysis, and semantic similarity
              — culminating in an AI-generated patent viability report.
            </p>
            <div className="hero-badges">
              <span className="badge badge-accent">IPC/CPC Matching</span>
              <span className="badge badge-accent">PageRank + BFS</span>
              <span className="badge badge-accent">SPECTER Embeddings</span>
              <span className="badge badge-accent">Cross-Encoder</span>
              <span className="badge badge-accent">Llama 3.3 70B</span>
            </div>
          </div>
        </div>
      </section>

      {/* Upload Section */}
      <section className="input-section animate-in-delay-1">
        <h2 className="section-title">
          <HiOutlineCloudArrowUp style={{ verticalAlign: 'middle', marginRight: 8 }} />
          Upload Invention CSV
        </h2>

        <div className="clay-card">
          {/* Drop Zone */}
          <div
            className={`upload-dropzone ${uploadedFile ? 'has-file' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => !uploadedFile && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />

            {!uploadedFile ? (
              <>
                <div className="dropzone-icon">
                  <HiOutlineCloudArrowUp />
                </div>
                <p className="dropzone-title">Drop your CSV file here</p>
                <p className="dropzone-hint">or click to browse</p>
                <p className="dropzone-format">
                  Required columns: title, abstract, ipc_codes, cpc_codes, keywords, citations, publication_year
                </p>
              </>
            ) : (
              <div className="file-info">
                <div className="file-icon">
                  <HiOutlineDocument />
                </div>
                <div className="file-details">
                  <span className="file-name">{uploadedFile.name}</span>
                  <span className="file-meta">
                    {(uploadedFile.size / 1024).toFixed(1)} KB
                    {previewData && ` · ${previewData.rowCount} row${previewData.rowCount !== 1 ? 's' : ''} · ${previewData.headers.length} columns`}
                  </span>
                  {previewData && (
                    <div className="file-columns">
                      {previewData.headers.map((h, i) => (
                        <span key={i} className="badge badge-accent">{h}</span>
                      ))}
                    </div>
                  )}
                </div>
                <button className="remove-file-btn" onClick={(e) => { e.stopPropagation(); handleRemoveFile(); }}>
                  <HiOutlineXMark />
                </button>
              </div>
            )}
          </div>

          {error && (
            <div className="error-banner">
              <HiOutlineExclamationTriangle />
              <span>{error}</span>
            </div>
          )}

          <div className="form-actions">
            <button
              className="clay-btn clay-btn-primary"
              onClick={handleSubmit}
              disabled={isSubmitting || pipelineStatus?.status === 'running' || !uploadedFile}
            >
              {pipelineStatus?.status === 'running' ? (
                <>
                  <span className="spinner" /> Running...
                </>
              ) : (
                <>
                  <HiOutlinePlay /> Run Pipeline
                </>
              )}
            </button>
          </div>
        </div>
      </section>

      {/* Pipeline Progress */}
      {pipelineStatus && pipelineStatus.status !== 'idle' && (
        <section className="progress-section animate-in-delay-2">
          <h2 className="section-title">
            <HiOutlineCog6Tooth style={{ verticalAlign: 'middle', marginRight: 8 }} />
            Pipeline Progress
            {pipelineStatus.elapsed_seconds != null && (
              <span className="elapsed-time">{pipelineStatus.elapsed_seconds}s</span>
            )}
          </h2>

          <div className="clay-card">
            <div className="pipeline-stages">
              {STAGES.map((stage, idx) => {
                const status = getStageStatus(stage.num);
                return (
                  <div key={stage.num} className={`stage-item stage-${status}`}>
                    <div className="stage-connector">
                      {idx > 0 && <div className={`connector-line ${status === 'pending' ? '' : 'connector-active'}`} />}
                    </div>
                    <div className="stage-circle">
                      {status === 'completed' ? (
                        <HiOutlineCheckCircle className="stage-icon-done" />
                      ) : status === 'active' ? (
                        <div className="spinner" />
                      ) : (
                        <span className="stage-num">{stage.num}</span>
                      )}
                    </div>
                    <div className="stage-info">
                      <span className="stage-name">{stage.name}</span>
                      <span className="stage-desc">{stage.desc}</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {pipelineStatus.status === 'error' && (
              <div className="error-banner" style={{ marginTop: 'var(--space-md)' }}>
                <HiOutlineExclamationTriangle />
                <span>{pipelineStatus.error}</span>
              </div>
            )}

            {pipelineStatus.status === 'completed' && (
              <div className="success-banner">
                <HiOutlineCheckCircle />
                <span>Pipeline completed successfully in {pipelineStatus.elapsed_seconds}s</span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Report Section */}
      {report && (
        <section className="report-section animate-in-delay-3">
          <div className="report-header">
            <h2 className="section-title">
              <HiOutlineDocumentText style={{ verticalAlign: 'middle', marginRight: 8 }} />
              Analysis Report
            </h2>
            <button className="clay-btn clay-btn-primary" onClick={handleDownloadPdf}>
              <HiOutlineArrowDownTray /> Download PDF
            </button>
          </div>

          <div className="clay-card report-card">
            <div className="report-content">{renderReport(report)}</div>
          </div>
        </section>
      )}
    </div>
  );
}
