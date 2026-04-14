import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell, LabelList,
  ScatterChart, Scatter, ZAxis,
} from 'recharts';
import {
  HiOutlineChartBar,
  HiOutlineFunnel,
  HiOutlineSparkles,
  HiOutlineLink,
  HiOutlineMagnifyingGlass,
  HiOutlineCircleStack,
  HiOutlineKey,
  HiOutlineServerStack,
} from 'react-icons/hi2';
import './Insights.css';

const API = 'http://localhost:5000/api';

const COLORS = ['#7c5cfc', '#a78bfa', '#c4b5fd', '#e0d7fe', '#f0ecff', '#6366f1', '#818cf8', '#4f46e5'];
const BRIGHT_COLORS = ['#7c5cfc', '#22c55e', '#f59e0b', '#ef4444', '#3b82f6', '#ec4899', '#14b8a6', '#8b5cf6'];
const PIE_COLORS = ['#7c5cfc', '#22c55e', '#f59e0b', '#3b82f6', '#ef4444', '#ec4899'];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="tooltip-label">{label}</p>
      {payload.map((p, i) => (
        <p key={i} className="tooltip-row" style={{ color: p.color || p.fill }}>
          <span className="tooltip-dot" style={{ background: p.color || p.fill }} />
          {p.name}: <strong>{typeof p.value === 'number' ? (p.value < 0.01 ? p.value.toFixed(6) : p.value.toFixed(4)) : p.value}</strong>
        </p>
      ))}
    </div>
  );
};

const ScatterTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div className="chart-tooltip">
      <p className="tooltip-label">{d.patent_id}</p>
      <p className="tooltip-row">Forward Citations: <strong>{d.forward}</strong></p>
      <p className="tooltip-row">Backward Citations: <strong>{d.backward}</strong></p>
      <p className="tooltip-row">Citation Score: <strong>{d.citation_score?.toFixed(4)}</strong></p>
    </div>
  );
};

const RADIAN = Math.PI / 180;
const renderPieLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + (outerRadius + 20) * Math.cos(-midAngle * RADIAN);
  const y = cy + (outerRadius + 20) * Math.sin(-midAngle * RADIAN);
  if (percent < 0.05) return null;
  return (
    <text x={x} y={y} fill="#6b6880" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central" fontSize={11}>
      {name} ({(percent * 100).toFixed(0)}%)
    </text>
  );
};

export default function Insights() {
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchInsights();
  }, []);

  const fetchInsights = async () => {
    try {
      const res = await fetch(`${API}/insights`);
      const data = await res.json();
      setInsights(data);
    } catch {
      setError('Cannot connect to backend.');
    }
    setLoading(false);
  };

  if (loading) {
    return (
      <div className="container">
        <div className="clay-card" style={{ textAlign: 'center', padding: 'var(--space-2xl)' }}>
          <div className="spinner" style={{ margin: '0 auto' }} />
          <p style={{ marginTop: 'var(--space-md)', color: 'var(--text-muted)' }}>Loading insights...</p>
        </div>
      </div>
    );
  }

  if (error || !insights) {
    return (
      <div className="container">
        <h1 className="page-title">Insights</h1>
        <div className="clay-card">
          <div className="empty-state">
            <div className="empty-state-icon">📊</div>
            <p className="empty-state-text">{error || 'No data available yet.'}</p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 'var(--space-sm)' }}>
              Run the pipeline from the Home page first.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const finalScores = insights.final_scores || [];
  const funnel = insights.funnel || [];
  const hopDist = insights.hop_distribution || [];
  const seedVsDiscovered = insights.seed_vs_discovered || [];
  const citationScatter = insights.citation_scatter || [];
  const keywordScores = insights.keyword_scores || [];
  const ipcDist = insights.ipc_distribution || [];
  const meta = insights.pipeline_meta || {};

  // Derived data
  const scoreBreakdown = finalScores.map((p) => ({
    name: p.patent_id,
    title: p.title,
    rerank_score: Math.abs(p.rerank_score),
    retrieval_score: p.retrieval_score,
    citation_score: p.citation_score,
  }));

  const yearCounts = {};
  finalScores.forEach((p) => {
    if (p.publication_year) {
      yearCounts[p.publication_year] = (yearCounts[p.publication_year] || 0) + 1;
    }
  });
  const yearData = Object.entries(yearCounts)
    .map(([year, count]) => ({ year, count }))
    .sort((a, b) => a.year - b.year);



  return (
    <div className="container">
      <div className="animate-in">
        <h1 className="page-title">Insights</h1>
        <p className="page-subtitle">Comprehensive visual analysis of pipeline results across all stages.</p>
      </div>

      {/* Pipeline Metadata Cards */}
      {(meta.embed_backend || funnel.length > 0) && (
        <div className="meta-cards animate-in-delay-1">
          {funnel.map((f, i) => (
            <div key={i} className="clay-card-sm stat-card">
              <span className="stat-label">{f.stage}</span>
              <span className="stat-value">{f.count.toLocaleString()}</span>
              <span className="stat-unit">patents</span>
            </div>
          ))}
          {meta.embed_backend && (
            <div className="clay-card-sm stat-card">
              <span className="stat-label">Embedding</span>
              <span className="stat-value stat-value-sm">{meta.embed_backend.toUpperCase()}</span>
              <span className="stat-unit">{meta.retrieval_metric}</span>
            </div>
          )}
          {meta.rerank_backend && (
            <div className="clay-card-sm stat-card">
              <span className="stat-label">Reranker</span>
              <span className="stat-value stat-value-sm">{meta.rerank_backend.toUpperCase()}</span>
              <span className="stat-unit">α = {meta.alpha}</span>
            </div>
          )}
        </div>
      )}

      <div className="insights-grid">
        {/* 1. Final Scores */}
        {finalScores.length > 0 && (
          <div className="clay-card chart-card animate-in-delay-1">
            <h3 className="chart-title">
              <HiOutlineSparkles className="chart-title-icon" />
              Final Scores — Top Patents
            </h3>
            <p className="chart-desc">Combined final score for each top result</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={finalScores} margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis dataKey="patent_id" tick={{ fontSize: 10, fill: '#6b6880' }} angle={-25} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 11, fill: '#6b6880' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="final_score" radius={[8, 8, 0, 0]} maxBarSize={50}>
                    {finalScores.map((_, idx) => (
                      <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                    ))}
                    <LabelList dataKey="final_score" position="top" formatter={(v) => v.toFixed(3)} style={{ fontSize: 11, fill: '#6b6880' }} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 2. Pipeline Funnel */}
        {funnel.length > 0 && (
          <div className="clay-card chart-card animate-in-delay-2">
            <h3 className="chart-title">
              <HiOutlineFunnel className="chart-title-icon" />
              Pipeline Funnel
            </h3>
            <p className="chart-desc">Number of patents at each pipeline stage</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={funnel} layout="vertical" margin={{ top: 10, right: 50, bottom: 10, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis type="number" tick={{ fontSize: 11, fill: '#6b6880' }} />
                  <YAxis type="category" dataKey="stage" tick={{ fontSize: 11, fill: '#2d2b3a', fontWeight: 500 }} width={120} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" radius={[0, 8, 8, 0]} maxBarSize={36}>
                    {funnel.map((_, idx) => (
                      <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                    ))}
                    <LabelList dataKey="count" position="right" style={{ fontSize: 12, fontWeight: 700, fill: '#2d2b3a' }} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 3. Score Breakdown */}
        {scoreBreakdown.length > 0 && (
          <div className="clay-card chart-card animate-in-delay-1">
            <h3 className="chart-title">
              <HiOutlineChartBar className="chart-title-icon" />
              Score Breakdown
            </h3>
            <p className="chart-desc">Rerank, retrieval, and citation scores per patent</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={scoreBreakdown} margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#6b6880' }} angle={-25} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 11, fill: '#6b6880' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  <Bar dataKey="rerank_score" name="Rerank" fill="#7c5cfc" radius={[4, 4, 0, 0]} maxBarSize={24} />
                  <Bar dataKey="retrieval_score" name="Retrieval" fill="#22c55e" radius={[4, 4, 0, 0]} maxBarSize={24} />
                  <Bar dataKey="citation_score" name="Citation" fill="#f59e0b" radius={[4, 4, 0, 0]} maxBarSize={24} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}


        {/* 5. Forward vs Backward Citations — Scatter */}
        {citationScatter.length > 0 && (
          <div className="clay-card chart-card animate-in-delay-1">
            <h3 className="chart-title">
              <HiOutlineLink className="chart-title-icon" />
              Citation Profile — Forward vs Backward
            </h3>
            <p className="chart-desc">Each dot is a patent. Size = citation score</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis type="number" dataKey="backward" name="Backward Citations" tick={{ fontSize: 11, fill: '#6b6880' }}
                    label={{ value: 'Backward Citations', position: 'bottom', fontSize: 11, fill: '#6b6880', offset: -2 }} />
                  <YAxis type="number" dataKey="forward" name="Forward Citations" tick={{ fontSize: 11, fill: '#6b6880' }}
                    label={{ value: 'Forward', angle: -90, position: 'left', fontSize: 11, fill: '#6b6880' }} />
                  <ZAxis type="number" dataKey="citation_score" range={[40, 400]} />
                  <Tooltip content={<ScatterTooltip />} />
                  <Scatter data={citationScatter} fill="#7c5cfc" fillOpacity={0.6} stroke="#7c5cfc" strokeWidth={1} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 6. PageRank Scores */}
        {finalScores.length > 0 && finalScores[0].pagerank_score != null && (
          <div className="clay-card chart-card animate-in-delay-2">
            <h3 className="chart-title">
              <HiOutlineServerStack className="chart-title-icon" />
              PageRank Scores
            </h3>
            <p className="chart-desc">Network influence ranking of top patents</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={finalScores} layout="vertical" margin={{ top: 10, right: 50, bottom: 10, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis type="number" tick={{ fontSize: 10, fill: '#6b6880' }} tickFormatter={(v) => v.toFixed(5)} />
                  <YAxis type="category" dataKey="patent_id" tick={{ fontSize: 10, fill: '#2d2b3a' }} width={120} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="pagerank_score" name="PageRank" fill="#14b8a6" radius={[0, 8, 8, 0]} maxBarSize={30}>
                    <LabelList dataKey="pagerank_score" position="right" formatter={(v) => v.toFixed(6)} style={{ fontSize: 10, fontWeight: 600, fill: '#2d2b3a' }} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 7. Keyword Scores */}
        {keywordScores.length > 0 && (
          <div className="clay-card chart-card animate-in-delay-1">
            <h3 className="chart-title">
              <HiOutlineKey className="chart-title-icon" />
              Keyword Relevance Scores
            </h3>
            <p className="chart-desc">Weighted keyword match scores from Stage 2</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={keywordScores} margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis dataKey="title" tick={{ fontSize: 9, fill: '#6b6880' }} angle={-30} textAnchor="end" height={80} />
                  <YAxis tick={{ fontSize: 11, fill: '#6b6880' }} allowDecimals={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="keyword_score" name="Keyword Score" fill="#8b5cf6" radius={[6, 6, 0, 0]} maxBarSize={40}>
                    <LabelList dataKey="keyword_score" position="top" style={{ fontSize: 11, fontWeight: 600, fill: '#2d2b3a' }} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 8. IPC Code Distribution */}
        {ipcDist.length > 0 && (
          <div className="clay-card chart-card animate-in-delay-2">
            <h3 className="chart-title">
              <HiOutlineCircleStack className="chart-title-icon" />
              IPC Code Distribution
            </h3>
            <p className="chart-desc">Technology domain breakdown from classified patents</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={ipcDist} margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis dataKey="code" tick={{ fontSize: 10, fill: '#6b6880', fontWeight: 600 }} angle={-30} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 11, fill: '#6b6880' }} allowDecimals={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" name="Patents" radius={[6, 6, 0, 0]} maxBarSize={40}>
                    {ipcDist.map((_, idx) => (
                      <Cell key={idx} fill={BRIGHT_COLORS[idx % BRIGHT_COLORS.length]} />
                    ))}
                    <LabelList dataKey="count" position="top" style={{ fontSize: 11, fontWeight: 700, fill: '#2d2b3a' }} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 9. Hop Distance Distribution — Pie */}
        {hopDist.length > 0 && (
          <div className="clay-card chart-card animate-in-delay-1">
            <h3 className="chart-title">
              <HiOutlineMagnifyingGlass className="chart-title-icon" />
              BFS Hop Distance Distribution
            </h3>
            <p className="chart-desc">How many patents found at each BFS depth from citation search</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={hopDist}
                    dataKey="count"
                    nameKey="hop"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    innerRadius={45}
                    paddingAngle={3}
                    label={renderPieLabel}
                    labelLine={{ stroke: '#c4b5fd', strokeWidth: 1 }}
                  >
                    {hopDist.map((_, idx) => (
                      <Cell key={idx} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 10. Seed vs BFS Discovered — Pie */}
        {seedVsDiscovered.length > 0 && (
          <div className="clay-card chart-card animate-in-delay-2">
            <h3 className="chart-title">
              <HiOutlineFunnel className="chart-title-icon" />
              Seed vs BFS Discovered
            </h3>
            <p className="chart-desc">What % of citation-expanded patents were original keyword seeds</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={seedVsDiscovered}
                    dataKey="count"
                    nameKey="type"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    innerRadius={45}
                    paddingAngle={4}
                    label={renderPieLabel}
                    labelLine={{ stroke: '#c4b5fd', strokeWidth: 1 }}
                  >
                    {seedVsDiscovered.map((_, idx) => (
                      <Cell key={idx} fill={[BRIGHT_COLORS[0], BRIGHT_COLORS[1]][idx]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 11. Publication Year */}
        {yearData.length > 0 && (
          <div className="clay-card chart-card animate-in-delay-1">
            <h3 className="chart-title">
              <HiOutlineChartBar className="chart-title-icon" />
              Publication Year Distribution
            </h3>
            <p className="chart-desc">Top result patents by publication year</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={yearData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis dataKey="year" tick={{ fontSize: 12, fill: '#6b6880' }} />
                  <YAxis tick={{ fontSize: 11, fill: '#6b6880' }} allowDecimals={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" name="Patents" fill="#7c5cfc" radius={[8, 8, 0, 0]} maxBarSize={50}>
                    <LabelList dataKey="count" position="top" style={{ fontSize: 13, fontWeight: 700, fill: '#2d2b3a' }} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 12. Forward vs Backward Citations Bar */}
        {finalScores.length > 0 && finalScores[0].forward_citations != null && (
          <div className="clay-card chart-card animate-in-delay-2">
            <h3 className="chart-title">
              <HiOutlineLink className="chart-title-icon" />
              Citation Counts — Top Patents
            </h3>
            <p className="chart-desc">Forward (cited by others) vs backward (cites others) for final results</p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={finalScores} margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis dataKey="patent_id" tick={{ fontSize: 10, fill: '#6b6880' }} angle={-25} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 11, fill: '#6b6880' }} allowDecimals={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  <Bar dataKey="forward_citations" name="Forward" fill="#3b82f6" radius={[4, 4, 0, 0]} maxBarSize={24} />
                  <Bar dataKey="backward_citations" name="Backward" fill="#f59e0b" radius={[4, 4, 0, 0]} maxBarSize={24} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* Score Cards */}
      {finalScores.length > 0 && (
        <section className="score-cards-section animate-in-delay-3">
          <h2 className="section-title" style={{ marginTop: 'var(--space-2xl)' }}>Top Patent Scores</h2>
          <div className="score-cards-grid">
            {finalScores.map((p, idx) => (
              <div key={idx} className="clay-card-sm score-card">
                <div className="score-card-rank">#{idx + 1}</div>
                <div className="score-card-id">{p.patent_id_full}</div>
                <div className="score-card-title">{p.title}</div>
                <div className="score-card-scores">
                  <div className="score-item">
                    <span className="score-label">Final</span>
                    <span className="score-value" style={{ color: 'var(--accent)' }}>{p.final_score.toFixed(4)}</span>
                  </div>
                  <div className="score-item">
                    <span className="score-label">Rerank</span>
                    <span className="score-value">{p.rerank_score.toFixed(3)}</span>
                  </div>
                  <div className="score-item">
                    <span className="score-label">Retrieval</span>
                    <span className="score-value">{p.retrieval_score.toFixed(4)}</span>
                  </div>
                  <div className="score-item">
                    <span className="score-label">Citation</span>
                    <span className="score-value">{p.citation_score.toFixed(4)}</span>
                  </div>
                  <div className="score-item">
                    <span className="score-label">PageRank</span>
                    <span className="score-value">{p.pagerank_score.toFixed(6)}</span>
                  </div>
                  <div className="score-item">
                    <span className="score-label">Year</span>
                    <span className="score-value">{p.publication_year || '—'}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
