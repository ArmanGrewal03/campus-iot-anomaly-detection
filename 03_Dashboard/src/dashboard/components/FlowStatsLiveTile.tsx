import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import InsightsRoundedIcon from '@mui/icons-material/InsightsRounded';

const GATEWAY_BASE = 'http://127.0.0.1:8003';

interface ModelInfo {
  model_name: string;
  model_file?: string;
  accuracy?: number;
  n_features?: number;
  training_date?: string;
}

interface ModelStatus {
  status?: string;
  model_type?: string;
  n_features?: number;
  last_test_date?: string;
}

interface ModelMetrics {
  metrics?: Record<string, unknown>;
}

function formatPercent(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  const pct = n <= 1 ? n * 100 : n;
  return `${pct.toFixed(1)}%`;
}

function inferModelType(name: string | undefined): string {
  if (!name) return '—';
  const n = name.toLowerCase();
  if (n.includes('rf')) return 'RFv1';
  if (n.includes('if')) return 'IFv1';
  if (n.includes('ae')) return 'AEv1';
  if (n.includes('cnn')) return 'CNN';
  if (n.includes('xgboost') || n.includes('xgb')) return 'XGBOOST';
  if (n.includes('lightgbm') || n.includes('lgbm')) return 'LIGHTGBM';
  if (n.includes('knn')) return 'KNN';
  if (n.includes('kmeans')) return 'KMEANS';
  return '—';
}

function pickMetric(metrics: Record<string, unknown> | undefined, keys: string[]): unknown {
  if (!metrics) return undefined;
  for (const key of keys) {
    if (metrics[key] != null) return metrics[key];
  }
  return undefined;
}

export default function FlowStatsLiveTile() {
  const [selectedModelName, setSelectedModelName] = React.useState<string>('—');
  const [modelInfo, setModelInfo] = React.useState<ModelInfo | null>(null);
  const [status, setStatus] = React.useState<ModelStatus | null>(null);
  const [metrics, setMetrics] = React.useState<ModelMetrics | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    const fetchModelSnapshot = async () => {
      try {
        const [selectedRes, modelsRes] = await Promise.all([
          fetch(`${GATEWAY_BASE}/get-model?t=${Date.now()}`),
          fetch(`${GATEWAY_BASE}/models?t=${Date.now()}`),
        ]);
        const selectedJson = await selectedRes.json() as { status?: string; model_name?: string };
        const modelsJson = await modelsRes.json() as { status?: string; models?: ModelInfo[] };
        if (cancelled) return;
        const selectedName = selectedRes.ok && selectedJson.status === 'success' && selectedJson.model_name
          ? selectedJson.model_name
          : (Array.isArray(modelsJson.models) && modelsJson.models[0]?.model_name) || '—';
        const activeModel = Array.isArray(modelsJson.models)
          ? modelsJson.models.find((model) => model.model_name === selectedName) ?? modelsJson.models[0] ?? null
          : null;
        const resolvedModelName = activeModel?.model_name ?? selectedName;

        setSelectedModelName(resolvedModelName);
        setModelInfo(activeModel);

        if (resolvedModelName && resolvedModelName !== '—') {
          const headers: Record<string, string> = { model_name: resolvedModelName };
          const [statusRes, metricsRes] = await Promise.all([
            fetch(`${GATEWAY_BASE}/model/status?t=${Date.now()}`, { headers }),
            fetch(`${GATEWAY_BASE}/model/metrics?t=${Date.now()}`, { headers }).catch(() => null),
          ]);
          if (cancelled) return;
          setStatus(statusRes.ok ? await statusRes.json() as ModelStatus : null);
          setMetrics(metricsRes?.ok ? await metricsRes.json() as ModelMetrics : null);
        } else {
          setStatus(null);
          setMetrics(null);
        }
      } catch {
        if (!cancelled) {
          setSelectedModelName('—');
          setModelInfo(null);
          setStatus(null);
          setMetrics(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchModelSnapshot();
    const interval = setInterval(fetchModelSnapshot, 15000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (loading) {
    return (
      <Card variant="outlined" sx={{ height: '100%', minHeight: 160 }}>
        <CardContent>
          <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
            <InsightsRoundedIcon sx={{ color: 'info.main', fontSize: 20 }} />
            <Typography variant="subtitle2">Selected model performance</Typography>
          </Stack>
          <Stack alignItems="center" justifyContent="center" sx={{ minHeight: 100 }}>
            <CircularProgress size={28} />
          </Stack>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      variant="outlined"
      sx={{
        height: '100%',
        minHeight: 160,
        borderTop: '3px solid',
        borderTopColor: 'info.main',
      }}
    >
      <CardContent>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <InsightsRoundedIcon sx={{ color: 'info.main', fontSize: 20 }} />
            <Typography variant="subtitle2" fontWeight={600}>
              Selected model performance
            </Typography>
          </Stack>
          <Chip size="small" label="Active" color="primary" sx={{ fontSize: '0.625rem', height: 18 }} />
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          {selectedModelName === '—'
            ? 'No model selected'
            : `Current model: ${selectedModelName}`}
        </Typography>
        <Stack spacing={0.5}>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">Accuracy</Typography>
            <Typography variant="body2" fontWeight={600}>
              {pickMetric(metrics?.metrics, ['accuracy']) != null
                ? formatPercent(pickMetric(metrics?.metrics, ['accuracy']))
                : modelInfo?.accuracy != null
                  ? formatPercent(modelInfo.accuracy)
                  : '—'}
            </Typography>
          </Stack>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">F1 score</Typography>
            <Typography variant="body2" fontWeight={600}>
              {pickMetric(metrics?.metrics, ['f1', 'f1_score', 'f1Score']) != null
                ? formatPercent(pickMetric(metrics?.metrics, ['f1', 'f1_score', 'f1Score']))
                : '—'}
            </Typography>
          </Stack>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">Model type</Typography>
            <Typography variant="body2" fontWeight={600}>{status?.model_type || inferModelType(selectedModelName)}</Typography>
          </Stack>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">Features</Typography>
            <Typography variant="body2" fontWeight={600}>{status?.n_features ?? modelInfo?.n_features ?? '—'}</Typography>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
