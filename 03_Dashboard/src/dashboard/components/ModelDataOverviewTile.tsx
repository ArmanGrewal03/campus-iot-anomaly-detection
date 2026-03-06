import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import PsychologyRoundedIcon from '@mui/icons-material/PsychologyRounded';

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
  metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
  };
}

export default function ModelDataOverviewTile() {
  const [model, setModel] = React.useState<ModelInfo | null>(null);
  const [status, setStatus] = React.useState<ModelStatus | null>(null);
  const [metrics, setMetrics] = React.useState<ModelMetrics | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    const fetchOverview = async () => {
      try {
        setError(null);
        const modelsRes = await fetch(`${GATEWAY_BASE}/models`);
        const modelsJson = await modelsRes.json() as { status?: string; models?: ModelInfo[]; detail?: string };
        if (cancelled) return;
        if (!modelsRes.ok || modelsJson.status !== 'success' || !Array.isArray(modelsJson.models) || modelsJson.models.length === 0) {
          setModel(null);
          setStatus(null);
          setMetrics(null);
          return;
        }
        const first = modelsJson.models[0];
        setModel(first);
        const headers: Record<string, string> = { model_name: first.model_name };
        const [statusRes, metricsRes] = await Promise.all([
          fetch(`${GATEWAY_BASE}/model/status`, { headers }),
          fetch(`${GATEWAY_BASE}/model/metrics`, { headers }).catch(() => null),
        ]);
        if (cancelled) return;
        if (statusRes.ok) {
          const statusJson = await statusRes.json() as ModelStatus;
          setStatus(statusJson);
        } else {
          setStatus(null);
        }
        if (metricsRes?.ok) {
          const metricsJson = await metricsRes.json() as ModelMetrics;
          setMetrics(metricsJson);
        } else {
          setMetrics(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError('Model API unavailable');
          setModel(null);
          setStatus(null);
          setMetrics(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchOverview();
    const interval = setInterval(fetchOverview, 15000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const accuracy = metrics?.metrics?.accuracy != null
    ? metrics.metrics.accuracy
    : model?.accuracy;
  const f1 = metrics?.metrics?.f1;

  return (
    <Card variant="outlined" sx={{ height: '100%', borderLeft: '4px solid', borderLeftColor: 'primary.main' }}>
      <CardContent sx={{ '&:last-child': { pb: 1.5 } }}>
        <Stack direction="row" alignItems="center" spacing={0.5} sx={{ mb: 1 }}>
          <PsychologyRoundedIcon sx={{ fontSize: 18, color: 'primary.main' }} />
          <Typography variant="subtitle2" fontWeight={600}>
            Model Data Overview
          </Typography>
        </Stack>
        {loading ? (
          <Stack alignItems="center" justifyContent="center" sx={{ py: 2 }}>
            <CircularProgress size={24} />
          </Stack>
        ) : error ? (
          <Typography variant="caption" color="text.secondary">
            {error}
          </Typography>
        ) : !model ? (
          <Typography variant="caption" color="text.secondary">
            No models available
          </Typography>
        ) : (
          <Stack spacing={1.5}>
            <Box>
              <Typography variant="caption" color="text.secondary" display="block">Model</Typography>
              <Typography variant="body2" fontWeight={600} noWrap title={model.model_name}>
                {model.model_name}
              </Typography>
            </Box>
            {status && (
              <Box>
                <Typography variant="caption" color="text.secondary" display="block">Status</Typography>
                <Chip
                  size="small"
                  label={status.status === 'trained' ? 'Trained' : status.status || '—'}
                  color={status.status === 'trained' ? 'success' : 'default'}
                  sx={{ height: 20, fontSize: '0.7rem' }}
                />
              </Box>
            )}
            {accuracy != null && (
              <Box>
                <Typography variant="caption" color="text.secondary" display="block">Accuracy</Typography>
                <Typography variant="body2" fontWeight={600}>
                  {(Number(accuracy) * 100).toFixed(1)}%
                </Typography>
              </Box>
            )}
            {f1 != null && (
              <Box>
                <Typography variant="caption" color="text.secondary" display="block">F1 Score</Typography>
                <Typography variant="body2" fontWeight={600}>
                  {(Number(f1) * 100).toFixed(1)}%
                </Typography>
              </Box>
            )}
            {status?.model_type && (
              <Box>
                <Typography variant="caption" color="text.secondary" display="block">Type</Typography>
                <Typography variant="body2" noWrap title={status.model_type}>
                  {status.model_type}
                </Typography>
              </Box>
            )}
          </Stack>
        )}
      </CardContent>
    </Card>
  );
}
