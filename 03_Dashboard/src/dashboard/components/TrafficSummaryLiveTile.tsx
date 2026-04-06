import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import ViewModuleRoundedIcon from '@mui/icons-material/ViewModuleRounded';

const GATEWAY_BASE = 'http://127.0.0.1:8003';

interface ModelInfo {
  model_name: string;
  model_file?: string;
  accuracy?: number;
  n_features?: number;
  training_date?: string;
}

export default function TrafficSummaryLiveTile() {
  const [summary, setSummary] = React.useState<{
    totalModels: number;
    selectedModel: string;
    latestModel: string | null;
    latestTrainingDate: string | null;
    bestAccuracy: number | null;
  }>({
    totalModels: 0,
    selectedModel: '—',
    latestModel: null,
    latestTrainingDate: null,
    bestAccuracy: null,
  });
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;

    const fetchLibrary = async () => {
      try {
        const [selectedRes, modelsRes] = await Promise.all([
          fetch(`${GATEWAY_BASE}/get-model?t=${Date.now()}`),
          fetch(`${GATEWAY_BASE}/models?t=${Date.now()}`),
        ]);

        const selectedJson = await selectedRes.json() as { status?: string; model_name?: string };
        const modelsJson = await modelsRes.json() as { status?: string; models?: ModelInfo[]; total_models?: number };
        if (cancelled) return;

        const models = Array.isArray(modelsJson.models) ? modelsJson.models : [];
        const latestModel = [...models].sort((a, b) => {
          const aTime = a.training_date ? Date.parse(a.training_date) : 0;
          const bTime = b.training_date ? Date.parse(b.training_date) : 0;
          return bTime - aTime;
        })[0] ?? null;

        const bestAccuracy = models.reduce<number | null>((best, model) => {
          if (model.accuracy == null) return best;
          const accuracyValue = Number(model.accuracy);
          if (!Number.isFinite(accuracyValue)) return best;
          return best == null || accuracyValue > best ? accuracyValue : best;
        }, null);

        setSummary({
          totalModels: modelsJson.total_models ?? models.length,
          selectedModel: (selectedRes.ok && selectedJson.status === 'success' && selectedJson.model_name) || '—',
          latestModel: latestModel?.model_name ?? null,
          latestTrainingDate: latestModel?.training_date ?? null,
          bestAccuracy,
        });
      } catch {
        if (!cancelled) {
          setSummary({
            totalModels: 0,
            selectedModel: '—',
            latestModel: null,
            latestTrainingDate: null,
            bestAccuracy: null,
          });
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchLibrary();
    const interval = setInterval(fetchLibrary, 20000);
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
            <ViewModuleRoundedIcon sx={{ color: 'secondary.main', fontSize: 20 }} />
            <Typography variant="subtitle2">Model library</Typography>
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
        borderTopColor: 'secondary.main',
      }}
    >
      <CardContent>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <ViewModuleRoundedIcon sx={{ color: 'secondary.main', fontSize: 20 }} />
            <Typography variant="subtitle2" fontWeight={600}>
              Model library
            </Typography>
          </Stack>
          <Chip size="small" label="Live" color="secondary" sx={{ fontSize: '0.625rem', height: 18 }} />
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          {summary.totalModels.toLocaleString()} trained models available
        </Typography>
        <Stack spacing={0.75}>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">Selected model</Typography>
            <Typography variant="body2" fontWeight={600} noWrap title={summary.selectedModel}>
              {summary.selectedModel}
            </Typography>
          </Stack>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">Latest model</Typography>
            <Typography variant="body2" fontWeight={600} noWrap title={summary.latestModel || ''}>
              {summary.latestModel ?? '—'}
            </Typography>
          </Stack>
        </Stack>
        <Box sx={{ mt: 1, height: 6, borderRadius: 1, bgcolor: 'action.hover', overflow: 'hidden', display: 'flex' }}>
          <Box
            sx={{
              width: `${Math.min(100, Math.max(0, summary.bestAccuracy != null ? summary.bestAccuracy * 100 : 0))}%`,
              bgcolor: 'secondary.main',
              borderRadius: 1,
            }}
          />
          <Box sx={{ flex: 1, bgcolor: 'info.main', borderRadius: 1 }} />
        </Box>
        <Stack direction="row" spacing={1} sx={{ mt: 1 }} flexWrap="wrap">
          <Chip
            size="small"
            label={summary.latestTrainingDate ? `Trained: ${summary.latestTrainingDate}` : 'Trained: —'}
            variant="outlined"
            sx={{ fontSize: '0.65rem', height: 20 }}
          />
          <Chip
            size="small"
            label={summary.bestAccuracy != null ? `Best acc: ${(summary.bestAccuracy * 100).toFixed(1)}%` : 'Best acc: —'}
            variant="outlined"
            sx={{ fontSize: '0.65rem', height: 20 }}
          />
        </Stack>
      </CardContent>
    </Card>
  );
}
