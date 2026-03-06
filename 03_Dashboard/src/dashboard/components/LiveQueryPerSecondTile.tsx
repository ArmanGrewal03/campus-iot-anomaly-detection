import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import { BarChart } from '@mui/x-charts/BarChart';

const LIVE_METRICS_BASE = 'http://127.0.0.1:8010';

export default function LiveQueryPerSecondTile() {
  const theme = useTheme();
  const [data, setData] = React.useState<number[]>([]);
  const [labels, setLabels] = React.useState<string[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    const fetchMetrics = async () => {
      try {
        const res = await fetch(`${LIVE_METRICS_BASE}/metrics`);
        const json = await res.json() as {
          status?: string;
          query_per_second?: number[];
          labels?: string[];
        };
        if (cancelled) return;
        if (json.status === 'success' && Array.isArray(json.query_per_second)) {
          setData(json.query_per_second);
          setLabels(Array.isArray(json.labels) ? json.labels : json.query_per_second.map((_: number, i: number) => String(i)));
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError('Service unavailable');
          setData([]);
          setLabels([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const xLabels = labels.length === data.length ? labels : data.map((_, i) => String(i));
  const currentQps = data.length > 0 ? data[data.length - 1] : 0;
  const teal = theme.palette.mode === 'light' ? '#009688' : '#4db6ac';

  if (loading && data.length === 0) {
    return (
      <Card variant="outlined" sx={{ height: '100%', minHeight: 220 }}>
        <CardContent>
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
            <Typography variant="subtitle2" fontWeight={600}>Query Per Second</Typography>
            <CircularProgress size={18} />
          </Stack>
          <Stack alignItems="center" justifyContent="center" sx={{ minHeight: 160 }}>
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
        minHeight: 220,
        borderTop: '3px solid',
        borderTopColor: 'success.main',
      }}
    >
      <CardContent sx={{ '&:last-child': { pb: 1.5 } }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
          <Stack direction="row" alignItems="center" spacing={0.5}>
            <Typography variant="subtitle2" fontWeight={600}>Query Per Second</Typography>
            <RefreshRoundedIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
          </Stack>
          <Stack direction="row" alignItems="center" spacing={1}>
            <Typography variant="caption" color="text.secondary">QPS <strong>{currentQps}</strong></Typography>
            <Chip size="small" label="Live" color="error" sx={{ fontSize: '0.625rem', height: 18 }} />
          </Stack>
        </Stack>
        {error && (
          <Typography variant="caption" color="error" sx={{ display: 'block', mb: 1 }}>{error}</Typography>
        )}
        {data.length === 0 ? (
          <Box sx={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography variant="caption" color="text.secondary">No data</Typography>
          </Box>
        ) : (
          <BarChart
            xAxis={[{ scaleType: 'band', data: xLabels, tickLabelStyle: { fontSize: 9 }, tickInterval: (_, i) => i % 12 === 0 || i === xLabels.length - 1 }]}
            yAxis={[{ tickMinStep: 1, valueFormatter: (v) => String(Math.round(Number(v))) }]}
            series={[{ id: 'qps', data, label: 'QPS', color: teal }]}
            height={180}
            margin={{ top: 8, right: 8, bottom: 24, left: 28 }}
            grid={{ vertical: false, horizontal: true }}
            borderRadius={2}
            slotProps={{ legend: { hidden: true } }}
          />
        )}
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
          Last 2 min · per 2s interval · refreshes every 2s
        </Typography>
      </CardContent>
    </Card>
  );
}
