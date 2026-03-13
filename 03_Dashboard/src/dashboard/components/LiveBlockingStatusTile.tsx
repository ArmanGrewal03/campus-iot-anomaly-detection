import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import { LineChart } from '@mui/x-charts/LineChart';

const LIVE_METRICS_BASE = 'http://127.0.0.1:8010';

function AreaGradient({ color, id }: { color: string; id: string }) {
  return (
    <defs>
      <linearGradient id={id} x1="50%" y1="0%" x2="50%" y2="100%">
        <stop offset="0%" stopColor={color} stopOpacity={0.55} />
        <stop offset="100%" stopColor={color} stopOpacity={0} />
      </linearGradient>
    </defs>
  );
}

export default function LiveBlockingStatusTile() {
  const theme = useTheme();
  const [data, setData] = React.useState<number[]>([]);
  const [labels, setLabels] = React.useState<string[]>([]);
  const [maxVal, setMaxVal] = React.useState<number>(0);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    const fetchMetrics = async () => {
      try {
        const res = await fetch(`${LIVE_METRICS_BASE}/metrics`);
        const json = await res.json() as {
          status?: string;
          blocking_status?: number[];
          labels?: string[];
          max_blocking_status?: number;
        };
        if (cancelled) return;
        if (json.status === 'success' && Array.isArray(json.blocking_status)) {
          setData(json.blocking_status);
          setLabels(Array.isArray(json.labels) ? json.labels : json.blocking_status.map((_: number, i: number) => String(i)));
          setMaxVal(json.max_blocking_status ?? Math.max(...json.blocking_status));
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
  const warmColor = theme.palette.mode === 'light' ? '#ff9800' : '#ffb74d';
  const gradientId = 'live-blocking-gradient';

  if (loading && data.length === 0) {
    return (
      <Card variant="outlined" sx={{ height: '100%', minHeight: 220 }}>
        <CardContent>
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
            <Typography variant="subtitle2" fontWeight={600}>Blocking Status</Typography>
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
        borderTopColor: 'warning.main',
      }}
    >
      <CardContent sx={{ '&:last-child': { pb: 1.5 } }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
          <Typography variant="subtitle2" fontWeight={600}>
            Blocking Status
          </Typography>
          <Stack direction="row" alignItems="center" spacing={1}>
            {maxVal > 0 && (
              <Typography variant="caption" color="text.secondary">
                Max <strong>{maxVal}</strong>
              </Typography>
            )}
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
          <LineChart
            xAxis={[{ scaleType: 'point', data: xLabels, tickLabelStyle: { fontSize: 9 }, tickInterval: (_, i) => i % 12 === 0 || i === xLabels.length - 1 }]}
            yAxis={[{ tickMinStep: 1, valueFormatter: (v) => String(Math.round(Number(v))) }]}
            series={[{
              id: 'blocking',
              label: 'Blocking',
              data,
              showMark: false,
              curve: 'natural',
              area: true,
              color: warmColor,
            }]}
            height={180}
            margin={{ top: 8, right: 8, bottom: 24, left: 36 }}
            grid={{ vertical: false, horizontal: true }}
            sx={{
              '& .MuiAreaElement-root': { fill: `url(#${gradientId})` },
              '& .MuiLineElement-root': { strokeWidth: 2 },
            }}
            slotProps={{ legend: { hidden: true } }}
          >
            <AreaGradient color={warmColor} id={gradientId} />
          </LineChart>
        )}
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
          Last 2 min · per 2s interval · refreshes every 2s
        </Typography>
      </CardContent>
    </Card>
  );
}
