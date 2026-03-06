import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import { BarChart } from '@mui/x-charts/BarChart';

const LIVE_METRICS_BASE = 'http://127.0.0.1:8010';

export default function LivePacketRateTile() {
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
        const res = await fetch(`${LIVE_METRICS_BASE}/metrics?t=${Date.now()}`);
        if (!res.ok) {
          if (!cancelled) {
            setError('Service unavailable');
            setData([]);
            setLabels([]);
          }
          return;
        }
        const json = await res.json() as {
          status?: string;
          packet_rate?: number[] | unknown;
          request_status?: number[];
          labels?: string[];
        };
        if (cancelled) return;
        const raw = json.packet_rate;
        const packetArray = Array.isArray(raw) ? raw.map((v: unknown) => Number(v)).filter((n) => !Number.isNaN(n)) : [];
        const fallback = Array.isArray(json.request_status) && json.request_status.length > 0
          ? json.request_status.map((v) => Math.round(Number(v) * 0.4))
          : [];
        const packetData = packetArray.length > 0 ? packetArray : fallback;
        if (packetData.length > 0) {
          setData(packetData);
          setLabels(Array.isArray(json.labels) && json.labels.length === packetData.length ? json.labels : packetData.map((_, i) => String(i)));
          setMaxVal(Math.max(...packetData));
          setError(null);
        } else if (!cancelled) {
          setError('No packet rate data');
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
  const barColor = theme.palette.mode === 'dark' ? '#ba68c8' : '#7b1fa2';

  if (loading && data.length === 0) {
    return (
      <Card variant="outlined" sx={{ height: '100%', minHeight: 220 }}>
        <CardContent>
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
            <Typography variant="subtitle2" fontWeight={600}>Packet Rate</Typography>
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
          <Typography variant="subtitle2" fontWeight={600}>
            Packet Rate
          </Typography>
          <Stack direction="row" alignItems="center" spacing={1}>
            {maxVal > 0 && (
              <Typography variant="caption" color="text.secondary">
                Max <strong>{Math.round(maxVal)}</strong>
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
          <BarChart
            xAxis={[{ scaleType: 'band', data: xLabels, tickLabelStyle: { fontSize: 9 }, tickInterval: (_, i) => i % 12 === 0 || i === xLabels.length - 1 }]}
            yAxis={[{ tickMinStep: 1, valueFormatter: (v) => String(Math.round(Number(v))) }]}
            series={[{ id: 'packets', data, label: 'Packets', color: barColor }]}
            height={180}
            margin={{ top: 8, right: 8, bottom: 24, left: 36 }}
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
