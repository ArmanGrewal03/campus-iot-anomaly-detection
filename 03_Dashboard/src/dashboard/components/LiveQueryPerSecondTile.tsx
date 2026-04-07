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

interface MetricsState {
  data: number[];
  labels: string[];
  loading: boolean;
  error: string | null;
}

function metricsReducer(
  state: MetricsState,
  action: { type: string; payload?: Partial<MetricsState> }
): MetricsState {
  if (action.type === 'SET_METRICS' && action.payload) {
    return { ...state, ...action.payload };
  }
  return state;
}

function LiveQueryPerSecondTileComponent() {
  const theme = useTheme();
  const [state, dispatch] = React.useReducer(metricsReducer, {
    data: [],
    labels: [],
    loading: true,
    error: null,
  });
  // Defer state updates so they don't block animation frames
  const deferredState = React.useDeferredValue(state);

  React.useEffect(() => {
    let cancelled = false;
    let idleCallbackId: number | null = null;
    
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
          const newLabels = Array.isArray(json.labels) ? json.labels : json.query_per_second.map((_: number, i: number) => String(i));
          dispatch({
            type: 'SET_METRICS',
            payload: {
              data: json.query_per_second,
              labels: newLabels,
              error: null,
              loading: false,
            },
          });
        }
      } catch (e) {
        if (!cancelled) {
          dispatch({
            type: 'SET_METRICS',
            payload: { error: 'Service unavailable', data: [], labels: [], loading: false },
          });
        }
      }
    };
    
    const scheduleFetch = () => {
      if (cancelled) return;
      // Use requestIdleCallback to defer fetch to idle time (doesn't block animation frame)
      idleCallbackId = requestIdleCallback(
        () => {
          fetchMetrics();
          // Schedule next fetch after idle completes
          setTimeout(() => {
            if (!cancelled) scheduleFetch();
          }, 2000);
        },
        { timeout: 3000 } // Fallback timeout to ensure fetch happens
      );
    };
    
    scheduleFetch();
    return () => {
      cancelled = true;
      if (idleCallbackId !== null) cancelIdleCallback(idleCallbackId);
    };
  }, []);

  const xLabels = React.useMemo(
    () => (deferredState.labels.length === deferredState.data.length ? deferredState.labels : deferredState.data.map((_, i) => String(i))),
    [deferredState.labels, deferredState.data.length]
  );
  const currentQps = React.useMemo(() => (deferredState.data.length > 0 ? deferredState.data[deferredState.data.length - 1] : 0), [deferredState.data]);
  const teal = theme.palette.mode === 'light' ? '#009688' : '#4db6ac';
  
  // Memoize chart config
  const chartConfig = React.useMemo(
    () => ({
      xAxis: [{ scaleType: 'band' as const, data: xLabels, tickLabelStyle: { fontSize: 9 }, tickInterval: (_: any, i: number) => i % 12 === 0 || i === xLabels.length - 1 }],
      yAxis: [{ tickMinStep: 1, valueFormatter: (v: any) => String(Math.round(Number(v))) }],
      series: [{ id: 'qps', data: deferredState.data, label: 'QPS', color: teal }],
      margin: { top: 8, right: 8, bottom: 24, left: 28 } as const,
    }),
    [xLabels, deferredState.data, teal]
  );

  if (deferredState.loading && deferredState.data.length === 0) {
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
        {deferredState.error && (
          <Typography variant="caption" color="error" sx={{ display: 'block', mb: 1 }}>{deferredState.error}</Typography>
        )}
        {deferredState.data.length === 0 ? (
          <Box sx={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography variant="caption" color="text.secondary">No data</Typography>
          </Box>
        ) : (
          <BarChart
            xAxis={chartConfig.xAxis}
            yAxis={chartConfig.yAxis}
            series={chartConfig.series}
            height={180}
            margin={chartConfig.margin}
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

export default React.memo(LiveQueryPerSecondTileComponent);
