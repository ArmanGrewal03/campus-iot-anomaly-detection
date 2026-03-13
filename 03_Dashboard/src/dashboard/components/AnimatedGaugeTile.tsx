import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import SpeedRoundedIcon from '@mui/icons-material/SpeedRounded';

const GATEWAY_BASE = 'http://127.0.0.1:8003';

export default function AnimatedGaugeTile() {
  const [anomalyRate, setAnomalyRate] = React.useState(0);
  const [loading, setLoading] = React.useState(true);
  const displayRate = React.useRef(0);
  const rafRef = React.useRef<number>();

  React.useEffect(() => {
    let cancelled = false;
    const fetchKpis = async () => {
      try {
        const res = await fetch(`${GATEWAY_BASE}/dashboard-kpis?t=${Date.now()}`);
        const json = await res.json() as { status?: string; anomaly_rate?: number };
        if (cancelled) return;
        if (json.status === 'success' && typeof json.anomaly_rate === 'number') {
          setAnomalyRate(Math.min(100, Math.max(0, json.anomaly_rate)));
        }
      } catch {
        if (!cancelled) setAnomalyRate(0);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchKpis();
    const interval = setInterval(fetchKpis, 10000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const [animatedRate, setAnimatedRate] = React.useState(0);
  React.useEffect(() => {
    const duration = 600;
    const start = displayRate.current;
    const end = anomalyRate;
    const startTime = performance.now();

    const tick = (now: number) => {
      const t = Math.min((now - startTime) / duration, 1);
      const eased = 1 - (1 - t) * (1 - t);
      const v = start + (end - start) * eased;
      displayRate.current = v;
      setAnimatedRate(v);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [anomalyRate]);

  const gaugeColor =
    animatedRate > 50 ? 'error.main' : animatedRate > 20 ? 'warning.main' : 'success.main';
  const riskLabel = animatedRate > 50 ? 'High Risk' : animatedRate > 20 ? 'Moderate' : 'Low Risk';

  if (loading) {
    return (
      <Card variant="outlined" sx={{ height: '100%', minHeight: 160 }}>
        <CardContent>
          <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
            <SpeedRoundedIcon sx={{ color: 'warning.main', fontSize: 20 }} />
            <Typography variant="subtitle2">Risk gauge</Typography>
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
        borderTopColor: animatedRate > 50 ? 'error.main' : animatedRate > 20 ? 'warning.main' : 'success.main',
      }}
    >
      <CardContent>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <SpeedRoundedIcon sx={{ color: gaugeColor, fontSize: 20 }} />
            <Typography variant="subtitle2" fontWeight={600}>
              Risk gauge
            </Typography>
          </Stack>
          <Chip
            size="small"
            label={riskLabel}
            color={animatedRate > 50 ? 'error' : animatedRate > 20 ? 'warning' : 'success'}
            sx={{ fontSize: '0.625rem', height: 18 }}
          />
        </Stack>
        <Box sx={{ position: 'relative', width: '100%', height: 72 }}>
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: 12,
              borderRadius: 2,
              bgcolor: 'action.hover',
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                height: '100%',
                width: `${animatedRate}%`,
                borderRadius: 2,
                bgcolor: gaugeColor,
                transition: 'width 0.1s linear',
              }}
            />
          </Box>
          <Typography
            variant="h5"
            component="p"
            sx={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              fontWeight: 700,
              fontVariantNumeric: 'tabular-nums',
              color: gaugeColor,
            }}
          >
            {animatedRate.toFixed(1)}%
          </Typography>
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
          Anomaly rate · gauge animates on update
        </Typography>
      </CardContent>
    </Card>
  );
}
