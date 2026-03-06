import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import TrendingUpRoundedIcon from '@mui/icons-material/TrendingUpRounded';

const GATEWAY_BASE = 'http://127.0.0.1:8003';

function useCountUp(target: number, durationMs = 600): number {
  const [display, setDisplay] = React.useState(target);
  const prevTarget = React.useRef(target);
  const rafRef = React.useRef<number>();

  React.useEffect(() => {
    if (target === prevTarget.current) return;
    const start = prevTarget.current;
    const startTime = performance.now();
    prevTarget.current = target;

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const t = Math.min(elapsed / durationMs, 1);
      const eased = 1 - (1 - t) * (1 - t);
      setDisplay(Math.round(start + (target - start) * eased));
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [target, durationMs]);

  return display;
}

export default function AnimatedCounterTile() {
  const [total, setTotal] = React.useState(0);
  const [loading, setLoading] = React.useState(true);
  const displayValue = useCountUp(total);

  React.useEffect(() => {
    let cancelled = false;
    const fetchKpis = async () => {
      try {
        const res = await fetch(`${GATEWAY_BASE}/dashboard-kpis?t=${Date.now()}`);
        const json = await res.json() as { status?: string; total_events?: number };
        if (cancelled) return;
        if (json.status === 'success' && typeof json.total_events === 'number') {
          setTotal(json.total_events);
        }
      } catch {
        if (!cancelled) setTotal(0);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchKpis();
    const interval = setInterval(fetchKpis, 8000);
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
            <TrendingUpRoundedIcon sx={{ color: 'success.main', fontSize: 20 }} />
            <Typography variant="subtitle2">Events (live)</Typography>
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
        borderTopColor: 'success.main',
      }}
    >
      <CardContent>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <TrendingUpRoundedIcon sx={{ color: 'success.main', fontSize: 20 }} />
            <Typography variant="subtitle2" fontWeight={600}>
              Events (live)
            </Typography>
          </Stack>
          <Chip size="small" label="Live" color="error" sx={{ fontSize: '0.625rem', height: 18 }} />
        </Stack>
        <Typography
          variant="h4"
          component="p"
          sx={{
            fontWeight: 700,
            fontVariantNumeric: 'tabular-nums',
            transition: 'transform 0.2s ease',
            '&:hover': { transform: 'scale(1.02)' },
          }}
        >
          {displayValue.toLocaleString()}
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
          Total websocket events · updates every 8s
        </Typography>
        <Box
          sx={{
            mt: 1,
            height: 4,
            borderRadius: 2,
            bgcolor: 'action.hover',
            overflow: 'hidden',
          }}
        >
          <Box
            sx={{
              height: '100%',
              width: '40%',
              bgcolor: 'success.main',
              borderRadius: 2,
              animation: 'shimmer 2s ease-in-out infinite',
              '@keyframes shimmer': { '0%, 100%': { opacity: 0.8 }, '50%': { opacity: 1 } },
            }}
          />
        </Box>
      </CardContent>
    </Card>
  );
}
