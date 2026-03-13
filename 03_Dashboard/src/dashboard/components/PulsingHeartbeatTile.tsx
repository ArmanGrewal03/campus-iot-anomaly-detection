import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import FavoriteRoundedIcon from '@mui/icons-material/FavoriteRounded';

const GATEWAY_BASE = 'http://127.0.0.1:8003';

export default function PulsingHeartbeatTile() {
  const [anomalyRate, setAnomalyRate] = React.useState(0);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    const fetchKpis = async () => {
      try {
        const res = await fetch(`${GATEWAY_BASE}/dashboard-kpis?t=${Date.now()}`);
        const json = await res.json() as { status?: string; anomaly_rate?: number };
        if (cancelled) return;
        if (json.status === 'success' && typeof json.anomaly_rate === 'number') {
          setAnomalyRate(json.anomaly_rate);
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

  const pulseColor =
    anomalyRate > 50 ? 'error.main' : anomalyRate > 20 ? 'warning.main' : 'success.main';

  return (
    <Card
      variant="outlined"
      sx={{
        height: '100%',
        minHeight: 160,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderTop: '3px solid',
        borderTopColor: anomalyRate > 50 ? 'error.main' : anomalyRate > 20 ? 'warning.main' : 'success.main',
      }}
    >
      <CardContent sx={{ width: '100%', textAlign: 'center' }}>
        <Stack direction="row" alignItems="center" justifyContent="center" spacing={1} sx={{ mb: 1 }}>
          <Typography variant="subtitle2" fontWeight={600}>
            System pulse
          </Typography>
          <Chip size="small" label="Live" color="error" sx={{ fontSize: '0.625rem', height: 18 }} />
        </Stack>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            width: 80,
            height: 80,
            mx: 'auto',
            '@keyframes pulseRing': {
              '0%, 100%': { transform: 'scale(1)', opacity: 0.25 },
              '50%': { transform: 'scale(1.15)', opacity: 0.1 },
            },
            '@keyframes pulseHeart': {
              '0%, 100%': { transform: 'scale(1)' },
              '14%': { transform: 'scale(1.15)' },
              '28%': { transform: 'scale(1)' },
              '42%': { transform: 'scale(1.12)' },
              '70%': { transform: 'scale(1)' },
            },
          }}
        >
          <Box
            sx={{
              position: 'absolute',
              width: 64,
              height: 64,
              borderRadius: '50%',
              bgcolor: pulseColor,
              opacity: 0.2,
              animation: 'pulseRing 1.5s ease-in-out infinite',
            }}
          />
          <Box
            sx={{
              position: 'absolute',
              width: 56,
              height: 56,
              borderRadius: '50%',
              bgcolor: pulseColor,
              opacity: 0.35,
              animation: 'pulseRing 1.5s ease-in-out infinite 0.15s',
            }}
          />
          <FavoriteRoundedIcon
            sx={{
              fontSize: 36,
              color: pulseColor,
              animation: 'pulseHeart 1.2s ease-in-out infinite',
            }}
          />
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          Anomaly rate {loading ? '…' : `${anomalyRate.toFixed(1)}%`} · Low / Moderate / High
        </Typography>
      </CardContent>
    </Card>
  );
}
