import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';

const GATEWAY_BASE = 'http://127.0.0.1:8003';

function formatTimeAgo(iso: string): string {
  try {
    const d = new Date(iso);
    const s = Math.round((Date.now() - d.getTime()) / 1000);
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return `${Math.floor(s / 86400)}d ago`;
  } catch {
    return '—';
  }
}

function getLocationLabel(location: unknown): string {
  if (!location || typeof location !== 'object') return '—';
  const o = location as Record<string, unknown>;
  const city = typeof o.city === 'string' ? o.city : null;
  const country = typeof o.country === 'string' ? o.country : null;
  if (city && country) return `${city}, ${country}`;
  return city || country || '—';
}

function isAnomaly(predictionResults: unknown): boolean {
  if (!predictionResults || typeof predictionResults !== 'object') return false;
  const preds = (predictionResults as Record<string, unknown>).predictions;
  if (!Array.isArray(preds) || preds.length === 0) return false;
  const first = preds[0];
  return typeof first === 'object' && first !== null && (first as Record<string, unknown>).prediction === 1;
}

interface LogEntry {
  id: number;
  network_id: string;
  timestamp: string;
  user_id: number | null;
  location: unknown;
  prediction_results?: unknown;
}

export default function LiveActivityTicker() {
  const [logs, setLogs] = React.useState<LogEntry[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    const fetchLogs = async () => {
      try {
        const res = await fetch(`${GATEWAY_BASE}/network-logs?limit=6&offset=0`);
        const json = await res.json() as { status?: string; logs?: LogEntry[]; returned_records?: number };
        if (cancelled) return;
        const list = Array.isArray(json.logs) ? json.logs : [];
        setLogs(list);
        setError(null);
      } catch (e) {
        if (!cancelled) {
          setError('Failed to load');
          setLogs([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchLogs();
    const interval = setInterval(fetchLogs, 6000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (loading) {
    return (
      <Card variant="outlined" sx={{ height: '100%', minHeight: 200 }}>
        <CardContent>
          <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
            <FiberManualRecordIcon sx={{ color: 'error.main', fontSize: 12, animation: 'pulse 1.5s ease-in-out infinite' }} />
            <Typography variant="subtitle2">Live Activity</Typography>
          </Stack>
          <Stack alignItems="center" justifyContent="center" sx={{ minHeight: 140 }}>
            <CircularProgress size={24} />
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
        minHeight: 200,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        borderTop: '3px solid',
        borderTopColor: 'success.main',
      }}
    >
      <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, '&:last-child': { pb: 1.5 } }}>
        <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1, flexShrink: 0 }}>
          <FiberManualRecordIcon
            sx={{
              color: 'error.main',
              fontSize: 12,
              '@keyframes pulse': { '0%, 100%': { opacity: 1 }, '50%': { opacity: 0.4 } },
              animation: 'pulse 1.5s ease-in-out infinite',
            }}
          />
          <Typography variant="subtitle2" fontWeight={600}>
            Live Activity
          </Typography>
          <Chip size="small" label="Live" color="error" sx={{ fontSize: '0.625rem', height: 18 }} />
        </Stack>
        {error && (
          <Typography variant="caption" color="text.secondary">
            {error}
          </Typography>
        )}
        <Stack spacing={0.75} sx={{ overflow: 'auto', flex: 1, minHeight: 0 }}>
          {logs.length === 0 && (
            <Typography variant="caption" color="text.secondary">
              No recent activity
            </Typography>
          )}
          {logs.map((log) => (
            <Box
              key={log.id}
              sx={{
                py: 0.75,
                px: 1,
                borderRadius: 1,
                bgcolor: 'action.hover',
                animation: 'slideIn 0.35s ease-out',
                '@keyframes slideIn': {
                  from: { opacity: 0, transform: 'translateY(-8px)' },
                  to: { opacity: 1, transform: 'translateY(0)' },
                },
              }}
            >
              <Stack direction="row" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={0.5}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  User {log.user_id ?? '—'} · {getLocationLabel(log.location)}
                </Typography>
                {isAnomaly(log.prediction_results) && (
                  <Chip size="small" label="Anomaly" color="warning" sx={{ fontSize: '0.6rem', height: 16 }} />
                )}
              </Stack>
              <Typography variant="caption" color="text.secondary">
                {formatTimeAgo(log.timestamp)}
              </Typography>
            </Box>
          ))}
        </Stack>
      </CardContent>
    </Card>
  );
}
