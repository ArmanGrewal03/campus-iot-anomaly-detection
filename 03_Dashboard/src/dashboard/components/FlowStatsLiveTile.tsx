import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import ShowChartRoundedIcon from '@mui/icons-material/ShowChartRounded';

const GATEWAY_BASE = 'http://127.0.0.1:8003';

interface HistoryRecord {
  id: number;
  data: Record<string, unknown> | null;
}

function num(v: unknown): number {
  if (v == null) return 0;
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

export default function FlowStatsLiveTile() {
  const [stats, setStats] = React.useState<{ avgDur: number; totalPkts: number; avgRate: number; count: number }>({
    avgDur: 0,
    totalPkts: 0,
    avgRate: 0,
    count: 0,
  });
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${GATEWAY_BASE}/history?limit=50&offset=0`);
        const json = await res.json() as { status?: string; history?: HistoryRecord[] };
        if (cancelled) return;
        const history = Array.isArray(json.history) ? json.history : [];
        let sumDur = 0;
        let sumPkts = 0;
        let sumRate = 0;
        let n = 0;
        for (const record of history) {
          const d = record.data;
          if (!d || typeof d !== 'object') continue;
          sumDur += num(d.dur);
          sumPkts += num(d.Spkts) + num(d.Dpkts);
          sumRate += num(d.rate);
          n += 1;
        }
        setStats({
          avgDur: n > 0 ? sumDur / n : 0,
          totalPkts: sumPkts,
          avgRate: n > 0 ? sumRate / n : 0,
          count: n,
        });
      } catch {
        if (!cancelled) setStats({ avgDur: 0, totalPkts: 0, avgRate: 0, count: 0 });
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchHistory();
    const interval = setInterval(fetchHistory, 12000);
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
            <ShowChartRoundedIcon sx={{ color: 'info.main', fontSize: 20 }} />
            <Typography variant="subtitle2">Flow stats</Typography>
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
        borderTopColor: 'info.main',
      }}
    >
      <CardContent>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <ShowChartRoundedIcon sx={{ color: 'info.main', fontSize: 20 }} />
            <Typography variant="subtitle2" fontWeight={600}>
              Flow stats (Analytics data)
            </Typography>
          </Stack>
          <Chip size="small" label="Live" color="error" sx={{ fontSize: '0.625rem', height: 18 }} />
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          From last {stats.count} records · dur, Spkts/Dpkts, rate
        </Typography>
        <Stack spacing={0.5}>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">Avg duration</Typography>
            <Typography variant="body2" fontWeight={600}>{stats.avgDur.toFixed(1)}</Typography>
          </Stack>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">Total packets</Typography>
            <Typography variant="body2" fontWeight={600}>{stats.totalPkts.toLocaleString()}</Typography>
          </Stack>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">Avg rate</Typography>
            <Typography variant="body2" fontWeight={600}>{stats.avgRate.toFixed(1)}</Typography>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
