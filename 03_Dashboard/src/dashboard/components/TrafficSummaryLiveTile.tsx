import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import StorageRoundedIcon from '@mui/icons-material/StorageRounded';

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

function getTopTag(data: Record<string, unknown>, prefix: string): string | null {
  const matches: string[] = [];
  for (const [k, v] of Object.entries(data)) {
    if (k.startsWith(prefix) && (v === 1 || v === '1' || v === true)) {
      const label = k.replace(prefix, '').replace(/_/g, ' ').trim() || k;
      matches.push(label);
    }
  }
  return matches[0] ?? null;
}

export default function TrafficSummaryLiveTile() {
  const [summary, setSummary] = React.useState<{
    sbytes: number;
    dbytes: number;
    topService: string | null;
    topProto: string | null;
    count: number;
  }>({ sbytes: 0, dbytes: 0, topService: null, topProto: null, count: 0 });
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${GATEWAY_BASE}/history?limit=50&offset=0`);
        const json = await res.json() as { status?: string; history?: HistoryRecord[] };
        if (cancelled) return;
        const history = Array.isArray(json.history) ? json.history : [];
        let totalSbytes = 0;
        let totalDbytes = 0;
        const serviceCounts: Record<string, number> = {};
        const protoCounts: Record<string, number> = {};
        for (const record of history) {
          const d = record.data;
          if (!d || typeof d !== 'object') continue;
          totalSbytes += num(d.sbytes);
          totalDbytes += num(d.dbytes);
          const svc = getTopTag(d, 'service_');
          if (svc) serviceCounts[svc] = (serviceCounts[svc] ?? 0) + 1;
          const proto = getTopTag(d, 'proto_');
          if (proto) protoCounts[proto] = (protoCounts[proto] ?? 0) + 1;
        }
        const topService =
          Object.keys(serviceCounts).length > 0
            ? Object.entries(serviceCounts).sort((a, b) => b[1] - a[1])[0][0]
            : null;
        const topProto =
          Object.keys(protoCounts).length > 0
            ? Object.entries(protoCounts).sort((a, b) => b[1] - a[1])[0][0]
            : null;
        setSummary({
          sbytes: totalSbytes,
          dbytes: totalDbytes,
          topService,
          topProto,
          count: history.length,
        });
      } catch {
        if (!cancelled) setSummary({ sbytes: 0, dbytes: 0, topService: null, topProto: null, count: 0 });
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
            <StorageRoundedIcon sx={{ color: 'primary.main', fontSize: 20 }} />
            <Typography variant="subtitle2">Traffic summary</Typography>
          </Stack>
          <Stack alignItems="center" justifyContent="center" sx={{ minHeight: 100 }}>
            <CircularProgress size={28} />
          </Stack>
        </CardContent>
      </Card>
    );
  }

  const totalBytes = summary.sbytes + summary.dbytes;
  const sbytesPct = totalBytes > 0 ? (summary.sbytes / totalBytes) * 100 : 50;

  return (
    <Card
      variant="outlined"
      sx={{
        height: '100%',
        minHeight: 160,
        borderTop: '3px solid',
        borderTopColor: 'primary.main',
      }}
    >
      <CardContent>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <StorageRoundedIcon sx={{ color: 'primary.main', fontSize: 20 }} />
            <Typography variant="subtitle2" fontWeight={600}>
              Traffic (Analytics data)
            </Typography>
          </Stack>
          <Chip size="small" label="Live" color="error" sx={{ fontSize: '0.625rem', height: 18 }} />
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          sbytes / dbytes from last {summary.count} flows
        </Typography>
        <Stack spacing={0.75}>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">sbytes</Typography>
            <Typography variant="body2" fontWeight={600}>{summary.sbytes.toLocaleString()}</Typography>
          </Stack>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" color="text.secondary">dbytes</Typography>
            <Typography variant="body2" fontWeight={600}>{summary.dbytes.toLocaleString()}</Typography>
          </Stack>
        </Stack>
        <Box sx={{ mt: 1, height: 6, borderRadius: 1, bgcolor: 'action.hover', overflow: 'hidden', display: 'flex' }}>
          <Box sx={{ width: `${sbytesPct}%`, bgcolor: 'primary.main', borderRadius: 1 }} />
          <Box sx={{ flex: 1, bgcolor: 'secondary.main', borderRadius: 1 }} />
        </Box>
        {(summary.topService || summary.topProto) && (
          <Stack direction="row" spacing={1} sx={{ mt: 1 }} flexWrap="wrap">
            {summary.topProto && (
              <Chip size="small" label={`Proto: ${summary.topProto}`} variant="outlined" sx={{ fontSize: '0.65rem', height: 20 }} />
            )}
            {summary.topService && (
              <Chip size="small" label={`Service: ${summary.topService}`} variant="outlined" sx={{ fontSize: '0.65rem', height: 20 }} />
            )}
          </Stack>
        )}
      </CardContent>
    </Card>
  );
}
