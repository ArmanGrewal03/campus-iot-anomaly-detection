import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import { BarChart } from '@mui/x-charts/BarChart';

interface Props {
  data: Record<string, unknown>[];
  loading: boolean;
}

export default function TopServicesChart({ data, loading }: Props) {
  const theme = useTheme();

  const { services, normalCounts, attackCounts } = React.useMemo(() => {
    const normalMap: Record<string, number> = {};
    const attackMap: Record<string, number> = {};
    for (const row of data) {
      const svc = String(row.service ?? '-').trim() || '-';
      const label = Number(row.label ?? 0);
      if (label === 0) {
        normalMap[svc] = (normalMap[svc] ?? 0) + 1;
      } else {
        attackMap[svc] = (attackMap[svc] ?? 0) + 1;
      }
    }
    const allSvcs = new Set([...Object.keys(normalMap), ...Object.keys(attackMap)]);
    const totals = [...allSvcs].map((s) => ({
      svc: s,
      total: (normalMap[s] ?? 0) + (attackMap[s] ?? 0),
    }));
    const top8 = totals.sort((a, b) => b.total - a.total).slice(0, 8).map((t) => t.svc);
    return {
      services: top8,
      normalCounts: top8.map((s) => normalMap[s] ?? 0),
      attackCounts: top8.map((s) => attackMap[s] ?? 0),
    };
  }, [data]);

  if (loading) {
    return (
      <Card variant="outlined" sx={{ width: '100%', height: '100%' }}>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>Top Services Targeted</Typography>
          <Stack alignItems="center" justifyContent="center" sx={{ minHeight: 280 }}>
            <CircularProgress size={28} />
          </Stack>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined" sx={{ width: '100%', height: '100%' }}>
      <CardContent>
        <Typography variant="subtitle2" gutterBottom>Top Services Targeted</Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
          Normal vs Attack traffic per application service
        </Typography>
        <BarChart
          borderRadius={6}
          colors={[theme.palette.primary.main, theme.palette.error.light]}
          xAxis={[{
            scaleType: 'band',
            data: services,
            tickLabelStyle: { fontSize: 11 },
          }]}
          yAxis={[{
            valueFormatter: (v) => Number(v) >= 1000 ? `${(Number(v) / 1000).toFixed(1)}k` : String(v),
          }]}
          series={[
            { id: 'normal', label: 'Normal', data: normalCounts, stack: 'A' },
            { id: 'attack', label: 'Attack', data: attackCounts, stack: 'A' },
          ]}
          height={300}
          margin={{ left: 48, right: 16, top: 20, bottom: 30 }}
          grid={{ horizontal: true }}
          slotProps={{ legend: { hidden: true } }}
        />
      </CardContent>
    </Card>
  );
}
