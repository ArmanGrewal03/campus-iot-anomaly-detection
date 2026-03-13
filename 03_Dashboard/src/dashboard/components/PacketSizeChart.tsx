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

export default function PacketSizeChart({ data, loading }: Props) {
  const theme = useTheme();

  const { metrics, normalVals, attackVals } = React.useMemo(() => {
    const fields = ['smean', 'dmean', 'sbytes', 'dbytes'] as const;
    const labels = ['Src Pkt Size', 'Dst Pkt Size', 'Src Bytes', 'Dst Bytes'];
    const normalSums: number[] = [0, 0, 0, 0];
    const attackSums: number[] = [0, 0, 0, 0];
    let normalCount = 0;
    let attackCount = 0;

    for (const row of data) {
      const label = Number(row.label ?? 0);
      const sums = label === 0 ? normalSums : attackSums;
      if (label === 0) normalCount++;
      else attackCount++;
      for (let i = 0; i < fields.length; i++) {
        sums[i] += Number(row[fields[i]] ?? 0) || 0;
      }
    }

    return {
      metrics: labels,
      normalVals: normalSums.map((s) => normalCount > 0 ? Math.round(s / normalCount) : 0),
      attackVals: attackSums.map((s) => attackCount > 0 ? Math.round(s / attackCount) : 0),
    };
  }, [data]);

  if (loading) {
    return (
      <Card variant="outlined" sx={{ width: '100%', height: '100%' }}>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>Packet Size Comparison</Typography>
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
        <Typography variant="subtitle2" gutterBottom>Packet Size Comparison</Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
          Average values — Normal vs Attack traffic
        </Typography>
        <BarChart
          borderRadius={6}
          colors={[theme.palette.primary.main, theme.palette.error.main]}
          xAxis={[{
            scaleType: 'band',
            data: metrics,
            tickLabelStyle: { fontSize: 11 },
          }]}
          yAxis={[{
            valueFormatter: (v) => Number(v) >= 1000 ? `${(Number(v) / 1000).toFixed(1)}k` : String(v),
          }]}
          series={[
            { id: 'normal', label: 'Normal (avg)', data: normalVals },
            { id: 'attack', label: 'Attack (avg)', data: attackVals },
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
