import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import { LineChart } from '@mui/x-charts/LineChart';

interface Props {
  data: Record<string, unknown>[];
  loading: boolean;
}

function AreaGradient({ color, id }: { color: string; id: string }) {
  return (
    <defs>
      <linearGradient id={id} x1="50%" y1="0%" x2="50%" y2="100%">
        <stop offset="0%" stopColor={color} stopOpacity={0.5} />
        <stop offset="100%" stopColor={color} stopOpacity={0} />
      </linearGradient>
    </defs>
  );
}

export default function DurationDistChart({ data, loading }: Props) {
  const theme = useTheme();

  const { binLabels, normalCounts, attackCounts } = React.useMemo(() => {
    // Bin connection durations into ranges
    const bins = [
      { label: '0s', max: 0.0001 },
      { label: '<0.01s', max: 0.01 },
      { label: '<0.1s', max: 0.1 },
      { label: '<1s', max: 1 },
      { label: '<10s', max: 10 },
      { label: '<60s', max: 60 },
      { label: '<5m', max: 300 },
      { label: '5m+', max: Infinity },
    ];
    const normal = new Array(bins.length).fill(0);
    const attack = new Array(bins.length).fill(0);

    for (const row of data) {
      const dur = Number(row.dur ?? 0);
      if (isNaN(dur)) continue;
      const label = Number(row.label ?? 0);
      for (let i = 0; i < bins.length; i++) {
        if (dur <= bins[i].max || i === bins.length - 1) {
          if (label === 0) normal[i]++;
          else attack[i]++;
          break;
        }
      }
    }
    return {
      binLabels: bins.map((b) => b.label),
      normalCounts: normal,
      attackCounts: attack,
    };
  }, [data]);

  if (loading) {
    return (
      <Card variant="outlined" sx={{ width: '100%', height: '100%' }}>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>Connection Duration</Typography>
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
        <Typography variant="subtitle2" gutterBottom>Connection Duration</Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
          Duration distribution — Normal vs Attack connections
        </Typography>
        <LineChart
          colors={[theme.palette.primary.main, theme.palette.error.main]}
          xAxis={[{
            scaleType: 'point',
            data: binLabels,
            tickLabelStyle: { fontSize: 11 },
          }]}
          yAxis={[{
            valueFormatter: (v) => Number(v) >= 1000 ? `${(Number(v) / 1000).toFixed(1)}k` : String(v),
          }]}
          series={[
            {
              id: 'normal',
              label: 'Normal',
              data: normalCounts,
              showMark: true,
              curve: 'natural',
              area: true,
            },
            {
              id: 'attack',
              label: 'Attack',
              data: attackCounts,
              showMark: true,
              curve: 'natural',
              area: true,
            },
          ]}
          height={300}
          margin={{ left: 48, right: 16, top: 20, bottom: 30 }}
          grid={{ horizontal: true }}
          sx={{
            '& .MuiAreaElement-series-normal': { fill: "url('#dur-normal')" },
            '& .MuiAreaElement-series-attack': { fill: "url('#dur-attack')" },
          }}
          slotProps={{ legend: { hidden: true } }}
        >
          <AreaGradient color={theme.palette.primary.main} id="dur-normal" />
          <AreaGradient color={theme.palette.error.main} id="dur-attack" />
        </LineChart>
      </CardContent>
    </Card>
  );
}
