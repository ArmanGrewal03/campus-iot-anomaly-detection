import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import { ScatterChart } from '@mui/x-charts/ScatterChart';

interface Props {
  data: Record<string, unknown>[];
  loading: boolean;
}

export default function TrafficSymmetryChart({ data, loading }: Props) {
  const theme = useTheme();

  const { normalPoints, attackPoints } = React.useMemo(() => {
    const normal: { x: number; y: number; id: number }[] = [];
    const attack: { x: number; y: number; id: number }[] = [];
    // Sample up to 500 points per class for performance
    let ni = 0, ai = 0;
    for (const row of data) {
      const sb = Number(row.sbytes ?? 0);
      const db = Number(row.dbytes ?? 0);
      if (isNaN(sb) || isNaN(db)) continue;
      const label = Number(row.label ?? 0);
      if (label === 0 && normal.length < 500) {
        normal.push({ x: sb, y: db, id: ni++ });
      } else if (label === 1 && attack.length < 500) {
        attack.push({ x: sb, y: db, id: ai++ });
      }
      if (normal.length >= 500 && attack.length >= 500) break;
    }
    return { normalPoints: normal, attackPoints: attack };
  }, [data]);

  if (loading) {
    return (
      <Card variant="outlined" sx={{ width: '100%', height: '100%' }}>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>Traffic Symmetry</Typography>
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
        <Typography variant="subtitle2" gutterBottom>Traffic Symmetry</Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
          Source bytes vs Destination bytes — Normal (blue) vs Attack (red)
        </Typography>
        <ScatterChart
          series={[
            {
              label: 'Normal',
              data: normalPoints,
              color: theme.palette.primary.main,
              markerSize: 3,
            },
            {
              label: 'Attack',
              data: attackPoints,
              color: theme.palette.error.main,
              markerSize: 3,
            },
          ]}
          xAxis={[{
            label: 'Source Bytes',
            valueFormatter: (v) => Number(v) >= 1000 ? `${(Number(v) / 1000).toFixed(0)}k` : String(v),
          }]}
          yAxis={[{
            label: 'Dest Bytes',
            valueFormatter: (v) => Number(v) >= 1000 ? `${(Number(v) / 1000).toFixed(0)}k` : String(v),
          }]}
          height={300}
          margin={{ left: 60, right: 16, top: 20, bottom: 40 }}
          grid={{ vertical: true, horizontal: true }}
        />
      </CardContent>
    </Card>
  );
}
