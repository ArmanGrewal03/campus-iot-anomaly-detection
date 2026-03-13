import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import { PieChart } from '@mui/x-charts/PieChart';

interface Props {
  data: Record<string, unknown>[];
  loading: boolean;
}

export default function ProtocolDistChart({ data, loading }: Props) {
  const theme = useTheme();

  const pieData = React.useMemo(() => {
    const map: Record<string, number> = {};
    for (const row of data) {
      const proto = String(row.proto ?? 'unknown').toLowerCase();
      map[proto] = (map[proto] ?? 0) + 1;
    }
    const palette = [
      theme.palette.primary.main,
      theme.palette.warning.main,
      theme.palette.success.main,
      theme.palette.error.main,
      theme.palette.info.main,
      theme.palette.grey[500],
    ];
    return Object.entries(map)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([label, value], i) => ({
        id: i,
        value,
        label: label.toUpperCase(),
        color: palette[i % palette.length],
      }));
  }, [data, theme]);

  if (loading) {
    return (
      <Card variant="outlined" sx={{ width: '100%', height: '100%' }}>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>Protocol Distribution</Typography>
          <Stack alignItems="center" justifyContent="center" sx={{ minHeight: 200 }}>
            <CircularProgress size={28} />
          </Stack>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined" sx={{ width: '100%', height: '100%' }}>
      <CardContent>
        <Typography variant="subtitle2" gutterBottom>Protocol Distribution</Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
          Network protocol breakdown (TCP, UDP, ICMP, etc.)
        </Typography>
        <PieChart
          series={[
            {
              data: pieData,
              innerRadius: '40%',
              outerRadius: '85%',
              paddingAngle: 2,
              cornerRadius: 4,
              highlightScope: { fade: 'global', highlight: 'item' },
            },
          ]}
          height={200}
          margin={{ top: 8, bottom: 8, left: 8, right: 100 }}
        />
      </CardContent>
    </Card>
  );
}
