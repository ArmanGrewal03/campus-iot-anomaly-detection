import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import { BarChart } from '@mui/x-charts/BarChart';

/** Distinct palette for attack categories — varied, readable, not all blue */
const CATEGORY_PALETTE = [
  '#c62828', // red
  '#ef6c00', // deep orange
  '#f9a825', // amber
  '#558b2f', // light green
  '#00695c', // teal
  '#1565c0', // blue
  '#6a1b9a', // purple
  '#ad1457', // pink
  '#37474f', // blue grey
  '#455a64', // blue grey light
];

interface Props {
  data: Record<string, unknown>[];
  loading: boolean;
}

export default function AttackCategoryChart({ data, loading }: Props) {
  const theme = useTheme();

  const { categories, counts } = React.useMemo(() => {
    const map: Record<string, number> = {};
    for (const row of data) {
      const cat = String(row.attack_cat ?? row.label ?? 'Unknown').trim() || 'Unknown';
      map[cat] = (map[cat] ?? 0) + 1;
    }
    const sorted = Object.entries(map).sort((a, b) => b[1] - a[1]).slice(0, 10);
    return { categories: sorted.map(([k]) => k), counts: sorted.map(([, v]) => v) };
  }, [data]);

  const barColors = React.useMemo(
    () => categories.map((_, i) => CATEGORY_PALETTE[i % CATEGORY_PALETTE.length]),
    [categories]
  );

  if (loading) {
    return (
      <Card
        variant="outlined"
        sx={{
          width: '100%',
          height: '100%',
          borderRadius: 2,
          boxShadow: (t) => t.shadows[1],
          borderColor: 'divider',
        }}
      >
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
            Attack Categories
          </Typography>
          <Stack alignItems="center" justifyContent="center" sx={{ minHeight: 200 }}>
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
        width: '100%',
        height: '100%',
        borderRadius: 2,
        boxShadow: (t) => t.shadows[1],
        borderColor: 'divider',
      }}
    >
      <CardContent sx={{ '&:last-child': { pb: 2 } }}>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          Attack Categories
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1.5, display: 'block' }}>
          Distribution of attack types across {data.length.toLocaleString()} sampled records
        </Typography>
        <BarChart
          layout="horizontal"
          yAxis={[
            {
              scaleType: 'band',
              data: categories,
              tickLabelStyle: { fontSize: 12, fontWeight: 500 },
              colorMap: {
                type: 'ordinal',
                values: categories,
                colors: barColors,
              },
            },
          ]}
          xAxis={[{
            valueFormatter: (v) => Number(v) >= 1000 ? `${(Number(v) / 1000).toFixed(1)}k` : String(v),
          }]}
          series={[{ id: 'count', data: counts, label: 'Records' }]}
          height={220}
          margin={{ left: 120, right: 24, top: 8, bottom: 24 }}
          grid={{ vertical: true }}
          borderRadius={6}
          slotProps={{ legend: { hidden: true } }}
          sx={{
            '& .MuiChartsAxis-tick': { fill: (theme.vars || theme).palette.text.secondary },
            '& .MuiBarElement-root': { rx: 4 },
          }}
        />
      </CardContent>
    </Card>
  );
}
