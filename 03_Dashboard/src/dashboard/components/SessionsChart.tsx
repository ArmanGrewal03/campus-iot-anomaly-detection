import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import { LineChart } from '@mui/x-charts/LineChart';

const GATEWAY_BASE = 'http://127.0.0.1:8003';

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

// Duration bins (seconds): [max value, label]
const DURATION_BINS: { max: number; label: string }[] = [
  { max: 0.001, label: '<1ms' },
  { max: 0.01, label: '1–10ms' },
  { max: 0.1, label: '10–100ms' },
  { max: 1, label: '0.1–1s' },
  { max: 10, label: '1–10s' },
  { max: 60, label: '10–60s' },
  { max: 300, label: '1–5m' },
  { max: Infinity, label: '5m+' },
];

export default function SessionsChart() {
  const theme = useTheme();
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [datasetName, setDatasetName] = React.useState<string>('');
  const [binLabels, setBinLabels] = React.useState<string[]>([]);
  const [normalData, setNormalData] = React.useState<number[]>([]);
  const [attackData, setAttackData] = React.useState<number[]>([]);

  React.useEffect(() => {
    let cancelled = false;

    async function fetchData() {
      try {
        setLoading(true);
        setError(null);

        const tablesRes = await fetch(`${GATEWAY_BASE}/tables`);
        const tablesJson = (await tablesRes.json()) as { status?: string; tables?: string[] };
        if (!tablesRes.ok || tablesJson.status !== 'success' || !tablesJson.tables?.length) {
          if (!cancelled) setError('No datasets. Upload data on the Model page.');
          return;
        }

        const names = tablesJson.tables
          .filter((t) => t.startsWith('csv_data_'))
          .map((t) => t.replace(/^csv_data_/, ''));
        const firstDataset = names[0];
        if (!firstDataset) {
          if (!cancelled) setError('No datasets.');
          return;
        }

        const viewRes = await fetch(`${GATEWAY_BASE}/view?limit=5000&offset=0`, {
          headers: { 'dataset-name': firstDataset },
        });
        const viewJson = (await viewRes.json()) as {
          status?: string;
          data?: { id: number; data: Record<string, unknown> }[];
        };

        if (!viewRes.ok || viewJson.status !== 'success' || !viewJson.data?.length) {
          if (!cancelled) setError('No records to display. Upload data on the Model page.');
          return;
        }

        const rows = viewJson.data.map((r) => r.data);
        const normal = new Array(DURATION_BINS.length).fill(0);
        const attack = new Array(DURATION_BINS.length).fill(0);

        for (const row of rows) {
          const dur = Number(row.dur ?? 0);
          if (typeof dur !== 'number' || isNaN(dur) || dur < 0) continue;
          const label = Number(row.label ?? 0);
          for (let i = 0; i < DURATION_BINS.length; i++) {
            if (dur <= DURATION_BINS[i].max) {
              if (label === 0) normal[i]++;
              else attack[i]++;
              break;
            }
          }
        }

        if (!cancelled) {
          setDatasetName(firstDataset);
          setBinLabels(DURATION_BINS.map((b) => b.label));
          setNormalData(normal);
          setAttackData(attack);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load data.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchData();
    return () => { cancelled = true; };
  }, []);

  const maxVal = Math.max(
    ...normalData,
    ...attackData,
    1
  );
  const yTickStep =
    maxVal <= 10 ? 5
    : maxVal <= 50 ? 10
    : maxVal <= 200 ? 50
    : maxVal <= 1000 ? 100
    : 500;

  const colorPalette = [
    theme.palette.primary.main,
    theme.palette.error.main,
  ];
  const chartHeight = 280;
  const cardMinHeight = 380;

  if (loading) {
    return (
      <Card variant="outlined" sx={{ width: '100%', minHeight: cardMinHeight }}>
        <CardContent>
          <Typography component="h2" variant="subtitle2" gutterBottom>
            Traffic by connection duration
          </Typography>
          <Stack alignItems="center" justifyContent="center" sx={{ minHeight: chartHeight }}>
            <CircularProgress size={32} />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              Loading from data ingestion…
            </Typography>
          </Stack>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="outlined" sx={{ width: '100%', minHeight: cardMinHeight }}>
        <CardContent>
          <Typography component="h2" variant="subtitle2" gutterBottom>
            Traffic by connection duration
          </Typography>
          <Alert severity="info">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined" sx={{ width: '100%', minHeight: cardMinHeight }}>
      <CardContent>
        <Typography component="h2" variant="subtitle2" gutterBottom>
          Traffic by connection duration
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1 }}>
          Normal vs Attack flows by duration bucket — dataset: {datasetName}
        </Typography>
        <LineChart
          colors={colorPalette}
          xAxis={[
            {
              scaleType: 'point',
              data: binLabels,
              tickLabelStyle: { fontSize: 11 },
            },
          ]}
          yAxis={[
            {
              tickMinStep: yTickStep,
              valueFormatter: (v) => (Number(v) >= 1000 ? `${Number(v) / 1000}k` : String(v)),
            },
          ]}
          series={[
            {
              id: 'normal',
              label: 'Normal',
              showMark: true,
              curve: 'natural',
              area: true,
              data: normalData,
            },
            {
              id: 'attack',
              label: 'Attack',
              showMark: true,
              curve: 'natural',
              area: true,
              data: attackData,
            },
          ]}
          height={chartHeight}
          margin={{ left: 48, right: 20, top: 20, bottom: 24 }}
          grid={{ horizontal: true }}
          sx={{
            '& .MuiAreaElement-series-normal': { fill: "url('#sessions-normal')" },
            '& .MuiAreaElement-series-attack': { fill: "url('#sessions-attack')" },
          }}
          slotProps={{ legend: { hidden: true } }}
        >
          <AreaGradient color={theme.palette.primary.main} id="sessions-normal" />
          <AreaGradient color={theme.palette.error.main} id="sessions-attack" />
        </LineChart>
      </CardContent>
    </Card>
  );
}
