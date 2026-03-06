import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import { BarChart } from '@mui/x-charts/BarChart';
import { useTheme } from '@mui/material/styles';

const GATEWAY_BASE = 'http://127.0.0.1:8003';

export default function PageViewsBarChart() {
  const theme = useTheme();
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [datasetName, setDatasetName] = React.useState<string>('');
  const [categories, setCategories] = React.useState<string[]>([]);
  const [trainingData, setTrainingData] = React.useState<number[]>([]);
  const [testingData, setTestingData] = React.useState<number[]>([]);
  const [totalRecords, setTotalRecords] = React.useState(0);

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

        const typeRes = await fetch(`${GATEWAY_BASE}/type-stats`, {
          headers: { dataset_name: firstDataset },
        });
        const typeJson = (await typeRes.json()) as {
          type_distribution?: Record<string, number>;
          type_training?: Record<string, number>;
          type_testing?: Record<string, number>;
          total_rows?: number;
          error?: string;
        };

        if (!typeRes.ok || typeJson.error) {
          if (!cancelled) setError(typeJson.error || 'Failed to load type stats.');
          return;
        }

        const dist = typeJson.type_distribution ?? {};
        const training = typeJson.type_training ?? {};
        const testing = typeJson.type_testing ?? {};
        const total = typeJson.total_rows ?? Object.values(dist).reduce((a, b) => a + b, 0);

        // Use top 7 categories by count so the bar chart stays readable
        const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]).slice(0, 7);
        const cats = entries.map(([k]) => k);
        const trainArr = cats.map((c) => training[c] ?? 0);
        const testArr = cats.map((c) => testing[c] ?? 0);

        if (!cancelled) {
          setDatasetName(firstDataset);
          setCategories(cats);
          setTrainingData(trainArr);
          setTestingData(testArr);
          setTotalRecords(total);
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

  const colorPalette = [
    (theme.vars || theme).palette.primary.dark,
    (theme.vars || theme).palette.primary.main,
    (theme.vars || theme).palette.primary.light,
  ];

  const axisLabels = categories.map((c) =>
    c === '0' ? 'Normal' : c === '1' ? 'Attack' : c
  );

  const maxStacked = trainingData.length
    ? Math.max(...trainingData.map((t, i) => t + (testingData[i] ?? 0)), 1)
    : 1;
  const yTickStep =
    maxStacked <= 5000 ? 1000
    : maxStacked <= 25000 ? 5000
    : maxStacked <= 100000 ? 10000
    : 25000;

  const chartHeight = 280;
  const cardMinHeight = 380;

  if (loading) {
    return (
      <Card variant="outlined" sx={{ width: '100%', minHeight: cardMinHeight }}>
        <CardContent>
          <Typography component="h2" variant="subtitle2" gutterBottom>
            Training vs testing by label
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
            Training vs testing by label
          </Typography>
          <Alert severity="info">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  if (categories.length === 0) {
    return (
      <Card variant="outlined" sx={{ width: '100%', minHeight: cardMinHeight }}>
        <CardContent>
          <Typography component="h2" variant="subtitle2" gutterBottom>
            Training vs testing by label
          </Typography>
          <Alert severity="info">No type/label distribution yet. Validate dataset on Model page.</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined" sx={{ width: '100%', minHeight: cardMinHeight }}>
      <CardContent>
        <Typography component="h2" variant="subtitle2" gutterBottom>
          Training vs testing by label
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1 }}>
          Stacked record count per label/category — dataset: {datasetName} ({totalRecords.toLocaleString()} total)
        </Typography>
        <BarChart
          borderRadius={8}
          colors={colorPalette}
          xAxis={[
            {
              scaleType: 'band',
              data: axisLabels,
              tickLabelStyle: { fontSize: 11 },
            },
          ]}
          yAxis={[
            {
              tickMinStep: yTickStep,
              valueFormatter: (v) =>
                Number(v) >= 1000 ? `${Number(v) / 1000}k` : String(v),
            },
          ]}
          series={[
            { id: 'training', label: 'Training', data: trainingData, stack: 'A' },
            { id: 'testing', label: 'Testing', data: testingData, stack: 'A' },
          ]}
          height={chartHeight}
          margin={{ left: 48, right: 0, top: 20, bottom: 24 }}
          grid={{ horizontal: true }}
          slotProps={{ legend: { hidden: true } }}
        />
      </CardContent>
    </Card>
  );
}
