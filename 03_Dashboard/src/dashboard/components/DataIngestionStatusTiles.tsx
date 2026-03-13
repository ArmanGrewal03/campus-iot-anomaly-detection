import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';

const GATEWAY_BASE = 'http://127.0.0.1:8003';

/**
 * Small status tiles that fetch from Data Ingestion (/tables, /stats).
 * Rendered above Copyright so you can see which tiles are working.
 */
export default function DataIngestionStatusTiles() {
  const [loading, setLoading] = React.useState(true);
  const [datasetName, setDatasetName] = React.useState<string | null>(null);
  const [totalRecords, setTotalRecords] = React.useState<number | null>(null);
  const [trainingRecords, setTrainingRecords] = React.useState<number | null>(null);
  const [testingRecords, setTestingRecords] = React.useState<number | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    async function fetchData() {
      try {
        setLoading(true);
        setError(null);

        const tablesRes = await fetch(`${GATEWAY_BASE}/tables`);
        const tablesJson = (await tablesRes.json()) as { status?: string; tables?: string[] };
        if (!tablesRes.ok || tablesJson.status !== 'success' || !tablesJson.tables?.length) {
          if (!cancelled) {
            setDatasetName(null);
            setTotalRecords(0);
            setError('No datasets');
          }
          return;
        }

        const names = tablesJson.tables
          .filter((t) => t.startsWith('csv_data_'))
          .map((t) => t.replace(/^csv_data_/, ''));
        const first = names[0];
        if (!first) {
          if (!cancelled) setError('No datasets');
          return;
        }

        const statsRes = await fetch(`${GATEWAY_BASE}/stats`, {
          headers: { 'dataset-name': first },
        });
        const statsJson = (await statsRes.json()) as {
          total_records?: number;
          training_records?: number;
          testing_records?: number;
          error?: string;
        };

        if (!statsRes.ok || statsJson.error) {
          if (!cancelled) setError(statsJson.error || 'Stats failed');
          return;
        }

        if (!cancelled) {
          setDatasetName(first);
          setTotalRecords(statsJson.total_records ?? 0);
          setTrainingRecords(statsJson.training_records ?? 0);
          setTestingRecords(statsJson.testing_records ?? 0);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Error');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchData();
    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <Card variant="outlined">
        <CardContent>
          <Stack direction="row" alignItems="center" spacing={1}>
            <CircularProgress size={20} />
            <Typography variant="body2" color="text.secondary">
              Loading ingestion status…
            </Typography>
          </Stack>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Data ingestion (working)
        </Typography>
        {error ? (
          <Typography variant="body2" color="text.secondary">
            {error}
          </Typography>
        ) : (
          <Stack spacing={0.5}>
            <Box>
              <Typography variant="caption" color="text.secondary">Dataset</Typography>
              <Typography variant="body2" fontWeight={600}>
                {datasetName ?? '—'}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Total records</Typography>
              <Typography variant="body2" fontWeight={600}>
                {(totalRecords ?? 0).toLocaleString()}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Training / Testing</Typography>
              <Typography variant="body2" fontWeight={600}>
                {(trainingRecords ?? 0).toLocaleString()} / {(testingRecords ?? 0).toLocaleString()}
              </Typography>
            </Box>
          </Stack>
        )}
      </CardContent>
    </Card>
  );
}
