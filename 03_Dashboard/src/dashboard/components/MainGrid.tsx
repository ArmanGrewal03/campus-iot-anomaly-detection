import * as React from 'react';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Backdrop from '@mui/material/Backdrop';
import Copyright from '../internals/components/Copyright';
import ChartUserByCountry from './ChartUserByCountry';
import CustomizedTreeView from './CustomizedTreeView';
import CustomizedDataGrid from './CustomizedDataGrid';
import PageViewsBarChart from './PageViewsBarChart';
import SessionsChart from './SessionsChart';
import StatCard, { StatCardProps } from './StatCard';

const GATEWAY_BASE = 'http://127.0.0.1:8003';
const USER_SERVICE_BASE = `${GATEWAY_BASE}`;

const fallbackCards: StatCardProps[] = [
  {
    title: 'Users',
    value: '—',
    interval: 'Total registered users',
    trend: 'neutral',
    data: [0, 0, 0, 0, 0, 0, 0],
  },
  {
    title: 'Anomalies',
    value: '—',
    interval: 'Predicted anomalies',
    trend: 'neutral',
    data: [0, 0, 0, 0, 0, 0, 0],
  },
  {
    title: 'Events',
    value: '—',
    interval: 'Total websocket events',
    trend: 'neutral',
    data: [0, 0, 0, 0, 0, 0, 0],
  },
];

export default function MainGrid() {
  const [cards, setCards] = React.useState<StatCardProps[]>(fallbackCards);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [initialLoading, setInitialLoading] = React.useState(true);
  const [kpisLoaded, setKpisLoaded] = React.useState(false);
  const [dataGridLoaded, setDataGridLoaded] = React.useState(false);

  // Check if both components have loaded to hide initial loading overlay
  React.useEffect(() => {
    if (kpisLoaded && dataGridLoaded) {
      setInitialLoading(false);
    }
  }, [kpisLoaded, dataGridLoaded]);

  React.useEffect(() => {
    const fetchKpis = async () => {
      try {
        setLoading(true);
        setError(null);

        const res = await fetch(`${USER_SERVICE_BASE}/dashboard-kpis`);
        const json = await res.json() as {
          status?: string;
          total_users?: number;
          total_events?: number;
          total_predictions?: number;
          total_anomalies?: number;
          anomaly_rate?: number;
          detail?: string;
        };

        if (!res.ok || json.status !== 'success') {
          const msg = json.detail || json.status || res.statusText;
          setError(`Failed to load KPIs: ${msg}`);
          setCards(fallbackCards);
          return;
        }

        const totalUsers = json.total_users ?? 0;
        const totalEvents = json.total_events ?? 0;
        const totalAnomalies = json.total_anomalies ?? 0;
        const anomalyRate = json.anomaly_rate ?? 0;

        const newCards: StatCardProps[] = [
          {
            title: 'Users',
            value: totalUsers.toLocaleString(),
            interval: 'Total registered users',
            trend: 'neutral',
            data: [totalUsers], // simple sparkline using total
          },
          {
            title: 'Anomalies',
            value: `${totalAnomalies.toLocaleString()} (${anomalyRate.toFixed(1)}%)`,
            interval: 'Predicted anomalies',
            trend: anomalyRate > 50 ? 'up' : 'neutral',
            data: [anomalyRate],
          },
          {
            title: 'Events',
            value: totalEvents.toLocaleString(),
            interval: 'Total websocket events',
            trend: 'neutral',
            data: [totalEvents],
          },
        ];

        setCards(newCards);
      } catch (err) {
        console.error('Failed to fetch dashboard KPIs:', err);
        setError('Failed to load dashboard KPIs. Check backend.');
        setCards(fallbackCards);
      } finally {
        setLoading(false);
        setKpisLoaded(true);
      }
    };

    fetchKpis();
  }, []);

  // Callback to notify when DataGrid finishes initial load
  const handleDataGridLoaded = React.useCallback(() => {
    setDataGridLoaded(true);
  }, []);

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' }, position: 'relative' }}>
      {/* Full-page loading overlay for initial load */}
      <Backdrop
        open={initialLoading}
        sx={{
          color: '#fff',
          zIndex: (theme) => theme.zIndex.drawer + 1,
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
        }}
      >
        <Stack spacing={2} alignItems="center">
          <CircularProgress size={48} />
          <Typography variant="h6">Loading dashboard...</Typography>
        </Stack>
      </Backdrop>

      {/* cards */}
      <Typography component="h2" variant="h6" sx={{ mb: 2 }}>
        Overview
      </Typography>
      {!initialLoading && loading && (
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
          <CircularProgress size={16} />
          <Typography variant="caption" color="text.secondary">
            Loading KPIs…
          </Typography>
        </Stack>
      )}
      {error && (
        <Alert severity="warning" sx={{ mb: 1 }}>
          {error}
        </Alert>
      )}
      <Grid
        container
        spacing={2}
        columns={12}
        sx={{ mb: (theme) => theme.spacing(2) }}
      >
        {cards.map((card, index) => (
          <Grid key={index} size={{ xs: 12, sm: 6, lg: 4 }}>
            <StatCard {...card} />
          </Grid>
        ))}
        <Grid size={{ xs: 12, md: 6 }}>
          <SessionsChart />
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <PageViewsBarChart />
        </Grid>
      </Grid>
      <Typography component="h2" variant="h6" sx={{ mb: 2 }}>
        Details
      </Typography>
      <Grid container spacing={2} columns={12}>
        <Grid size={{ xs: 12, lg: 9 }}>
          <CustomizedDataGrid onInitialLoadComplete={handleDataGridLoaded} hideLoadingDuringInitialLoad={initialLoading} />
        </Grid>
        <Grid size={{ xs: 12, lg: 3 }}>
          <Stack gap={2} direction={{ xs: 'column', sm: 'row', lg: 'column' }}>
            <CustomizedTreeView />
            <ChartUserByCountry />
          </Stack>
        </Grid>
      </Grid>
      <Copyright sx={{ my: 4 }} />
    </Box>
  );
}
