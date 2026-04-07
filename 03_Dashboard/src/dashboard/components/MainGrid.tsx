import * as React from 'react';
import { useNavigate } from 'react-router-dom';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Backdrop from '@mui/material/Backdrop';
import Copyright from '../internals/components/Copyright';
import InteractiveGlobe from './InteractiveGlobe';
import CustomizedDataGrid from './CustomizedDataGrid';
import PageViewsBarChart from './PageViewsBarChart';
import AttackCategoryChart from './AttackCategoryChart';
import ProtocolDistChart from './ProtocolDistChart';
import StatCard, { StatCardProps } from './StatCard';
import FlowStatsLiveTile from './FlowStatsLiveTile';
import TrafficSummaryLiveTile from './TrafficSummaryLiveTile';
import ModelDataOverviewTile from './ModelDataOverviewTile';
import LiveRequestsStatusTile from './LiveRequestsStatusTile';
import LiveBlockingStatusTile from './LiveBlockingStatusTile';
import LiveQueryPerSecondTile from './LiveQueryPerSecondTile';
import LivePacketRateTile from './LivePacketRateTile';

const GATEWAY_BASE = 'http://127.0.0.1:8003';
const USER_SERVICE_BASE = `${GATEWAY_BASE}`;
const KPI_CACHE_KEY = 'dashboard_kpis_cache';

const fallbackCards: StatCardProps[] = [
  { title: 'Users', value: '—', interval: 'Total registered users', trend: 'neutral', data: [0, 0, 0, 0, 0, 0, 0], chartVariant: 'sparkline', gaugeValue: 0 },
  { title: 'Events', value: '—', interval: 'Events per day (last 7 days)', trend: 'neutral', data: [0, 0, 0, 0, 0, 0, 0], chartVariant: 'bar', gaugeValue: 0 },
  { title: 'Anomalies', value: '—', interval: 'Predicted anomalies', trend: 'neutral', data: [0, 0, 0, 0, 0, 0, 0], chartVariant: 'progress', gaugeValue: 0 },
  { title: 'Predictions', value: '—', interval: 'Predictions per day (last 7 days)', trend: 'neutral', data: [0, 0, 0, 0, 0, 0, 0], chartVariant: 'sparkline', gaugeValue: 0 },
];

function getCachedKpis(): Record<string, unknown> | null {
  try {
    const raw = sessionStorage.getItem(KPI_CACHE_KEY);
    if (!raw) return null;
    const cached = JSON.parse(raw) as { data: Record<string, unknown>; ts: number };
    if (Date.now() - cached.ts > 60000) return null; // stale after 60s
    return cached.data;
  } catch { return null; }
}
function setCachedKpis(data: Record<string, unknown>) {
  try { sessionStorage.setItem(KPI_CACHE_KEY, JSON.stringify({ data, ts: Date.now() })); } catch { /* ignore */ }
}

export default function MainGrid() {
  const navigate = useNavigate();
  const [cards, setCards] = React.useState<StatCardProps[]>(fallbackCards);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [initialLoading, setInitialLoading] = React.useState(true);
  const [kpisLoaded, setKpisLoaded] = React.useState(false);
  const [dataGridLoaded, setDataGridLoaded] = React.useState(false);

  // Shared dataset sample for insight charts (fetched once, passed to all)
  const [sampleData, setSampleData] = React.useState<Record<string, unknown>[]>([]);
  const [sampleLoading, setSampleLoading] = React.useState(true);

  // Hide full-page overlay as soon as KPIs are in (or timeout). DataGrid and other tiles load in place.
  React.useEffect(() => {
    if (kpisLoaded) {
      setInitialLoading(false);
    }
  }, [kpisLoaded]);

  // Fallback: if backend is slow/unreachable, show the page after 2s so user isn't stuck
  React.useEffect(() => {
    const t = setTimeout(() => setInitialLoading(false), 2000);
    return () => clearTimeout(t);
  }, []);

  // Overview KPIs: real data from User Service API (via Gateway) — GET /dashboard-kpis
  // Initial fetch + poll every 10s so Users count updates live as demo users are added.
  const applyKpis = React.useCallback((json: {
    total_users?: number;
    users_per_day?: number[];
    total_events?: number;
    events_per_day?: number[];
    total_predictions?: number;
    predictions_per_day?: number[];
    total_anomalies?: number;
    anomalies_per_day?: number[];
    anomaly_rate?: number;
  }) => {
    const totalUsers = json.total_users ?? 0;
    const totalEvents = json.total_events ?? 0;
    const totalPredictions = json.total_predictions ?? 0;
    const totalAnomalies = json.total_anomalies ?? 0;
    const anomalyRate = json.anomaly_rate ?? 0;
    const pad = (v: number) => Array(7).fill(v) as number[];
    const usersChartData =
      Array.isArray(json.users_per_day) && json.users_per_day.length === 7
        ? json.users_per_day
        : pad(totalUsers);
    const eventsChartData =
      Array.isArray(json.events_per_day) && json.events_per_day.length === 7
        ? json.events_per_day
        : pad(totalEvents);
    const predictionsChartData =
      Array.isArray(json.predictions_per_day) && json.predictions_per_day.length === 7
        ? json.predictions_per_day
        : pad(totalPredictions);
    const anomaliesChartData =
      Array.isArray(json.anomalies_per_day) && json.anomalies_per_day.length === 7
        ? json.anomalies_per_day
        : pad(anomalyRate);
    setCards([
      { title: 'Users', value: totalUsers.toLocaleString(), interval: 'Users created per day (last 7 days)', trend: 'neutral', data: usersChartData, chartVariant: 'sparkline', gaugeValue: 0 },
      { title: 'Events', value: totalEvents.toLocaleString(), interval: 'Events per day (last 7 days)', trend: 'neutral', data: eventsChartData, chartVariant: 'bar', gaugeValue: 0 },
      { title: 'Anomalies', value: `${totalAnomalies.toLocaleString()} (${anomalyRate.toFixed(1)}%)`, interval: 'Predicted anomalies', trend: anomalyRate > 50 ? 'up' : 'neutral', data: anomaliesChartData, chartVariant: 'progress', gaugeValue: anomalyRate },
      { title: 'Predictions', value: totalPredictions.toLocaleString(), interval: 'Predictions per day (last 7 days)', trend: 'neutral', data: predictionsChartData, chartVariant: 'sparkline', gaugeValue: 0 },
    ]);
  }, []);

  React.useEffect(() => {
    let cancelled = false;

    // Instant paint from session cache (if available) so the page doesn't flash "Loading"
    const cached = getCachedKpis();
    if (cached) {
      applyKpis(cached as Parameters<typeof applyKpis>[0]);
      setKpisLoaded(true);
    }

    const fetchKpis = async (isInitial: boolean) => {
      try {
        if (isInitial && !cached) {
          setLoading(true);
          setError(null);
        }
        const res = await fetch(`${USER_SERVICE_BASE}/dashboard-kpis?t=${Date.now()}`);
        const json = await res.json() as {
          status?: string;
          total_users?: number;
          users_per_day?: number[];
          total_events?: number;
          events_per_day?: number[];
          total_predictions?: number;
          predictions_per_day?: number[];
          total_anomalies?: number;
          anomalies_per_day?: number[];
          anomaly_rate?: number;
          detail?: string;
        };
        if (cancelled) return;
        if (!res.ok || json.status !== 'success') {
          const msg = json.detail || json.status || res.statusText;
          if (!cached) setError(`Failed to load KPIs: ${msg}`);
          return;
        }
        applyKpis(json);
        setCachedKpis(json as Record<string, unknown>);
      } catch (err) {
        if (!cancelled && !cached) {
          console.error('Failed to fetch dashboard KPIs:', err);
          setError('Failed to load dashboard KPIs. Check backend.');
          setCards(fallbackCards);
        }
      } finally {
        if (!cancelled && isInitial) {
          setLoading(false);
          setKpisLoaded(true);
        }
      }
    };

    fetchKpis(true);
    const interval = setInterval(() => fetchKpis(false), 10000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [applyKpis]);

  // Fetch a sample of ingested records for the insight charts (one fetch, shared by all tiles)
  React.useEffect(() => {
    let cancelled = false;
    async function fetchSample() {
      try {
        setSampleLoading(true);
        const tablesRes = await fetch(`${GATEWAY_BASE}/tables`);
        const tablesJson = (await tablesRes.json()) as { status?: string; tables?: string[] };
        if (!tablesRes.ok || tablesJson.status !== 'success' || !tablesJson.tables?.length) return;
        const dsNames = tablesJson.tables
          .filter((t) => t.startsWith('csv_data_'))
          .map((t) => t.replace(/^csv_data_/, ''));
        if (dsNames.length === 0) return;
        const ds = dsNames[0];
        const viewRes = await fetch(`${GATEWAY_BASE}/view?limit=3000&offset=0`, {
          headers: { dataset_name: ds },
        });
        const viewJson = (await viewRes.json()) as {
          data?: { id: number; data: Record<string, unknown> }[];
        };
        if (!cancelled && viewJson.data) {
          setSampleData(viewJson.data.map((r) => r.data));
        }
      } catch (err) {
        console.error('Failed to fetch sample data for insight charts:', err);
      } finally {
        if (!cancelled) setSampleLoading(false);
      }
    }
    fetchSample();
    return () => { cancelled = true; };
  }, []);

  // Callback to notify when DataGrid finishes initial load
  const handleDataGridLoaded = React.useCallback(() => {
    setDataGridLoaded(true);
  }, []);

  const handleKpiCardClick = React.useCallback(() => {
    navigate('/analytics');
  }, [navigate]);

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
      <Typography component="h2" variant="h5" sx={{ mt: 3, mb: -6 }}>
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
      <Grid container spacing={2} columns={12} sx={{ mt: -2, mb: (theme) => theme.spacing(2) }}>
        {/* Left: 4 KPI tiles shifted down so gap to Records by class = 16px (same as Users–Anomalies) */}
        <Grid size={{ xs: 12, md: 6 }} sx={{ mt: { xs: 0, md: 8 } }}>
          <Grid container spacing={2} columns={12}>
            <Grid size={{ xs: 12, sm: 6 }}>{cards[0] && <StatCard {...cards[0]} onClick={handleKpiCardClick} />}</Grid>
            <Grid size={{ xs: 12, sm: 6 }}>{cards[1] && <StatCard {...cards[1]} onClick={handleKpiCardClick} />}</Grid>
            <Grid size={{ xs: 12, sm: 6 }}>{cards[2] && <StatCard {...cards[2]} onClick={handleKpiCardClick} />}</Grid>
            <Grid size={{ xs: 12, sm: 6 }}>{cards[3] && <StatCard {...cards[3]} onClick={handleKpiCardClick} />}</Grid>
          </Grid>
        </Grid>
        {/* Top right: globe shifted down visually only (transform keeps layout/spacing unchanged) */}
        <Grid size={{ xs: 12, md: 6 }} sx={{ position: 'relative', mt: -6, transform: { xs: 'none', md: 'translateY(40px)' } }}>
          <InteractiveGlobe height={640} seamless />
        </Grid>
      </Grid>

      {/* Live metrics: Request Status (left), Blocking Status (right); then QPS (left), Packet Rate (right) */}
      <Grid container spacing={2} columns={12} sx={{ mb: (theme) => theme.spacing(2) }}>
        <Grid
          size={{ xs: 12, md: 6 }}
          sx={{
            position: 'relative',
            zIndex: 2,
            mt: -20,
            bgcolor: 'background.default',
            minHeight: 1,
          }}
        >
          <LiveRequestsStatusTile />
        </Grid>
        <Grid
          size={{ xs: 12, md: 6 }}
          sx={{
            position: 'relative',
            zIndex: 2,
            mt: -20,
            bgcolor: 'background.default',
            minHeight: 1,
          }}
        >
          <LiveBlockingStatusTile />
        </Grid>
        <Grid
          size={{ xs: 12, md: 6 }}
          sx={{
            position: 'relative',
            zIndex: 2,
            bgcolor: 'background.default',
            minHeight: 1,
          }}
        >
          <LiveQueryPerSecondTile />
        </Grid>
        <Grid
          size={{ xs: 12, md: 6 }}
          sx={{
            position: 'relative',
            zIndex: 2,
            bgcolor: 'background.default',
            minHeight: 1,
          }}
        >
          <LivePacketRateTile />
        </Grid>
        {/* Training vs testing — left half; selected-model summary + model library — right quarter; active model overview — right quarter */}
        <Grid size={{ xs: 12, md: 6 }} sx={{ mt: 1 }}>
          <PageViewsBarChart />
        </Grid>
        <Grid size={{ xs: 12, md: 3 }} sx={{ mt: 1 }}>
          <Stack spacing={2}>
            <FlowStatsLiveTile />
            <TrafficSummaryLiveTile />
          </Stack>
        </Grid>
        <Grid size={{ xs: 12, md: 3 }} sx={{ mt: 1 }}>
          <ModelDataOverviewTile />
        </Grid>
      </Grid>

      {/* Network Insights (left) and Recent Activity (left of right column) — same row, shift up */}
      <Grid container spacing={2} columns={12} sx={{ mt: 2, mb: 1 }}>
        <Grid size={{ xs: 12, md: 5 }}>
          <Typography component="h2" variant="h6">
            Network Insights
          </Typography>
        </Grid>
        <Grid size={{ xs: 12, md: 7 }}>
          <Typography component="h2" variant="h6" sx={{ textAlign: 'left' }}>
            Recent Activity
          </Typography>
        </Grid>
      </Grid>
      <Grid container spacing={2} columns={12} sx={{ mb: 2 }} alignItems="flex-start">
        {/* Left: Attack Categories + Protocol Distribution stacked */}
        <Grid size={{ xs: 12, md: 5 }}>
          <Grid container spacing={2} columns={12}>
            <Grid size={{ xs: 12 }}>
              <AttackCategoryChart data={sampleData} loading={sampleLoading} />
            </Grid>
            <Grid size={{ xs: 12 }}>
              <ProtocolDistChart data={sampleData} loading={sampleLoading} />
            </Grid>
          </Grid>
        </Grid>
        {/* Right: Recent Activity — shift left to align with Attack Categories; fixed height to match Protocol Distribution bottom */}
        <Grid size={{ xs: 12, md: 7 }} sx={{ display: 'flex', flexDirection: 'column', ml: { xs: 0, md: -1 }, mt: { xs: 0, md: -1 } }}>
          <Box sx={{ height: 653, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <CustomizedDataGrid
              onInitialLoadComplete={handleDataGridLoaded}
              hideLoadingDuringInitialLoad={initialLoading}
              fillHeight
            />
          </Box>
        </Grid>
      </Grid>

      <Copyright sx={{ my: 4 }} />
    </Box>
  );
}
