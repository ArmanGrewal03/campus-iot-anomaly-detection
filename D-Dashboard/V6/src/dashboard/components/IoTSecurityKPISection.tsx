import * as React from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Grid from '@mui/material/Grid';
import CircularProgress from '@mui/material/CircularProgress';
import LinearProgress, { linearProgressClasses } from '@mui/material/LinearProgress';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { useTheme } from '@mui/material/styles';
import SecurityRoundedIcon from '@mui/icons-material/SecurityRounded';
import WarningAmberRoundedIcon from '@mui/icons-material/WarningAmberRounded';
import TrendingUpRoundedIcon from '@mui/icons-material/TrendingUpRounded';
import SpeedRoundedIcon from '@mui/icons-material/SpeedRounded';
import PublicRoundedIcon from '@mui/icons-material/PublicRounded';
import ScheduleRoundedIcon from '@mui/icons-material/ScheduleRounded';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import StorageRoundedIcon from '@mui/icons-material/StorageRounded';
import MemoryRoundedIcon from '@mui/icons-material/MemoryRounded';
import DnsRoundedIcon from '@mui/icons-material/DnsRounded';
import CloudRoundedIcon from '@mui/icons-material/CloudRounded';
import LockRoundedIcon from '@mui/icons-material/LockRounded';
import ShieldRoundedIcon from '@mui/icons-material/ShieldRounded';
import BuildRoundedIcon from '@mui/icons-material/BuildRounded';
import TerminalRoundedIcon from '@mui/icons-material/TerminalRounded';
import LanRoundedIcon from '@mui/icons-material/LanRounded';
import HubRoundedIcon from '@mui/icons-material/HubRounded';
import StarRoundedIcon from '@mui/icons-material/StarRounded';
import StarBorderRoundedIcon from '@mui/icons-material/StarBorderRounded';
import AvTimerRoundedIcon from '@mui/icons-material/AvTimerRounded';
import CompareArrowsRoundedIcon from '@mui/icons-material/CompareArrowsRounded';
import TagRoundedIcon from '@mui/icons-material/TagRounded';
import { BarChart } from '@mui/x-charts/BarChart';
import { LineChart } from '@mui/x-charts/LineChart';
import { SparkLineChart } from '@mui/x-charts/SparkLineChart';
import { Gauge } from '@mui/x-charts/Gauge';
import { PieChart } from '@mui/x-charts/PieChart';
import StatCard, { StatCardProps } from './StatCard';
import InteractiveGlobe from './InteractiveGlobe';

/* Mock KPI data – replace with real data when integrating */
const kpiCardsData: StatCardProps[] = [
  {
    title: 'Anomalies Detected',
    value: '247',
    interval: 'Last 24 hours',
    trend: 'up',
    data: [18, 22, 28, 24, 30, 26, 32, 28, 35, 38, 34, 40, 42, 38, 44, 48, 45, 50, 52, 48, 55, 58, 54, 60, 62, 58, 65, 68, 72, 70],
  },
  {
    title: 'Devices Monitored',
    value: '1,284',
    interval: 'Campus IoT & network',
    trend: 'neutral',
    data: [1200, 1210, 1220, 1230, 1240, 1250, 1260, 1255, 1265, 1270, 1275, 1280, 1282, 1284, 1280, 1282, 1284, 1286, 1284, 1282, 1284, 1286, 1284, 1285, 1284, 1283, 1284, 1285, 1284, 1284],
  },
  {
    title: 'Active Alerts',
    value: '12',
    interval: 'Require attention',
    trend: 'down',
    data: [24, 22, 20, 18, 19, 17, 16, 15, 14, 15, 14, 13, 14, 13, 12, 13, 12, 11, 12, 12, 11, 12, 11, 12, 12, 11, 12, 12, 12, 12],
  },
];

const threatLevelValue = 34; // 0–100, mock "risk score"
const threatLevelLabel = threatLevelValue <= 33 ? 'Low' : threatLevelValue <= 66 ? 'Medium' : 'High';

const anomalyTypesData = {
  labels: ['Unauthorized access', 'Malware', 'Suspicious traffic', 'Data exfil', 'Other'],
  values: [42, 28, 18, 8, 4],
};

const recentAlertsMock = [
  { id: '1', severity: 'high' as const, message: 'Multiple failed SSH attempts – Building A', time: '2m ago' },
  { id: '2', severity: 'medium' as const, message: 'Unusual traffic spike – Lab network', time: '15m ago' },
  { id: '3', severity: 'low' as const, message: 'New device joined – IoT hub', time: '1h ago' },
  { id: '4', severity: 'high' as const, message: 'Anomalous data transfer volume', time: '2h ago' },
  { id: '5', severity: 'medium' as const, message: 'Access control anomaly – East wing', time: '3h ago' },
];

const trafficOverTimeLabels = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'];
const trafficOverTimeData = [1200, 800, 2400, 3200, 2800, 3600, 2200];

const deviceHealthMock = [
  { label: 'Healthy', value: 1180, color: 'success.main' },
  { label: 'Warning', value: 72, color: 'warning.main' },
  { label: 'Critical', value: 32, color: 'error.main' },
];

/* Mock data for new visuals (sparklines, heatmap, treemap, leaderboard, etc.) */
const sparklineDataA = [22, 28, 24, 30, 26, 34, 32, 38, 35, 42];
const sparklineDataB = [120, 115, 125, 118, 130, 122, 128, 135, 130, 138];
const sparklineXLabels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'];
const bulletValue = 72;
const bulletTarget = 85;
const bulletMax = 100;
const heatmapData = [
  [12, 18, 22, 15, 20],
  [8, 14, 28, 19, 24],
  [16, 24, 32, 22, 18],
  [10, 20, 26, 30, 14],
  [14, 16, 20, 24, 28],
];
const treemapItems = [
  { id: 'a', value: 40, label: 'Ingest' },
  { id: 'b', value: 30, label: 'Process' },
  { id: 'c', value: 20, label: 'Export' },
  { id: 'd', value: 10, label: 'Other' },
];
const sunburstData = [
  { id: '0', value: 50, label: 'North' },
  { id: '1', value: 30, label: 'South' },
  { id: '2', value: 20, label: 'East' },
];
const slaValue = 99.92;
const slaTarget = 99.9;
const choroplethRegions = [
  { id: 'R1', value: 85, label: 'Zone A' },
  { id: 'R2', value: 62, label: 'Zone B' },
  { id: 'R3', value: 91, label: 'Zone C' },
  { id: 'R4', value: 45, label: 'Zone D' },
];
const leaderboardItems = [
  { rank: 1, name: 'Building A', value: 1240 },
  { rank: 2, name: 'Building B', value: 980 },
  { rank: 3, name: 'Lab 1', value: 756 },
  { rank: 4, name: 'East Wing', value: 542 },
  { rank: 5, name: 'Campus Core', value: 418 },
];
const deltaKpis = [
  { label: 'Events', value: '4.2K', delta: 12, positive: true },
  { label: 'Blocked', value: '89', delta: -5, positive: false },
];
const thresholdValue = 68;
const thresholdZones = [
  { max: 33, color: 'success.main' },
  { max: 66, color: 'warning.main' },
  { max: 100, color: 'error.main' },
];

export default function IoTSecurityKPISection() {
  const theme = useTheme();
  const barColors = [
    (theme.vars || theme).palette.error.main,
    (theme.vars || theme).palette.warning.main,
    (theme.vars || theme).palette.info.main,
    (theme.vars || theme).palette.primary.main,
    (theme.vars || theme).palette.grey[500],
  ];

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' }, mt: 4 }}>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
        <SecurityRoundedIcon sx={{ color: 'primary.main', fontSize: 28 }} />
        <Box>
          <Typography component="h2" variant="h6" sx={{ fontWeight: 600 }}>
            IoT Security & Anomaly KPIs
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Test designs for campus monitoring dashboard — replace with live data when ready
          </Typography>
        </Box>
      </Stack>

      {/* Layout: tiles left, globe right; tiles below globe for seamless look */}
      <Grid container spacing={2} columns={12}>
        {/* Left column: KPI tiles */}
        <Grid size={{ xs: 12, md: 7 }}>
          <Grid container spacing={2} columns={12}>
            {kpiCardsData.map((card, index) => (
              <Grid key={index} size={{ xs: 12, sm: 6 }}>
                <StatCard {...card} />
              </Grid>
            ))}
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card variant="outlined" sx={{ height: '100%', flexGrow: 1 }}>
                <CardContent>
                  <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                    <TrendingUpRoundedIcon sx={{ color: 'text.secondary', fontSize: 20 }} />
                    <Typography component="h2" variant="subtitle2">
                      Threat Level
                    </Typography>
                  </Stack>
                  <Stack spacing={1.5}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Typography variant="h4" component="p">
                        {threatLevelLabel}
                      </Typography>
                      <Chip
                        size="small"
                        label={`${threatLevelValue}%`}
                        color={threatLevelValue > 66 ? 'error' : threatLevelValue > 33 ? 'warning' : 'success'}
                      />
                    </Stack>
                    <Typography variant="caption" color="text.secondary">
                      Current risk score (0–100)
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={threatLevelValue}
                      sx={{
                        height: 8,
                        borderRadius: 1,
                        [`&.${linearProgressClasses.root}`]: {
                          backgroundColor: (theme.vars || theme).palette.action.hover,
                        },
                        [`& .${linearProgressClasses.bar}`]: {
                          borderRadius: 1,
                          backgroundColor:
                            threatLevelValue > 66
                              ? (theme.vars || theme).palette.error.main
                              : threatLevelValue > 33
                                ? (theme.vars || theme).palette.warning.main
                                : (theme.vars || theme).palette.success.main,
                        },
                      }}
                    />
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12 }}>
              <Card variant="outlined" sx={{ width: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Anomaly types (mock)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Distribution by category — last 7 days
                  </Typography>
                  <Box sx={{ width: '100%', height: 260 }}>
                    <BarChart
                      height={240}
                      borderRadius={6}
                      colors={barColors}
                      xAxis={[{ scaleType: 'band', data: anomalyTypesData.labels }]}
                      yAxis={[{}]}
                      series={[{ data: anomalyTypesData.values, label: 'Count', id: 'count' }]}
                      margin={{ top: 20, bottom: 60, left: 50, right: 20 }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card variant="outlined" sx={{ width: '100%', height: '100%' }}>
                <CardContent>
                  <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                    <WarningAmberRoundedIcon sx={{ color: 'warning.main', fontSize: 20 }} />
                    <Typography component="h2" variant="subtitle2">
                      Recent alerts (mock)
                    </Typography>
                  </Stack>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>
                    Sample events for layout testing
                  </Typography>
                  <List dense disablePadding>
                    {recentAlertsMock.map((alert) => (
                      <ListItem key={alert.id} disablePadding sx={{ py: 0.5 }}>
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          <Box
                            sx={{
                              width: 8,
                              height: 8,
                              borderRadius: '50%',
                              bgcolor:
                                alert.severity === 'high'
                                  ? 'error.main'
                                  : alert.severity === 'medium'
                                    ? 'warning.main'
                                    : 'success.main',
                            }}
                          />
                        </ListItemIcon>
                        <ListItemText
                          primary={alert.message}
                          secondary={alert.time}
                          primaryTypographyProps={{ variant: 'body2' }}
                          secondaryTypographyProps={{ variant: 'caption' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card variant="outlined" sx={{ width: '100%', height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Network traffic (mock)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Requests per hour — last 24h
                  </Typography>
                  <Box sx={{ width: '100%', height: 200 }}>
                    <LineChart
                      colors={[(theme.vars || theme).palette.primary.main]}
                      xAxis={[{ scaleType: 'point', data: trafficOverTimeLabels }]}
                      yAxis={[{}]}
                      series={[{ data: trafficOverTimeData, label: 'Requests', id: 'traffic' }]}
                      height={180}
                      margin={{ top: 20, bottom: 30, left: 45, right: 20 }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Avg response time (mock)
                  </Typography>
                  <Typography variant="h4" component="p">
                    42 ms
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                    Detection pipeline
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Device health (mock)
                  </Typography>
                  <Stack spacing={1}>
                    {deviceHealthMock.map((item) => (
                      <Stack key={item.label} direction="row" alignItems="center" justifyContent="space-between">
                        <Typography variant="body2">{item.label}</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {item.value}
                        </Typography>
                      </Stack>
                    ))}
                  </Stack>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                    By status
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* New visuals: sparklines, bullet, heatmap, treemap, sunburst, SLA, choropleth, leaderboard, delta KPIs, threshold */}
            <Grid size={{ xs: 12 }}>
              <Typography component="h3" variant="subtitle2" color="text.secondary" sx={{ mb: 1.5, fontWeight: 600 }}>
                Visuals (mock) — sparklines, bullet, heatmap, treemap, sunburst, SLA, choropleth, leaderboard, delta, threshold
              </Typography>
            </Grid>
            {/* Sparklines */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Sparklines (mock)
                  </Typography>
                  <Stack spacing={2}>
                    <Box>
                      <Typography variant="caption" color="text.secondary">Anomalies</Typography>
                      <Box sx={{ width: '100%', height: 36 }}>
                        <SparkLineChart
                          colors={[(theme.vars || theme).palette.error.main]}
                          data={sparklineDataA}
                          xAxis={{ scaleType: 'band', data: sparklineXLabels }}
                          height={36}
                          showHighlight
                          showTooltip
                        />
                      </Box>
                    </Box>
                    <Box>
                      <Typography variant="caption" color="text.secondary">Throughput</Typography>
                      <Box sx={{ width: '100%', height: 36 }}>
                        <SparkLineChart
                          colors={[(theme.vars || theme).palette.primary.main]}
                          data={sparklineDataB}
                          xAxis={{ scaleType: 'band', data: sparklineXLabels }}
                          height={36}
                          showHighlight
                          showTooltip
                        />
                      </Box>
                    </Box>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            {/* Bullet chart */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Bullet chart (mock)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Current {bulletValue} / Target {bulletTarget}
                  </Typography>
                  <Box sx={{ position: 'relative', height: 24, borderRadius: 1, overflow: 'hidden', bgcolor: 'action.hover' }}>
                    <Box
                      sx={(t) => ({
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        bottom: 0,
                        width: `${(bulletValue / bulletMax) * 100}%`,
                        bgcolor: (t.vars || t).palette.primary.main,
                        borderRadius: 1,
                      })}
                    />
                    <Box
                      sx={(t) => ({
                        position: 'absolute',
                        left: `${(bulletTarget / bulletMax) * 100}%`,
                        top: 0,
                        bottom: 0,
                        width: 2,
                        bgcolor: (t.vars || t).palette.common.black,
                        opacity: 0.8,
                      })}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            {/* Heatmap */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Heatmap (mock)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Activity by segment × time
                  </Typography>
                  <Stack spacing={0.5}>
                    {heatmapData.map((row, i) => (
                      <Stack key={i} direction="row" spacing={0.5}>
                        {row.map((v, j) => (
                          <Box
                            key={j}
                            sx={(t) => ({
                              flex: 1,
                              height: 20,
                              borderRadius: 0.5,
                              bgcolor: (t.vars || t).palette.primary.main,
                              opacity: 0.2 + (v / 32) * 0.8,
                            })}
                          />
                        ))}
                      </Stack>
                    ))}
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            {/* Treemap */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Treemap (mock)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Volume by category
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, width: '100%', height: 100 }}>
                    {treemapItems.map((item, i) => {
                      const pct = (item.value / treemapItems.reduce((s, x) => s + x.value, 0)) * 100;
                      const paletteKeys = ['primary', 'info', 'success', 'warning'] as const;
                      const paletteKey = paletteKeys[i];
                      return (
                        <Box
                          key={item.id}
                          sx={(t) => ({
                            width: `${pct}%`,
                            minWidth: 40,
                            flex: pct < 25 ? `1 1 ${pct}%` : undefined,
                            height: 22,
                            bgcolor: (t.vars || t).palette[paletteKey].main,
                            borderRadius: 1,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          })}
                        >
                          <Typography variant="caption" sx={{ color: 'primary.contrastText', fontWeight: 600 }}>
                            {item.label}
                          </Typography>
                        </Box>
                      );
                    })}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            {/* Sunburst (donut PieChart) */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Sunburst / donut (mock)
                  </Typography>
                  <Box sx={{ width: '100%', height: 160 }}>
                    <PieChart
                      series={[{ data: sunburstData, innerRadius: 40, outerRadius: 60 }]}
                      height={160}
                      width={160}
                      margin={{ top: 5, bottom: 5, left: 5, right: 5 }}
                      colors={[(theme.vars || theme).palette.primary.main, (theme.vars || theme).palette.info.main, (theme.vars || theme).palette.success.main]}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            {/* SLA indicator */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    SLA indicator (mock)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Target ≥ {slaTarget}%
                  </Typography>
                  <Box sx={{ width: '100%', height: 100 }}>
                    <Gauge
                      value={slaValue}
                      valueMin={95}
                      valueMax={100}
                      width={140}
                      height={100}
                      text={({ value }) => (value != null ? `${value?.toFixed(2)}%` : '')}
                    />
                  </Box>
                  <Chip
                    size="small"
                    label={slaValue >= slaTarget ? 'Met' : 'Below'}
                    color={slaValue >= slaTarget ? 'success' : 'error'}
                    sx={{ mt: 0.5 }}
                  />
                </CardContent>
              </Card>
            </Grid>
            {/* Choropleth-style (region grid) */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Choropleth-style (mock)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Coverage by zone
                  </Typography>
                  <Stack direction="row" flexWrap="wrap" gap={0.5}>
                    {choroplethRegions.map((r) => (
                      <Box
                        key={r.id}
                        sx={(t) => ({
                          width: 'calc(50% - 4px)',
                          minWidth: 60,
                          p: 0.75,
                          borderRadius: 1,
                          bgcolor: (t.vars || t).palette.primary.main,
                          opacity: 0.2 + (r.value / 100) * 0.8,
                          textAlign: 'center',
                        })}
                      >
                        <Typography variant="caption" sx={{ color: 'primary.contrastText', fontWeight: 600 }}>
                          {r.label}
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'primary.contrastText', display: 'block' }}>
                          {r.value}%
                        </Typography>
                      </Box>
                    ))}
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            {/* Leaderboard */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Leaderboard (mock)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Top 5 by events
                  </Typography>
                  <List dense disablePadding>
                    {leaderboardItems.map((item) => (
                      <ListItem key={item.rank} disablePadding sx={{ py: 0.25 }}>
                        <ListItemIcon sx={{ minWidth: 28 }}>
                          <Typography variant="body2" fontWeight={700} color="text.secondary">
                            #{item.rank}
                          </Typography>
                        </ListItemIcon>
                        <ListItemText
                          primary={item.name}
                          secondary={item.value.toLocaleString()}
                          primaryTypographyProps={{ variant: 'body2' }}
                          secondaryTypographyProps={{ variant: 'caption' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
            {/* Delta / change KPIs */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Delta / change KPIs (mock)
                  </Typography>
                  <Stack spacing={1.5}>
                    {deltaKpis.map((k) => (
                      <Stack key={k.label} direction="row" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={0.5}>
                        <Typography variant="body2" color="text.secondary">
                          {k.label}
                        </Typography>
                        <Stack direction="row" alignItems="center" spacing={0.5}>
                          <Typography variant="body2" fontWeight={600}>
                            {k.value}
                          </Typography>
                          <Chip
                            size="small"
                            label={k.delta > 0 ? `+${k.delta}%` : `${k.delta}%`}
                            color={k.positive ? 'success' : 'error'}
                            variant="outlined"
                            sx={{ height: 20, fontSize: '0.7rem' }}
                          />
                        </Stack>
                      </Stack>
                    ))}
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            {/* Threshold-based status indicator */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Threshold status (mock)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Risk score: {thresholdValue} — {thresholdValue <= 33 ? 'Low' : thresholdValue <= 66 ? 'Medium' : 'High'}
                  </Typography>
                  <Box sx={{ position: 'relative', height: 16, borderRadius: 1, overflow: 'hidden', display: 'flex' }}>
                    {thresholdZones.map((zone, i) => {
                      const prevMax = i === 0 ? 0 : thresholdZones[i - 1].max;
                      const zoneColor = zone.color === 'success.main' ? 'success' : zone.color === 'warning.main' ? 'warning' : 'error';
                      return (
                        <Box
                          key={i}
                          sx={(t) => ({
                            flex: zone.max - prevMax,
                            height: '100%',
                            bgcolor: (t.vars || t).palette[zoneColor].main,
                          })}
                        />
                      );
                    })}
                  </Box>
                  <Box
                    sx={(t) => ({
                      position: 'absolute',
                      left: `${thresholdValue}%`,
                      top: -2,
                      bottom: -2,
                      width: 3,
                      bgcolor: (t.vars || t).palette.common.black,
                      borderRadius: 1,
                      transform: 'translateX(-50%)',
                    })}
                  />
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Right column: globe only (no card/section) + small tiles below – seamless */}
        <Grid size={{ xs: 12, md: 5 }} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'stretch' }}>
          <Box sx={{ flex: 1, minHeight: 420 }}>
            <InteractiveGlobe height={420} seamless />
          </Box>
          <Grid container spacing={2} columns={12} sx={{ mt: 2 }}>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Model rate (mock)
                  </Typography>
                  <Typography variant="h5" component="p">
                    94.2%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Accuracy
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Blocked (mock)
                  </Typography>
                  <Typography variant="h5" component="p">
                    89
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Last 24h
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent>
                  <Typography component="h2" variant="subtitle2" gutterBottom>
                    Segments
                  </Typography>
                  <Typography variant="h5" component="p">
                    8
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Monitored
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Brand new tiles – different designs */}
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card
                variant="outlined"
                sx={(theme) => ({
                  height: '100%',
                  borderLeft: `4px solid ${(theme.vars || theme).palette.success.main}`,
                  borderRadius: 2,
                })}
              >
                <CardContent>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={1}>
                    <Stack direction="row" alignItems="center" spacing={1}>
                      <CheckCircleRoundedIcon sx={{ color: 'success.main', fontSize: 28 }} />
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          System status
                        </Typography>
                        <Typography variant="h6" component="p" sx={{ fontWeight: 600 }}>
                          Operational
                        </Typography>
                      </Box>
                    </Stack>
                    <Chip label="Live" size="small" color="success" sx={{ fontWeight: 600 }} />
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card
                variant="outlined"
                sx={(theme) => ({
                  height: '100%',
                  background: `linear-gradient(135deg, ${(theme.vars || theme).palette.primary.main}08 0%, transparent 60%)`,
                  borderRadius: 2,
                })}
              >
                <CardContent>
                  <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 1 }}>
                    <SpeedRoundedIcon sx={{ color: 'primary.main', fontSize: 24 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      Uptime (mock)
                    </Typography>
                  </Stack>
                  <Stack direction="row" alignItems="baseline" spacing={1}>
                    <Typography variant="h4" component="p" sx={{ fontWeight: 700 }}>
                      99.94%
                    </Typography>
                    <Chip label="+0.01%" size="small" color="success" variant="outlined" />
                  </Stack>
                  <Typography variant="caption" color="text.secondary">
                    Last 30 days
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    Pipeline health (mock)
                  </Typography>
                  <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.5 }}>
                    <Typography variant="h6" component="p">
                      87%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      ingestion
                    </Typography>
                  </Stack>
                  <LinearProgress
                    variant="determinate"
                    value={87}
                    sx={{
                      height: 6,
                      borderRadius: 1,
                      [`&.${linearProgressClasses.root}`]: {
                        backgroundColor: (theme.vars || theme).palette.action.hover,
                      },
                      [`& .${linearProgressClasses.bar}`]: {
                        borderRadius: 1,
                      },
                    }}
                  />
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card
                variant="outlined"
                sx={(theme) => ({
                  height: '100%',
                  borderRadius: 2,
                  borderTop: `3px solid ${(theme.vars || theme).palette.info.main}`,
                })}
              >
                <CardContent>
                  <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                    <PublicRoundedIcon sx={{ color: 'info.main', fontSize: 22 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      Top regions (mock)
                    </Typography>
                  </Stack>
                  <Stack spacing={0.5}>
                    {['North America · 42%', 'Europe · 28%', 'Asia-Pacific · 18%'].map((label, i) => (
                      <Typography key={i} variant="body2" color="text.secondary">
                        {label}
                      </Typography>
                    ))}
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent>
                  <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                    <ScheduleRoundedIcon sx={{ color: 'text.secondary', fontSize: 22 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      Last scan (mock)
                    </Typography>
                  </Stack>
                  <Typography variant="h6" component="p" sx={{ fontWeight: 600 }}>
                    2 min ago
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Full network sweep
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <Card
                variant="outlined"
                sx={(theme) => ({
                  height: '100%',
                  borderRadius: 2,
                  bgcolor: 'action.hover',
                })}
              >
                <CardContent>
                  <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                    <StorageRoundedIcon sx={{ color: 'text.secondary', fontSize: 22 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      Throughput (mock)
                    </Typography>
                  </Stack>
                  <Stack direction="row" spacing={2}>
                    <Box>
                      <Typography variant="caption" color="text.secondary">In</Typography>
                      <Typography variant="h6" component="p">12.4 K/s</Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" color="text.secondary">Out</Typography>
                      <Typography variant="h6" component="p">8.1 K/s</Typography>
                    </Box>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>

            {/* Extra row to fill space under globe */}
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Latency (mock)</Typography>
                  <Typography variant="h6" component="p">38 ms</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Events today (mock)</Typography>
                  <Typography variant="h6" component="p">1,247</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Data volume (mock)</Typography>
                  <Typography variant="h6" component="p">2.4 GB</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">API calls (mock)</Typography>
                  <Typography variant="h6" component="p">4.2K</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Sessions (mock)</Typography>
                  <Typography variant="h6" component="p">892</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Cache hit (mock)</Typography>
                  <Typography variant="h6" component="p">94%</Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* More tiles under globe */}
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Queries/min (mock)</Typography>
                  <Typography variant="h6" component="p">412</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Active users (mock)</Typography>
                  <Typography variant="h6" component="p">24</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Retries (mock)</Typography>
                  <Typography variant="h6" component="p">12</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Failover (mock)</Typography>
                  <Typography variant="h6" component="p">0</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Log level (mock)</Typography>
                  <Typography variant="h6" component="p">INFO</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Backup (mock)</Typography>
                  <Typography variant="h6" component="p">OK</Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* 20+ varied tile designs – different styles, theme-aligned */}
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 3, borderLeft: `4px solid ${(t.vars || t).palette.warning.main}` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Stack direction="row" alignItems="center" spacing={1}>
                    <MemoryRoundedIcon sx={{ color: 'warning.main', fontSize: 20 }} />
                    <Typography variant="caption" color="text.secondary">Memory (mock)</Typography>
                  </Stack>
                  <Typography variant="h6" component="p" sx={{ fontWeight: 700 }}>62%</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 0, borderRight: `4px solid ${(t.vars || t).palette.error.main}` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Errors (mock)</Typography>
                  <Typography variant="h6" component="p" color="error.main">3</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, borderTop: `3px solid ${(t.vars || t).palette.success.main}` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <DnsRoundedIcon sx={{ color: 'success.main', fontSize: 22, mb: 0.5 }} />
                  <Typography variant="caption" color="text.secondary">DNS (mock)</Typography>
                  <Typography variant="h6" component="p">Healthy</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card sx={(t) => ({ height: '100%', borderRadius: 2, bgcolor: (t.vars || t).palette.primary.main, color: (t.vars || t).palette.primary.contrastText })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" sx={{ opacity: 0.9 }}>Connections (mock)</Typography>
                  <Typography variant="h5" component="p" sx={{ fontWeight: 700 }}>1,842</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, borderStyle: 'dashed' }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Placeholder (mock)</Typography>
                  <Typography variant="body2" color="text.secondary" fontStyle="italic">—</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5, display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">CPU (mock)</Typography>
                    <Typography variant="h6" component="p">28%</Typography>
                  </Box>
                  <CloudRoundedIcon sx={{ color: 'text.secondary', fontSize: 32, opacity: 0.6 }} />
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Auth (mock)</Typography>
                  <Stack direction="row" alignItems="baseline" spacing={1}>
                    <Typography variant="h6" component="p">2FA</Typography>
                    <Chip label="On" size="small" color="success" />
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, bgcolor: (t.vars || t).palette.success.main + '14' })}>
                <CardContent sx={{ py: 1.5 }}>
                  <LockRoundedIcon sx={{ color: 'success.main', fontSize: 20 }} />
                  <Typography variant="caption" color="text.secondary">Encryption (mock)</Typography>
                  <Typography variant="h6" component="p" sx={{ fontWeight: 600 }}>AES-256</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Stack (mock)</Typography>
                  <Stack spacing={0.25} sx={{ mt: 0.5 }}>
                    <Typography variant="body2">Ingest: 1.2K</Typography>
                    <Typography variant="body2">Process: 980</Typography>
                    <Typography variant="body2">Export: 890</Typography>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Health (mock)</Typography>
                  <LinearProgress variant="determinate" value={78} sx={{ height: 6, borderRadius: 1, mt: 0.5 }} />
                  <Typography variant="body2" sx={{ mt: 0.5 }}>78%</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, borderBottom: `3px solid ${(t.vars || t).palette.info.main}` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <ShieldRoundedIcon sx={{ color: 'info.main', fontSize: 22 }} />
                  <Typography variant="caption" color="text.secondary">Firewall (mock)</Typography>
                  <Typography variant="h6" component="p">Active</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                    <Chip label="v2.1" size="small" variant="outlined" />
                    <Chip label="Stable" size="small" color="success" />
                  </Stack>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>Version (mock)</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5, textAlign: 'center' }}>
                  <Typography variant="h4" component="p" sx={{ fontWeight: 800, color: 'primary.main' }}>99.9</Typography>
                  <Typography variant="caption" color="text.secondary">SLA % (mock)</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, bgcolor: (t.vars || t).palette.warning.main + '12' })}>
                <CardContent sx={{ py: 1.5 }}>
                  <BuildRoundedIcon sx={{ color: 'warning.main', fontSize: 22 }} />
                  <Typography variant="caption" color="text.secondary">Build (mock)</Typography>
                  <Typography variant="h6" component="p">#2041</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Queue (mock)</Typography>
                  <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                    <Box sx={{ flex: 1, textAlign: 'center', p: 0.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                      <Typography variant="body2" fontWeight={600}>12</Typography>
                      <Typography variant="caption" color="text.secondary">Pending</Typography>
                    </Box>
                    <Box sx={{ flex: 1, textAlign: 'center', p: 0.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                      <Typography variant="body2" fontWeight={600}>48</Typography>
                      <Typography variant="caption" color="text.secondary">Done</Typography>
                    </Box>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, boxShadow: 1 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <TerminalRoundedIcon sx={{ color: 'text.secondary', fontSize: 20 }} />
                  <Typography variant="caption" color="text.secondary">CLI (mock)</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', mt: 0.5 }}>ready</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, borderLeft: `4px solid ${(t.vars || t).palette.secondary.main}` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Replicas (mock)</Typography>
                  <Typography variant="h6" component="p">3 / 3</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Peers (mock)</Typography>
                  <List dense disablePadding sx={{ mt: 0.5 }}>
                    <ListItem disablePadding sx={{ py: 0 }}>
                      <ListItemIcon sx={{ minWidth: 28 }}><Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: 'success.main' }} /></ListItemIcon>
                      <ListItemText primary="Node A" secondary="Primary" primaryTypographyProps={{ variant: 'body2' }} secondaryTypographyProps={{ variant: 'caption' }} />
                    </ListItem>
                    <ListItem disablePadding sx={{ py: 0 }}>
                      <ListItemIcon sx={{ minWidth: 28 }}><Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: 'text.secondary' }} /></ListItemIcon>
                      <ListItemText primary="Node B" secondary="Replica" primaryTypographyProps={{ variant: 'body2' }} secondaryTypographyProps={{ variant: 'caption' }} />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, background: `linear-gradient(180deg, ${(t.vars || t).palette.info.main}10 0%, transparent 100%)` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <LanRoundedIcon sx={{ color: 'info.main', fontSize: 22 }} />
                  <Typography variant="caption" color="text.secondary">LAN (mock)</Typography>
                  <Typography variant="h6" component="p">192.168.1.x</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Typography variant="caption" color="text.secondary">Jitter (mock)</Typography>
                    <Chip label="Low" size="small" variant="outlined" color="success" />
                  </Stack>
                  <Typography variant="h6" component="p">2.1 ms</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <HubRoundedIcon sx={{ color: 'primary.main', fontSize: 24, display: 'block', mb: 0.5 }} />
                  <Typography variant="caption" color="text.secondary">Hubs (mock)</Typography>
                  <Typography variant="h6" component="p">5</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, border: `2px solid ${(t.vars || t).palette.divider}` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Throttle (mock)</Typography>
                  <Typography variant="h6" component="p">1K/s</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 4 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Timezone (mock)</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>UTC</Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Full-width row: 20+ tiles with distinctly different designs */}
        <Grid size={{ xs: 12 }} sx={{ mt: 2 }}>
          <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1.5, fontWeight: 600 }}>
            More KPI tiles — varied designs (mock)
          </Typography>
          <Grid container spacing={2} columns={24}>
            {/* 1. Circular progress with value in center */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 2, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Box sx={{ position: 'relative', display: 'inline-flex' }}>
                    <CircularProgress variant="determinate" value={72} size={64} thickness={4} sx={{ color: 'primary.main' }} />
                    <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="body2" fontWeight={700}>72%</Typography>
                    </Box>
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>Load (mock)</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 2. Split: left color block, right content */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', overflow: 'hidden', borderRadius: 2 }}>
                <Stack direction="row" sx={{ minHeight: 90 }}>
                  <Box sx={{ width: 8, bgcolor: 'error.main', flexShrink: 0 }} />
                  <CardContent sx={{ flex: 1, py: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Typography variant="caption" color="text.secondary">Critical (mock)</Typography>
                    <Typography variant="h6" component="p" color="error.main">2</Typography>
                  </CardContent>
                </Stack>
              </Card>
            </Grid>
            {/* 3. Single huge number, no border */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', border: 'none', boxShadow: 'none', bgcolor: 'transparent' }}>
                <CardContent sx={{ py: 1.5, textAlign: 'center' }}>
                  <Typography variant="h3" component="p" sx={{ fontWeight: 800, letterSpacing: '-0.02em' }}>4.2K</Typography>
                  <Typography variant="caption" color="text.secondary">Requests (mock)</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 4. Status dots row */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>Signal (mock)</Typography>
                  <Stack direction="row" spacing={0.5}>
                    {[1, 2, 3, 4, 5].map((i) => (
                      <Box key={i} sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: i <= 4 ? 'success.main' : 'action.hover' }} />
                    ))}
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            {/* 5. Table-style key-value */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5, px: 2 }}>
                  <Stack direction="row" justifyContent="space-between" sx={{ borderBottom: 1, borderColor: 'divider', py: 0.5 }}>
                    <Typography variant="body2" color="text.secondary">Host</Typography>
                    <Typography variant="body2" fontWeight={600}>api-01</Typography>
                  </Stack>
                  <Stack direction="row" justifyContent="space-between" sx={{ py: 0.5 }}>
                    <Typography variant="body2" color="text.secondary">Port</Typography>
                    <Typography variant="body2" fontWeight={600}>443</Typography>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            {/* 6. Badge overlay style */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, position: 'relative' }}>
                <Chip label="BETA" size="small" sx={{ position: 'absolute', top: 8, right: 8, fontSize: '0.65rem' }} color="primary" />
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Feature (mock)</Typography>
                  <Typography variant="h6" component="p">v2</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 7. Sparkline + value */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={1}>
                    <Box>
                      <Typography variant="caption" color="text.secondary">Trend (mock)</Typography>
                      <Typography variant="h6" component="p">↑ 12%</Typography>
                    </Box>
                    <Box sx={{ width: 56, height: 28 }}>
                      <LineChart
                        colors={[(theme.vars || theme).palette.success.main]}
                        xAxis={[{ scaleType: 'point', data: ['1', '2', '3', '4', '5', '6'] }]}
                        series={[{ data: [2, 5, 3, 8, 6, 10], id: 's' }]}
                        height={28}
                        margin={{ top: 2, bottom: 2, left: 2, right: 2 }}
                      />
                    </Box>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            {/* 8. Two-tone top half background */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, background: `linear-gradient(180deg, ${(t.vars || t).palette.primary.main}12 0%, transparent 45%)` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Bandwidth (mock)</Typography>
                  <Typography variant="h6" component="p" sx={{ fontWeight: 700 }}>128 Mbps</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 9. Icon top-right corner */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, position: 'relative' }}>
                <SpeedRoundedIcon sx={{ position: 'absolute', top: 8, right: 8, fontSize: 20, color: 'text.secondary', opacity: 0.5 }} />
                <CardContent sx={{ py: 1.5, pr: 4 }}>
                  <Typography variant="caption" color="text.secondary">Throughput (mock)</Typography>
                  <Typography variant="h6" component="p">98%</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 10. Star rating style */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>Score (mock)</Typography>
                  <Stack direction="row" spacing={0.25}>
                    {[1, 2, 3, 4, 5].map((i) => (i <= 4 ? <StarRoundedIcon key={i} sx={{ fontSize: 18, color: 'warning.main' }} /> : <StarBorderRoundedIcon key={i} sx={{ fontSize: 18, color: 'action.disabled' }} />))}
                  </Stack>
                  <Typography variant="body2" color="text.secondary">4/5</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 11. Chip cloud */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Stack direction="row" alignItems="center" spacing={0.5} sx={{ mb: 0.5 }}>
                    <TagRoundedIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
                    <Typography variant="caption" color="text.secondary">Tags (mock)</Typography>
                  </Stack>
                  <Stack direction="row" flexWrap="wrap" gap={0.5} useFlexGap>
                    <Chip label="IoT" size="small" variant="outlined" />
                    <Chip label="Campus" size="small" variant="outlined" />
                    <Chip label="Secure" size="small" variant="outlined" />
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            {/* 12. Quote / callout style */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, borderLeft: `4px solid ${(t.vars || t).palette.info.main}` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="body2" fontStyle="italic" color="text.secondary">
                    All systems nominal (mock)
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 13. Today vs Yesterday */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Stack direction="row" alignItems="center" spacing={0.5} sx={{ mb: 0.5 }}>
                    <CompareArrowsRoundedIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
                    <Typography variant="caption" color="text.secondary">Compare (mock)</Typography>
                  </Stack>
                  <Stack direction="row" spacing={1}>
                    <Typography variant="body2">Today <strong>1.2K</strong></Typography>
                    <Typography variant="body2" color="text.secondary">vs 1.1K</Typography>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            {/* 14. Timer / countdown style */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <AvTimerRoundedIcon sx={{ color: 'primary.main', fontSize: 28, mb: 0.5 }} />
                  <Typography variant="h5" component="p" sx={{ fontVariantNumeric: 'tabular-nums' }}>00:42</Typography>
                  <Typography variant="caption" color="text.secondary">Avg duration (mock)</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 15. Avatar / initials circle */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5, display: 'flex', alignItems: 'center', gap: 1.5 }}>
                  <Box sx={{ width: 40, height: 40, borderRadius: '50%', bgcolor: 'primary.main', color: 'primary.contrastText', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '0.875rem' }}>
                    API
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Service (mock)</Typography>
                    <Typography variant="body2" fontWeight={600}>Gateway</Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            {/* 16. Striped background */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, background: `repeating-linear-gradient(-45deg, transparent, transparent 6px, ${(t.vars || t).palette.action.hover} 6px, ${(t.vars || t).palette.action.hover} 12px)` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Draft (mock)</Typography>
                  <Typography variant="body2" fontWeight={600}>3 items</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 17. Thick bottom border only */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, border: 'none', borderBottom: `4px solid ${(t.vars || t).palette.warning.main}` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Pending (mock)</Typography>
                  <Typography variant="h6" component="p">7</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 18. Metric with delta badge */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Detections (mock)</Typography>
                  <Stack direction="row" alignItems="baseline" spacing={1} flexWrap="wrap">
                    <Typography variant="h6" component="p">156</Typography>
                    <Chip label="+8%" size="small" color="success" sx={{ height: 20, fontSize: '0.7rem' }} />
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
            {/* 19. Minimal label + huge number */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 2, textAlign: 'center' }}>
                  <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: 1 }}>Nodes (mock)</Typography>
                  <Typography variant="h3" component="p" sx={{ fontWeight: 800, mt: 0.5 }}>24</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 20. Thick left + gradient */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, borderLeft: `4px solid ${(t.vars || t).palette.secondary.main}`, background: `linear-gradient(90deg, ${(t.vars || t).palette.secondary.main}08 0%, transparent 40%)` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Replica lag (mock)</Typography>
                  <Typography variant="h6" component="p">0.2s</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 21. Monospace block */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, bgcolor: 'grey.900', color: 'grey.100' }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" sx={{ color: 'grey.400' }}>Log (mock)</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem', mt: 0.5 }}>OK 200</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 22. Rounded pill value */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>Status (mock)</Typography>
                  <Chip label="Synced" size="small" color="success" sx={{ borderRadius: 3, fontWeight: 600 }} />
                </CardContent>
              </Card>
            </Grid>
            {/* 23. Inset / double border look */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={(t) => ({ height: '100%', borderRadius: 2, boxShadow: `inset 0 0 0 2px ${(t.vars || t).palette.divider}` })}>
                <CardContent sx={{ py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">Read-only (mock)</Typography>
                  <Typography variant="body2" fontWeight={600}>Yes</Typography>
                </CardContent>
              </Card>
            </Grid>
            {/* 24. Dot + label inline */}
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent sx={{ py: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'success.main', animation: 'pulse 2s ease-in-out infinite', '@keyframes pulse': { '0%, 100%': { opacity: 1 }, '50%': { opacity: 0.5 } } }} />
                  <Box>
                    <Typography variant="caption" color="text.secondary">Stream (mock)</Typography>
                    <Typography variant="body2" fontWeight={600}>Active</Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
}
