import * as React from 'react';
import { useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { SparkLineChart } from '@mui/x-charts/SparkLineChart';
import { areaElementClasses } from '@mui/x-charts/LineChart';
import { BarChart } from '@mui/x-charts/BarChart';
import PeopleRoundedIcon from '@mui/icons-material/PeopleRounded';
import WifiTetheringErrorRoundedIcon from '@mui/icons-material/WifiTetheringErrorRounded';
import RouterRoundedIcon from '@mui/icons-material/RouterRounded';
import AssessmentRoundedIcon from '@mui/icons-material/AssessmentRounded';

export type StatCardProps = {
  title: string;
  value: string;
  interval: string;
  trend: 'up' | 'down' | 'neutral';
  data: number[];
  chartVariant?: 'sparkline' | 'progress' | 'bar';
  gaugeValue?: number;
  onClick?: () => void;
};

function getXAxisLabels(length: number): string[] {
  const now = new Date();
  return Array.from({ length }, (_, i) => {
    const d = new Date(now);
    d.setDate(d.getDate() - (length - 1 - i));
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  });
}

function AreaGradient({ color, id }: { color: string; id: string }) {
  return (
    <defs>
      <linearGradient id={id} x1="50%" y1="0%" x2="50%" y2="100%">
        <stop offset="0%" stopColor={color} stopOpacity={0.4} />
        <stop offset="100%" stopColor={color} stopOpacity={0} />
      </linearGradient>
    </defs>
  );
}

const cardIcons: Record<string, React.ReactElement> = {
  Users: <PeopleRoundedIcon fontSize="small" />,
  Anomalies: <WifiTetheringErrorRoundedIcon fontSize="small" />,
  Events: <RouterRoundedIcon fontSize="small" />,
  Predictions: <AssessmentRoundedIcon fontSize="small" />,
};

const accentColors: Record<string, string> = {
  Users: 'primary.main',
  Anomalies: 'warning.main',
  Events: 'success.main',
  Predictions: 'info.main',
};

export default function StatCard({
  title,
  value,
  interval,
  trend,
  data,
  chartVariant = 'bar',
  gaugeValue = 0,
  onClick,
}: StatCardProps) {
  const theme = useTheme();
  const isLight = theme.palette.mode === 'light';

  const chartData = data.length >= 2 ? data : [data[0] ?? 0, data[0] ?? 0];
  const xAxisLabels = getXAxisLabels(chartData.length);
  const gradientId = `sparkline-grad-${title}`;
  const accentColor = accentColors[title] ?? 'primary.main';

  const resolvedAccent =
    title === 'Users'
      ? (isLight ? theme.palette.primary.main : theme.palette.primary.light)
      : title === 'Anomalies'
      ? (isLight ? theme.palette.warning.main : theme.palette.warning.light)
      : title === 'Events'
      ? (isLight ? theme.palette.success.main : theme.palette.success.light)
      : (isLight ? theme.palette.info.main : theme.palette.info.light);

  const chipColor: 'success' | 'error' | 'warning' | 'default' =
    trend === 'up' ? 'error' : trend === 'down' ? 'success' : 'default';

  const chipLabel = title === 'Anomalies'
    ? (gaugeValue > 50 ? 'High Risk' : gaugeValue > 20 ? 'Moderate' : 'Low Risk')
    : 'Live';

  const chipColorResolved: 'success' | 'error' | 'warning' | 'default' =
    chipLabel === 'Live'
      ? 'error'
      : title === 'Anomalies'
        ? (gaugeValue > 50 ? 'error' : gaugeValue > 20 ? 'warning' : 'success')
        : chipColor;

  const anomalyColor =
    gaugeValue > 50 ? theme.palette.error.main
    : gaugeValue > 20 ? theme.palette.warning.main
    : theme.palette.success.main;

  const anomalyPct = Math.min(100, Math.max(0, gaugeValue));

  return (
    <Card
      variant="outlined"
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onClick();
        }
      } : undefined}
      sx={{
        height: '100%',
        flexGrow: 1,
        maxHeight: 230,
        borderTop: `3px solid`,
        borderTopColor: accentColor,
        display: 'flex',
        flexDirection: 'column',
        cursor: onClick ? 'pointer' : 'default',
        transition: onClick ? 'transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease' : undefined,
        '&:hover': onClick
          ? {
              transform: 'translateY(-2px)',
              boxShadow: 4,
            }
          : undefined,
      }}
    >
      <CardContent
        sx={{
          py: 1.25,
          px: 1.5,
          '&:last-child': { pb: 1.25 },
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
        }}
      >
        {/* Header: icon + title + chip */}
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Box sx={{ color: accentColor, display: 'flex' }}>
              {cardIcons[title] ?? null}
            </Box>
            <Typography component="h2" variant="subtitle2" sx={{ fontSize: '0.8125rem' }}>
              {title}
            </Typography>
          </Stack>
          <Chip
            size="small"
            color={chipColorResolved}
            label={chipLabel}
            sx={{ fontSize: '0.625rem', height: 18 }}
          />
        </Stack>

        {/* Value + interval (Users: animate when value updates for live feel) */}
        {title === 'Users' ? (
          <Box
            key={value}
            sx={{
              fontWeight: 600,
              mb: 0,
              '@keyframes usersValueIn': {
                '0%': { opacity: 0, transform: 'translateY(-6px)' },
                '100%': { opacity: 1, transform: 'translateY(0)' },
              },
              animation: 'usersValueIn 0.35s ease-out',
            }}
          >
            <Typography variant="h5" component="p" sx={{ fontWeight: 600 }}>
              {value}
            </Typography>
          </Box>
        ) : (
          <Typography variant="h5" component="p" sx={{ fontWeight: 600, mb: 0 }}>
            {value}
          </Typography>
        )}
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, lineHeight: 1.2 }}>
          {interval}
        </Typography>

        {/* Chart area — pushed lower via flex */}
        <Box sx={{ mt: 'auto', minHeight: 0, flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}>
          {chartVariant === 'sparkline' && (
            <Box sx={{ width: '100%', height: 44 }}>
              <SparkLineChart
                colors={[resolvedAccent]}
                data={chartData}
                area
                showHighlight
                showTooltip
                curve="natural"
                xAxis={{ scaleType: 'band', data: xAxisLabels }}
                sx={{
                  [`& .${areaElementClasses.root}`]: {
                    fill: `url(#${gradientId})`,
                  },
                }}
              >
                <AreaGradient color={resolvedAccent} id={gradientId} />
              </SparkLineChart>
            </Box>
          )}

          {chartVariant === 'progress' && (
            <Stack spacing={0.5} sx={{ width: '100%' }}>
              <Box
                sx={{
                  width: '100%',
                  height: 10,
                  borderRadius: 1,
                  bgcolor: theme.palette.divider || theme.palette.action.disabledBackground,
                  overflow: 'hidden',
                }}
              >
                <Box
                  sx={{
                    width: `${anomalyPct}%`,
                    height: '100%',
                    bgcolor: anomalyColor,
                    borderRadius: 1,
                    transition: 'width 0.3s ease',
                  }}
                />
              </Box>
              <Stack direction="row" justifyContent="space-between">
                <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
                  0%
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 600, fontSize: '0.65rem' }}>
                  {gaugeValue > 50 ? 'High Risk' : gaugeValue > 20 ? 'Moderate' : 'Low Risk'}
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
                  100%
                </Typography>
              </Stack>
            </Stack>
          )}

          {chartVariant === 'bar' && (
            <Box sx={{ width: '100%', height: 48 }}>
              <BarChart
                borderRadius={4}
                colors={[resolvedAccent]}
                xAxis={[
                  {
                    scaleType: 'band',
                    data: xAxisLabels,
                    tickLabelStyle: { fontSize: 8 },
                  },
                ]}
                yAxis={[{ tickMinStep: 1 }]}
                series={[{ id: 'kpi', data: chartData, label: title }]}
                height={48}
                margin={{ top: 2, right: 2, bottom: 18, left: 2 }}
                grid={{ horizontal: true }}
                slotProps={{ legend: { hidden: true } }}
              />
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}
