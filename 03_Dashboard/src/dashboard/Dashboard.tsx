import type {} from '@mui/x-date-pickers/themeAugmentation';
import type {} from '@mui/x-charts/themeAugmentation';
import type {} from '@mui/x-data-grid/themeAugmentation';
import type {} from '@mui/x-tree-view/themeAugmentation';
import * as React from 'react';
import { alpha } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import { Outlet } from 'react-router-dom';
import AppNavbar from './components/AppNavbar';
import Header from './components/Header';
import SideMenu from './components/SideMenu';
import { addActivityLog, updateActivityLog } from './components/activityLog';
import AppTheme from '../shared-theme/AppTheme';
import {
  chartsCustomizations,
  dataGridCustomizations,
  datePickersCustomizations,
  treeViewCustomizations,
} from './theme/customizations';

const xThemeComponents = {
  ...chartsCustomizations,
  ...dataGridCustomizations,
  ...datePickersCustomizations,
  ...treeViewCustomizations,
};

const USER_SERVICE_BASE = 'http://127.0.0.1:8002';
const WS_BASE = 'ws://127.0.0.1:8002';

type PredictionLike = {
  prediction?: number;
  label?: string;
  probability_safe?: number;
  probability_unsafe?: number;
  confidence?: number;
  attack_cat?: string | null;
};

type HistoryRecordLite = {
  network_id: string;
  prediction_results: {
    predictions?: PredictionLike[];
  } | null;
};

function normalizePredictionLabel(label?: string): string {
  return (label || '').trim().toLowerCase();
}

function getPredictionStatus(prediction: PredictionLike | null | undefined): 'unsafe' | 'safe' | 'pending' {
  if (!prediction) return 'pending';
  if (prediction.prediction === 1) return 'unsafe';
  if (prediction.prediction === 0) return 'safe';

  const normalized = normalizePredictionLabel(prediction.label);
  if (!normalized) return 'pending';
  if (normalized.includes('unsafe') || normalized.includes('anomaly') || normalized.includes('attack')) {
    return 'unsafe';
  }
  if (normalized.includes('safe') || normalized.includes('normal') || normalized.includes('benign')) {
    return 'safe';
  }
  return 'pending';
}

function formatPercent(probability?: number): string | null {
  if (typeof probability !== 'number' || Number.isNaN(probability)) {
    return null;
  }
  return `${(probability * 100).toFixed(1)}%`;
}

function extractPath(inputUrl: string): string {
  try {
    return new URL(inputUrl, window.location.origin).pathname;
  } catch {
    return inputUrl;
  }
}

function actionMessage(path: string, method: string, ok: boolean): string | null {
  if (path.includes('/predict') && method === 'POST') {
    return ok ? 'Prediction completed' : 'Prediction failed';
  }
  if (path.includes('/publish') && method === 'POST') {
    return ok ? 'Prediction request submitted' : 'Prediction request failed';
  }
  if (path.includes('/train') && method === 'POST') {
    return ok ? 'New model trained' : 'Model training failed';
  }
  if (path.includes('/test') && method === 'POST') {
    return ok ? 'Model test completed' : 'Model test failed';
  }
  if (path.includes('/upload') && method === 'POST') {
    return ok ? 'Dataset uploaded' : 'Dataset upload failed';
  }
  if (path.includes('/validate') && method === 'POST') {
    return ok ? 'Dataset validation completed' : 'Dataset validation failed';
  }
  if (path.includes('/set-model') && method === 'POST') {
    return ok ? 'Active model changed' : 'Active model change failed';
  }
  if (/\/models\/[^/]+$/.test(path) && method === 'DELETE') {
    return ok ? 'Model deleted' : 'Model deletion failed';
  }
  if (path.includes('/recompute-predictions') && method === 'POST') {
    return ok ? 'Predictions recomputed' : 'Prediction recompute failed';
  }
  return null;
}

export default function Dashboard(props: { disableCustomTheme?: boolean }) {
  React.useEffect(() => {
    const g = window as Window & {
      __dashboardFetchPatched?: boolean;
      __dashboardOriginalFetch?: typeof fetch;
    };

    if (g.__dashboardFetchPatched) return;

    g.__dashboardOriginalFetch = window.fetch.bind(window);
    window.fetch = async (...args: Parameters<typeof fetch>) => {
      const requestInfo = args[0];
      const url = typeof requestInfo === 'string' ? requestInfo : requestInfo.url;
      const method = (
        args[1]?.method ||
        (typeof requestInfo !== 'string' ? requestInfo.method : 'GET') ||
        'GET'
      ).toUpperCase();
      const path = extractPath(url);

      try {
        const response = await g.__dashboardOriginalFetch!(...args);
        const message = actionMessage(path, method, response.ok);
        if (message) {
          addActivityLog('action', message);
        }
        return response;
      } catch (error) {
        const message = actionMessage(path, method, false);
        if (message) {
          addActivityLog('action', message);
        }
        throw error;
      }
    };

    g.__dashboardFetchPatched = true;
  }, []);

  React.useEffect(() => {
    let unmounted = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    const notificationTimers: ReturnType<typeof setTimeout>[] = [];
    const recentInsert: Record<string, number> = {};

    const scheduleResolve = async (notificationId: string, networkId: string | null, attempt: number = 0) => {
      try {
        const res = await fetch(`${USER_SERVICE_BASE}/history?limit=100&offset=0`, { method: 'GET' });
        if (!res.ok) {
          throw new Error(`History fetch failed (${res.status})`);
        }

        const json = await res.json() as { history?: HistoryRecordLite[] };
        const records = Array.isArray(json.history) ? json.history : [];
        const target = networkId
          ? records.find((record) => record.network_id === networkId) ?? records[0]
          : records[0];

        if (!target) {
          if (attempt < 2 && !unmounted) {
            const timer = setTimeout(() => {
              void scheduleResolve(notificationId, networkId, attempt + 1);
            }, 8000);
            notificationTimers.push(timer);
          }
          return;
        }

        const prediction = target.prediction_results?.predictions?.[0];
        const status = getPredictionStatus(prediction ?? null);

        if (status === 'pending') {
          if (attempt < 2 && !unmounted) {
            const timer = setTimeout(() => {
              void scheduleResolve(notificationId, networkId, attempt + 1);
            }, 8000);
            notificationTimers.push(timer);
          }
          return;
        }

        const percent =
          status === 'unsafe'
            ? formatPercent(prediction?.probability_unsafe) || formatPercent(prediction?.confidence)
            : formatPercent(prediction?.probability_safe) || formatPercent(prediction?.confidence);

        if (status === 'unsafe') {
          const attackCat = prediction?.attack_cat && prediction.attack_cat !== 'Normal'
            ? ` | Attack Category: ${prediction.attack_cat}`
            : '';
          updateActivityLog(notificationId, {
            message: `UNSAFE${percent ? ` ${percent}` : ''}${attackCat}`,
            status: 'error',
          });
        } else {
          updateActivityLog(notificationId, {
            message: `SAFE${percent ? ` ${percent}` : ''}`,
            status: 'success',
          });
        }
      } catch {
        if (attempt < 2 && !unmounted) {
          const timer = setTimeout(() => {
            void scheduleResolve(notificationId, networkId, attempt + 1);
          }, 8000);
          notificationTimers.push(timer);
        }
      }
    };

    const connect = () => {
      if (unmounted) return;

      ws = new WebSocket(`${WS_BASE}/ws/view-data`);

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'ping') return;

          const networkId = typeof data?.network_id === 'string'
            ? data.network_id
            : typeof data?.data?.network_id === 'string'
              ? data.data.network_id
              : null;

          const dedupeKey = networkId || `event_${Date.now()}`;
          const lastTs = recentInsert[dedupeKey] || 0;
          if (Date.now() - lastTs <= 4000) {
            return;
          }

          recentInsert[dedupeKey] = Date.now();
          const notificationId = addActivityLog('action', 'Prediction pending...', 'pending');
          const timer = setTimeout(() => {
            void scheduleResolve(notificationId, networkId, 0);
          }, 8000);
          notificationTimers.push(timer);
        } catch {
          // Ignore malformed websocket payloads.
        }
      };

      ws.onclose = (event) => {
        if (unmounted) return;
        if (event.code !== 1000 && event.code !== 1001) {
          reconnectTimer = setTimeout(connect, 5000);
        }
      };
    };

    connect();

    return () => {
      unmounted = true;
      notificationTimers.forEach((timer) => clearTimeout(timer));
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        try {
          ws.close(1000, 'Dashboard unmounting');
        } catch {
          // Ignore close errors.
        }
      }
    };
  }, []);

  return (
    <AppTheme {...props} themeComponents={xThemeComponents}>
      <CssBaseline enableColorScheme />
      <Box sx={{ display: 'flex' }}>
        <SideMenu />
        <AppNavbar />
        {/* Main content */}
        <Box
          component="main"
          sx={(theme) => ({
            flexGrow: 1,
            backgroundColor: theme.vars
              ? `rgba(${theme.vars.palette.background.defaultChannel} / 1)`
              : alpha(theme.palette.background.default, 1),
            overflow: 'auto',
            overflowX: 'hidden',
            scrollbarGutter: 'stable',
          })}
        >
          <Stack
            spacing={2}
            sx={{
              alignItems: 'center',
              mx: 3,
              pb: 5,
              mt: { xs: 8, md: 0 },
            }}
          >
            <Header />
            <Outlet />
          </Stack>
        </Box>
      </Box>
    </AppTheme>
  );
}
