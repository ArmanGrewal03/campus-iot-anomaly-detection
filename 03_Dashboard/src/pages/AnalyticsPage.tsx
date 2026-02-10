import * as React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import { DataGrid, GridColDef, GridRenderCellParams } from '@mui/x-data-grid';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import WifiIcon from '@mui/icons-material/Wifi';
import WifiOffIcon from '@mui/icons-material/WifiOff';

const USER_SERVICE_BASE = 'http://127.0.0.1:8002'; // User Service
const WS_BASE = 'ws://127.0.0.1:8002'; // WebSocket base URL

interface HistoryRecord {
  id: number;
  network_id: string;
  timestamp: string;
  user_id: number | null;
  os: string | null;
  browser: string | null;
  location: {
    city?: string;
    country?: string;
    latitude?: number;
    longitude?: number;
  } | null;
  data: Record<string, unknown> | null;
  prediction_results: {
    prediction?: number;
    probability?: number;
    is_anomaly?: boolean;
  } | null;
}

export default function AnalyticsPage() {
  const [history, setHistory] = React.useState<HistoryRecord[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [totalRecords, setTotalRecords] = React.useState(0);
  const [wsConnected, setWsConnected] = React.useState(false);
  const wsReconnectAttemptsRef = React.useRef(0);
  const wsRef = React.useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollingIntervalRef = React.useRef<ReturnType<typeof setInterval> | null>(null);
  const [snackbar, setSnackbar] = React.useState<{ open: boolean; message: string; severity: 'success' | 'error' | 'info' | 'warning' }>({
    open: false,
    message: '',
    severity: 'success',
  });

  const fetchHistory = React.useCallback(async (limit: number = 100, offset: number = 0) => {
    setLoading(true);
    try {
      const res = await fetch(`${USER_SERVICE_BASE}/history?limit=${limit}&offset=${offset}`);
      const json = await res.json() as { 
        status?: string; 
        history?: HistoryRecord[]; 
        total?: number;
        detail?: string;
      };
      
      if (res.ok && json.status === 'success') {
        if (json.history && Array.isArray(json.history)) {
          setHistory(json.history);
          setTotalRecords(json.total || json.history.length);
        } else {
          setHistory([]);
          setTotalRecords(0);
        }
      } else {
        setHistory([]);
        setTotalRecords(0);
        if (json.detail) {
          setSnackbar({ open: true, message: `Failed to fetch history: ${json.detail}`, severity: 'error' });
        }
      }
    } catch (err) {
      console.error('Failed to fetch history:', err);
      setHistory([]);
      setTotalRecords(0);
      setSnackbar({ open: true, message: 'Failed to fetch history. Is the User Service running?', severity: 'error' });
    } finally {
      setLoading(false);
    }
  }, []);

  // Connect to WebSocket
  const connectWebSocket = React.useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    try {
      const ws = new WebSocket(`${WS_BASE}/ws/data-stream`);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setWsConnected(true);
        wsReconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', data);
          
          // Refresh history when new data arrives
          fetchHistory(100, 0);
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setWsConnected(false);
        wsRef.current = null;

        // Attempt to reconnect after a delay (exponential backoff, max 30 seconds)
        const delay = Math.min(1000 * Math.pow(2, wsReconnectAttemptsRef.current), 30000);
        wsReconnectAttemptsRef.current += 1;
        
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log(`Attempting to reconnect WebSocket (attempt ${wsReconnectAttemptsRef.current})...`);
          connectWebSocket();
        }, delay);
      };
    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setWsConnected(false);
    }
  }, [fetchHistory]);

  // Initialize: fetch history, connect WebSocket, and start polling
  React.useEffect(() => {
    fetchHistory(100, 0);
    connectWebSocket();

    // Poll for updated predictions every 30 seconds
    pollingIntervalRef.current = setInterval(() => {
      console.log('Polling for updated history/predictions...');
      fetchHistory(100, 0);
    }, 30000); // 30 seconds

    // Cleanup on unmount
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [fetchHistory, connectWebSocket]); // Include dependencies

  const columns: GridColDef[] = [
    { field: 'id', headerName: 'ID', width: 70 },
    { field: 'network_id', headerName: 'Network ID', width: 200 },
    { 
      field: 'timestamp', 
      headerName: 'Timestamp', 
      width: 180,
      valueFormatter: (params: { value: string | null | undefined }) => {
        if (!params.value) return '';
        try {
          return new Date(params.value).toLocaleString();
        } catch {
          return params.value;
        }
      }
    },
    { field: 'user_id', headerName: 'User ID', width: 100 },
    { field: 'os', headerName: 'OS', width: 120 },
    { field: 'browser', headerName: 'Browser', width: 120 },
    {
      field: 'location',
      headerName: 'Location',
      width: 200,
      valueFormatter: (params: { value: { city?: string; country?: string } | null | undefined }) => {
        if (!params.value) return 'N/A';
        const loc = params.value;
        if (loc.city && loc.country) {
          return `${loc.city}, ${loc.country}`;
        }
        return loc.city || loc.country || 'N/A';
      }
    },
    {
      field: 'prediction_results',
      headerName: 'Prediction',
      width: 150,
      renderCell: (params: GridRenderCellParams<HistoryRecord>) => {
        if (!params.value) {
          return <Chip label="Pending" color="default" size="small" />;
        }
        const pred = params.value as { is_anomaly?: boolean; prediction?: number; probability?: number };
        if (pred.is_anomaly) {
          return <Chip label="Anomaly" color="error" size="small" />;
        } else if (pred.is_anomaly === false) {
          return <Chip label="Normal" color="success" size="small" />;
        }
        return <Chip label="Unknown" color="default" size="small" />;
      }
    },
    {
      field: 'data',
      headerName: 'Data Preview',
      width: 200,
      valueFormatter: (params: { value: Record<string, unknown> | null | undefined }) => {
        if (!params.value) return 'N/A';
        const data = params.value;
        const keys = Object.keys(data);
        return keys.length > 0 ? `${keys.length} fields` : 'Empty';
      }
    },
  ];

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2.5 }}>
        <Stack>
          <Typography component="h1" variant="h5" sx={{ fontWeight: 600 }}>
            Analytics
          </Typography>
          <Typography color="text.secondary">
            Real-time network logs and anomaly detection results.
          </Typography>
        </Stack>
        <Stack direction="row" spacing={1} alignItems="center">
          <Chip
            icon={wsConnected ? <WifiIcon /> : <WifiOffIcon />}
            label={wsConnected ? 'Connected' : 'Disconnected'}
            color={wsConnected ? 'success' : 'default'}
            size="small"
          />
          <Button
            variant="outlined"
            startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <RefreshRoundedIcon />}
            onClick={() => fetchHistory(100, 0)}
            disabled={loading}
          >
            Refresh
          </Button>
        </Stack>
      </Stack>

      <Card variant="outlined">
        <CardContent>
          <Box sx={{ height: 600, width: '100%' }}>
            {loading && history.length === 0 ? (
              <Stack alignItems="center" justifyContent="center" sx={{ height: '100%' }}>
                <CircularProgress />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Loading history...
                </Typography>
              </Stack>
            ) : history.length === 0 ? (
              <Stack alignItems="center" justifyContent="center" sx={{ height: '100%' }}>
                <Typography variant="body2" color="text.secondary">
                  No history records found.
                </Typography>
              </Stack>
            ) : (
              <DataGrid
                rows={history}
                columns={columns}
                getRowId={(row) => row.id}
                initialState={{ pagination: { paginationModel: { pageSize: 25 } } }}
                pageSizeOptions={[10, 25, 50, 100]}
                disableRowSelectionOnClick
                loading={loading}
                rowCount={totalRecords}
              />
            )}
          </Box>
        </CardContent>
      </Card>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
          severity={snackbar.severity}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
