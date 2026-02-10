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
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import { DataGrid, GridColDef, GridRenderCellParams } from '@mui/x-data-grid';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import WifiIcon from '@mui/icons-material/Wifi';
import WifiOffIcon from '@mui/icons-material/WifiOff';

const USER_SERVICE_BASE = 'http://127.0.0.1:8002'; // User Service
const MODEL_API_BASE = 'http://127.0.0.1:8001'; // Model Service
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
    status?: string;
    predictions?: Array<{
      prediction?: number;
      label?: string;
      probability_safe?: number;
      probability_unsafe?: number;
      confidence?: number;
    }>;
    timestamp?: string;
  } | null;
  session_active_time: string | null;
  is_active: boolean;
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
  const [availableModels, setAvailableModels] = React.useState<Array<{ model_name: string; training_date?: string; n_features?: number; accuracy?: number }>>([]);
  const [selectedModel, setSelectedModel] = React.useState<string>('');
  const [modelsLoading, setModelsLoading] = React.useState(false);

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

  // Connect to WebSocket - single persistent connection
  const connectWebSocket = React.useCallback(() => {
    // Don't create a new connection if one already exists and is open or connecting
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
        return; // Already connected or connecting
      }
      // Clean up old connection if it's in a closing/closed state
      wsRef.current = null;
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

      ws.onclose = (event) => {
        console.log('WebSocket disconnected', event.code, event.reason);
        setWsConnected(false);
        wsRef.current = null;

        // Only attempt to reconnect if it was an unexpected close (not a normal closure)
        // Normal closure codes: 1000 (normal), 1001 (going away)
        // Don't reconnect if it was a normal close or if we're unmounting
        if (event.code !== 1000 && event.code !== 1001) {
          // Only reconnect once, with a longer delay
          if (wsReconnectAttemptsRef.current === 0) {
            wsReconnectAttemptsRef.current = 1;
            reconnectTimeoutRef.current = setTimeout(() => {
              console.log('Attempting to reconnect WebSocket...');
              connectWebSocket();
            }, 5000); // 5 second delay
          }
        }
      };
    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setWsConnected(false);
    }
  }, [fetchHistory]);

  // Fetch available models
  const fetchModels = React.useCallback(async () => {
    setModelsLoading(true);
    try {
      const res = await fetch(`${MODEL_API_BASE}/models`);
      const json = await res.json() as { 
        status?: string; 
        models?: Array<{ model_name: string; training_date?: string; n_features?: number; accuracy?: number }>; 
        total_models?: number;
        detail?: string;
      };
      
      if (res.ok && json.status === 'success') {
        if (json.models && Array.isArray(json.models) && json.models.length > 0) {
          setAvailableModels(json.models);
          // Set first model as default if none selected
          if (!selectedModel && json.models.length > 0) {
            setSelectedModel(json.models[0].model_name);
          }
        } else {
          setAvailableModels([]);
        }
      } else {
        setAvailableModels([]);
        if (json.detail) {
          console.warn('Failed to fetch models:', json.detail);
        }
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
      setAvailableModels([]);
    } finally {
      setModelsLoading(false);
    }
  }, [selectedModel]);

  // Initialize: fetch history and connect WebSocket (single persistent connection)
  React.useEffect(() => {
    fetchHistory(100, 0);
    fetchModels();
    connectWebSocket();

    // Poll for updated predictions every 30 seconds (as backup if websocket fails)
    pollingIntervalRef.current = setInterval(() => {
      // Only poll if websocket is not connected
      if (!wsConnected) {
        console.log('Polling for updated history/predictions (websocket disconnected)...');
        fetchHistory(100, 0);
      }
    }, 30000); // 30 seconds

    // Cleanup on unmount
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      if (wsRef.current) {
        // Close with normal closure code to prevent reconnection
        wsRef.current.close(1000, 'Component unmounting');
        wsRef.current = null;
      }
    };
  }, [fetchHistory, connectWebSocket, wsConnected, fetchModels]); // Include dependencies

  const columns: GridColDef[] = [
    { field: 'id', headerName: 'ID', width: 70 },
    { field: 'network_id', headerName: 'Network ID', width: 200 },
    { 
      field: 'timestamp', 
      headerName: 'Timestamp', 
      width: 180,
      valueFormatter: (params: { value: string | null | undefined } | null | undefined) => {
        if (!params || !params.value) return '';
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
      valueFormatter: (params: { value: { city?: string; country?: string } | null | undefined } | null | undefined) => {
        if (!params || !params.value) return 'N/A';
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
      width: 280,
      renderCell: (params: GridRenderCellParams<HistoryRecord>) => {
        if (!params.value) {
          return <Chip label="Pending" color="default" size="small" />;
        }
        const predResults = params.value as {
          status?: string;
          predictions?: Array<{
            prediction?: number;
            label?: string;
            probability_safe?: number;
            probability_unsafe?: number;
            confidence?: number;
          }>;
          timestamp?: string;
        };
        
        // Extract the first prediction from the predictions array
        const prediction = predResults.predictions && predResults.predictions.length > 0 
          ? predResults.predictions[0] 
          : null;
        
        if (!prediction) {
          return <Chip label="Unknown" color="default" size="small" />;
        }
        
        // prediction: 0 = safe, 1 = unsafe/anomaly
        const isAnomaly = prediction.prediction === 1;
        const label = prediction.label || (isAnomaly ? 'Anomaly' : 'Safe');
        const confidence = prediction.confidence !== undefined 
          ? (prediction.confidence * 100).toFixed(1) + '%' 
          : null;
        const probUnsafe = prediction.probability_unsafe !== undefined
          ? (prediction.probability_unsafe * 100).toFixed(1) + '%'
          : null;
        
        // Color code probability_unsafe: red if high (>50%), green if low (<=50%)
        const probUnsafeColor = prediction.probability_unsafe !== undefined
          ? (prediction.probability_unsafe > 0.5 ? 'error.main' : 'success.main')
          : 'text.secondary';
        
        return (
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Chip 
              label={label} 
              color={isAnomaly ? 'error' : 'success'} 
              size="small" 
            />
            {confidence && (
              <Typography variant="caption" color="text.secondary">
                ({confidence})
              </Typography>
            )}
            {probUnsafe && (
              <Typography variant="caption" sx={{ color: probUnsafeColor, fontWeight: 'medium' }}>
                Unsafe: {probUnsafe}
              </Typography>
            )}
          </Stack>
        );
      }
    },
    {
      field: 'data',
      headerName: 'Data Preview',
      width: 200,
      valueFormatter: (params: { value: Record<string, unknown> | null | undefined } | null | undefined) => {
        if (!params || !params.value) return 'N/A';
        const data = params.value;
        const keys = Object.keys(data);
        return keys.length > 0 ? `${keys.length} fields` : 'Empty';
      }
    },
    {
      field: 'session_active_time',
      headerName: 'Session Start',
      width: 180,
      valueFormatter: (params: { value: string | null | undefined } | null | undefined) => {
        if (!params || !params.value) return 'N/A';
        try {
          return new Date(params.value).toLocaleString();
        } catch {
          return params.value;
        }
      }
    },
    {
      field: 'is_active',
      headerName: 'Session Status',
      width: 140,
      renderCell: (params: GridRenderCellParams<HistoryRecord>) => {
        if (!params || params.value === null || params.value === undefined) {
          return <Chip label="Unknown" color="default" size="small" />;
        }
        const isActive = params.value as boolean;
        return (
          <Chip
            label={isActive ? 'Active' : 'Inactive'}
            color={isActive ? 'success' : 'default'}
            size="small"
          />
        );
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
        <Stack direction="row" spacing={2} alignItems="center">
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel id="model-select-label">Model</InputLabel>
            <Select
              labelId="model-select-label"
              id="model-select"
              value={selectedModel}
              label="Model"
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={modelsLoading || availableModels.length === 0}
            >
              {availableModels.map((model) => (
                <MenuItem key={model.model_name} value={model.model_name}>
                  {model.model_name}
                  {model.accuracy !== undefined && ` (${(model.accuracy * 100).toFixed(1)}%)`}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
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
