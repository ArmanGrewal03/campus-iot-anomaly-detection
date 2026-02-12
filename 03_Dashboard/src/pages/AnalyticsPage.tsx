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
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Sphere, useTexture } from '@react-three/drei';
import * as THREE from 'three';

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

// Helper function to convert lat/lon to 3D coordinates on a sphere
function latLonToXYZ(lat: number, lon: number, radius: number = 1): [number, number, number] {
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;
  const x = radius * Math.cos(latRad) * Math.cos(lonRad);
  const y = radius * Math.sin(latRad);
  const z = radius * Math.cos(latRad) * Math.sin(lonRad);
  return [x, y, z];
}

// 3D Globe Component
function Globe3D({ locations }: { locations: Array<{ record: HistoryRecord; lat: number; lon: number }> }) {
  const globeRef = React.useRef<THREE.Mesh>(null);
  const [hoveredId, setHoveredId] = React.useState<number | null>(null);
  
  // Load Earth texture map - using a world map texture
  // Using a reliable texture source from Three.js examples
  const earthTexture = useTexture('https://raw.githubusercontent.com/mrdoob/three.js/r129/examples/textures/planets/earth_atmos_2048.jpg');

  useFrame(() => {
    if (globeRef.current) {
      globeRef.current.rotation.y += 0.001;
    }
  });

  return (
    <>
      {/* Globe sphere with Earth texture */}
      <Sphere ref={globeRef} args={[1, 64, 64]}>
        <meshStandardMaterial
          map={earthTexture}
          roughness={0.8}
          metalness={0.2}
        />
      </Sphere>
      
      {/* Grid lines for better visualization */}
      <Sphere args={[1.01, 32, 32]}>
        <meshBasicMaterial
          color="#3f51b5"
          wireframe
          transparent
          opacity={0.1}
        />
      </Sphere>

      {/* Session markers */}
      {locations.map(({ record, lat, lon }) => {
        const [x, y, z] = latLonToXYZ(lat, lon, 1.02);
        const predResults = record.prediction_results;
        const prediction =
          predResults?.predictions && predResults.predictions.length > 0
            ? predResults.predictions[0]
            : null;
        const isAnomaly = prediction?.prediction === 1;
        const status = prediction
          ? isAnomaly
            ? 'Anomaly'
            : 'Safe'
          : 'Pending';
        
        const color = status === 'Anomaly' ? '#d32f2f' : status === 'Safe' ? '#2e7d32' : '#757575';

        const isHovered = hoveredId === record.id;

        return (
          <React.Fragment key={record.id}>
            <mesh 
              position={[x, y, z]}
              onPointerEnter={() => setHoveredId(record.id)}
              onPointerLeave={() => setHoveredId(null)}
            >
              <sphereGeometry args={[0.02, 16, 16]} />
              <meshStandardMaterial 
                color={color} 
                emissive={color} 
                emissiveIntensity={isHovered ? 1 : 0.5}
              />
            </mesh>
            {/* Line connecting marker to globe surface */}
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array([x, y, z, x * 0.98, y * 0.98, z * 0.98])}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color={color} transparent opacity={isHovered ? 0.8 : 0.5} />
            </line>
            {/* HTML tooltip - only show when hovered */}
            {isHovered && (
              <Html position={[x * 1.1, y * 1.1, z * 1.1]} distanceFactor={5} center>
                <Box
                  sx={{
                    bgcolor: 'rgba(255, 255, 255, 0.95)',
                    color: 'text.primary',
                    border: `1px solid ${color}`,
                    borderRadius: 0.5,
                    p: 0.75,
                    maxWidth: 140,
                    minWidth: 100,
                    boxShadow: 3,
                    pointerEvents: 'auto',
                  }}
                >
                  <Stack spacing={0.5}>
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        fontWeight: 600, 
                        fontSize: '0.75rem', 
                        lineHeight: 1.2,
                        color: 'text.primary',
                        display: 'block',
                      }}
                    >
                      Session #{record.id}
                    </Typography>
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        fontSize: '0.7rem', 
                        lineHeight: 1.3,
                        color: 'text.secondary',
                        display: 'block',
                      }}
                    >
                      {record.location?.city || 'Unknown'}
                      {record.location?.country && `, ${record.location.country}`}
                    </Typography>
                    <Chip
                      label={status}
                      color={isAnomaly ? 'error' : status === 'Safe' ? 'success' : 'default'}
                      size="small"
                      sx={{ 
                        height: 18, 
                        fontSize: '0.65rem',
                        '& .MuiChip-label': { px: 0.75, py: 0.25 },
                        width: 'fit-content',
                      }}
                    />
                  </Stack>
                </Box>
              </Html>
            )}
          </React.Fragment>
        );
      })}

      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
    </>
  );
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
  const [mapView, setMapView] = React.useState<'2d' | '3d'>('3d');

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

  // Set the selected model in the user service
  const setModelInBackend = React.useCallback(async (modelName: string) => {
    try {
      const res = await fetch(`${USER_SERVICE_BASE}/set-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: modelName }),
      });
      if (res.ok) {
        const json = await res.json();
        console.log(`Model set to ${modelName}:`, json);
      } else {
        console.warn(`Failed to set model to ${modelName}:`, res.statusText);
      }
    } catch (err) {
      console.error(`Error setting model to ${modelName}:`, err);
    }
  }, []);

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
            const firstModel = json.models[0].model_name;
            setSelectedModel(firstModel);
            await setModelInBackend(firstModel);
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
  }, [selectedModel, setModelInBackend]);

  // Sync selected model to backend when it changes
  React.useEffect(() => {
    if (selectedModel) {
      setModelInBackend(selectedModel);
    }
  }, [selectedModel, setModelInBackend]);

  // Initialize: fetch history and connect WebSocket (single persistent connection)
  React.useEffect(() => {
    fetchHistory(100, 0);
    fetchModels();
    connectWebSocket();

    // Poll for updated predictions every 120 seconds (as backup if websocket fails)
    pollingIntervalRef.current = setInterval(() => {
      // Only poll if websocket is not connected
      if (!wsConnected) {
        console.log('Polling for updated history/predictions (websocket disconnected)...');
        fetchHistory(100, 0);
      }
    }, 120000); // 120 seconds (2 minutes)

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
    { 
      field: 'id', 
      headerName: 'ID', 
      width: 70,
      sortable: true,
      filterable: true,
    },
    { 
      field: 'network_id', 
      headerName: 'Network ID', 
      width: 200,
      sortable: true,
      filterable: true,
    },
    { 
      field: 'timestamp', 
      headerName: 'Timestamp', 
      width: 180,
      sortable: true,
      filterable: true,
      type: 'dateTime',
      valueFormatter: (params: { value: string | null | undefined } | null | undefined) => {
        if (!params || !params.value) return '';
        try {
          return new Date(params.value).toLocaleString();
        } catch {
          return params.value;
        }
      }
    },
    { 
      field: 'user_id', 
      headerName: 'User ID', 
      width: 100,
      sortable: true,
      filterable: true,
      type: 'number',
    },
    { 
      field: 'os', 
      headerName: 'OS', 
      width: 120,
      sortable: true,
      filterable: true,
    },
    { 
      field: 'browser', 
      headerName: 'Browser', 
      width: 120,
      sortable: true,
      filterable: true,
    },
    {
      field: 'location',
      headerName: 'Location',
      width: 200,
      sortable: true,
      filterable: true,
      valueGetter: (value: any, row: HistoryRecord) => {
        const loc = row.location;
        if (!loc) return 'N/A';
        if (loc.city && loc.country) {
          return `${loc.city}, ${loc.country}`;
        }
        return loc.city || loc.country || 'N/A';
      },
      valueFormatter: (params: { value: string | null | undefined } | null | undefined) => {
        if (!params || !params.value) return 'N/A';
        return params.value;
      },
    },
    {
      field: 'prediction_results',
      headerName: 'Prediction',
      width: 280,
      sortable: true,
      filterable: true,
      valueGetter: (value: any, row: HistoryRecord) => {
        const predResults = row.prediction_results;
        if (!predResults || !predResults.predictions || predResults.predictions.length === 0) return 'Pending';
        const prediction = predResults.predictions[0];
        return prediction.label || (prediction.prediction === 1 ? 'Anomaly' : 'Safe');
      },
      renderCell: (params: GridRenderCellParams<HistoryRecord>) => {
        // Always read the full prediction object from the row to avoid conflicts with valueGetter
        const predResults = (params.row as HistoryRecord).prediction_results as {
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
          // If there is no prediction yet, show Pending instead of Unknown
          return <Chip label="Pending" color="default" size="small" />;
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
      sortable: false,
      filterable: false,
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
      sortable: true,
      filterable: true,
      type: 'dateTime',
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
      sortable: true,
      filterable: true,
      type: 'boolean',
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
              onChange={(e) => {
                const newModel = e.target.value;
                setSelectedModel(newModel);
                setModelInBackend(newModel);
              }}
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
                initialState={{ 
                  pagination: { paginationModel: { pageSize: 25 } },
                  sorting: {
                    sortModel: [{ field: 'timestamp', sort: 'desc' }],
                  },
                }}
                pageSizeOptions={[10, 25, 50, 100]}
                disableRowSelectionOnClick
                loading={loading}
                rowCount={totalRecords}
                filterMode="client"
                slotProps={{
                  filterPanel: {
                    filterFormProps: {
                      logicOperatorInputProps: {
                        variant: 'outlined',
                        size: 'small',
                      },
                      columnInputProps: {
                        variant: 'outlined',
                        size: 'small',
                      },
                      operatorInputProps: {
                        variant: 'outlined',
                        size: 'small',
                      },
                      valueInputProps: {
                        variant: 'outlined',
                        size: 'small',
                      },
                    },
                  },
                }}
                disableColumnFilter={false}
                disableColumnMenu={false}
                columnVisibilityModel={{
                  // All columns visible by default
                }}
              />
            )}
          </Box>
        </CardContent>
      </Card>

      {/* 3D Globe Card */}
      <Card variant="outlined" sx={{ mt: 2 }}>
        <CardContent>
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              3D Session Locations Globe
            </Typography>
            <Stack direction="row" spacing={1}>
              <Button
                size="small"
                variant={mapView === '3d' ? 'contained' : 'outlined'}
                onClick={() => setMapView('3d')}
              >
                3D Globe
              </Button>
              <Button
                size="small"
                variant={mapView === '2d' ? 'contained' : 'outlined'}
                onClick={() => setMapView('2d')}
              >
                2D Map
              </Button>
            </Stack>
          </Stack>
          <Box sx={{ height: 500, width: '100%', position: 'relative', bgcolor: '#000', borderRadius: 1 }}>
            {(() => {
              // Extract locations with valid coordinates
              const locationsWithCoords = history
                .filter(
                  (record) =>
                    record.location &&
                    typeof record.location.latitude === 'number' &&
                    typeof record.location.longitude === 'number' &&
                    !isNaN(record.location.latitude) &&
                    !isNaN(record.location.longitude)
                )
                .map((record) => ({
                  record,
                  lat: record.location!.latitude!,
                  lon: record.location!.longitude!,
                }));

              if (locationsWithCoords.length === 0) {
                return (
                  <Stack alignItems="center" justifyContent="center" sx={{ height: '100%' }}>
                    <Typography variant="body2" color="text.secondary">
                      No location data available. Sessions will appear on the globe once they include coordinates.
                    </Typography>
                  </Stack>
                );
              }

              if (mapView === '3d') {
                return (
                  <Canvas camera={{ position: [0, 0, 3], fov: 50 }}>
                    <Globe3D locations={locationsWithCoords} />
                    <OrbitControls
                      enableZoom={true}
                      enablePan={false}
                      minDistance={2}
                      maxDistance={5}
                      autoRotate={false}
                      rotateSpeed={0.5}
                    />
                  </Canvas>
                );
              } else {
                // 2D map fallback (simplified)
                const avgLat = locationsWithCoords.reduce((sum, l) => sum + l.lat, 0) / locationsWithCoords.length;
                const avgLon = locationsWithCoords.reduce((sum, l) => sum + l.lon, 0) / locationsWithCoords.length;
                
                return (
                  <Box sx={{ height: '100%', width: '100%' }}>
                    <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>
                      2D map view - Switch to 3D Globe for interactive visualization
                    </Typography>
                    <Stack spacing={1} sx={{ p: 2 }}>
                      {locationsWithCoords.slice(0, 10).map(({ record, lat, lon }) => {
                        const predResults = record.prediction_results;
                        const prediction =
                          predResults?.predictions && predResults.predictions.length > 0
                            ? predResults.predictions[0]
                            : null;
                        const isAnomaly = prediction?.prediction === 1;
                        const status = prediction
                          ? isAnomaly
                            ? 'Anomaly'
                            : 'Safe'
                          : 'Pending';
                        
                        return (
                          <Box key={record.id} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Chip
                              label={status}
                              color={isAnomaly ? 'error' : status === 'Safe' ? 'success' : 'default'}
                              size="small"
                            />
                            <Typography variant="body2">
                              {record.location?.city}, {record.location?.country} ({lat.toFixed(2)}, {lon.toFixed(2)})
                            </Typography>
                          </Box>
                        );
                      })}
                      {locationsWithCoords.length > 10 && (
                        <Typography variant="caption" color="text.secondary">
                          ... and {locationsWithCoords.length - 10} more locations
                        </Typography>
                      )}
                    </Stack>
                  </Box>
                );
              }
            })()}
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
