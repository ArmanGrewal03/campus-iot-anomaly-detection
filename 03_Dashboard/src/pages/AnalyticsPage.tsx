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
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import IconButton from '@mui/material/IconButton';
import { DataGrid, GridColDef, GridRenderCellParams } from '@mui/x-data-grid';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import WifiIcon from '@mui/icons-material/Wifi';
import WifiOffIcon from '@mui/icons-material/WifiOff';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CloseIcon from '@mui/icons-material/Close';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Sphere, useTexture } from '@react-three/drei';
import * as THREE from 'three';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { ErrorBoundary } from '../App';

// Direct connections to backend services (bypass gateway for Analytics page)
const USER_SERVICE_BASE = 'http://127.0.0.1:8002'; // User Service - direct connection
const MODEL_API_BASE = 'http://127.0.0.1:8001'; // Model Service - direct connection
const DATA_INGESTION_SERVICE_BASE = 'http://127.0.0.1:8000'; // Data Ingestion Service - direct connection
const WS_BASE = 'ws://127.0.0.1:8002'; // WebSocket - direct connection

interface HistoryRecord {
  id: number;
  network_id: string;
  timestamp: string;
  utc_timestamp?: string;  // Explicit UTC timestamp
  session_start_time?: string;  // Session start time
  user_id: number | null;
  os: string | null;
  browser: string | null;
  location: {
    city?: string;
    country?: string;
    name?: string;
    latitude?: number;
    longitude?: number;
    ssh?: boolean;
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
      attack_cat?: string | null;
      attack_cat_probabilities?: Record<string, number>;
    }>;
    timestamp?: string;
    model_name?: string;
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

// Component to fit map bounds to show all markers
function FitBounds({ locations }: { locations: Array<{ lat: number; lon: number }> }) {
  const map = useMap();
  
  React.useEffect(() => {
    if (locations.length > 0) {
      const bounds = locations.reduce(
        (acc, { lat, lon }) => {
          return acc.extend([lat, lon]);
        },
        new L.LatLngBounds([locations[0].lat, locations[0].lon], [locations[0].lat, locations[0].lon])
      );
      map.fitBounds(bounds, { padding: [20, 20], maxZoom: 17 });
    }
  }, [locations, map]);
  
  return null;
}

// 3D Globe Component
function Globe3D({ locations }: { locations: Array<{ record: HistoryRecord; lat: number; lon: number }> }) {
  const groupRef = React.useRef<THREE.Group>(null);
  const [hoveredId, setHoveredId] = React.useState<number | null>(null);
  
  // Load Earth texture map - using a texture with lighter/bluer oceans
  // Using a texture with better ocean contrast against dark background
  const earthTexture = useTexture('https://raw.githubusercontent.com/mrdoob/three.js/r129/examples/textures/planets/earth_atmos_2048.jpg');

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.001;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Globe sphere with Earth texture */}
      <Sphere args={[1, 64, 64]}>
        <meshStandardMaterial
          map={earthTexture}
          roughness={0.6}
          metalness={0.05}
          color="#e8f4f8"
          emissive="#002244"
          emissiveIntensity={0.15}
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
                      {/* Prefer TMU building name if available, or show SSH coordinates */}
                      {record.location?.ssh
                        ? `SSH: ${record.location.latitude?.toFixed(4) || 'N/A'}, ${record.location.longitude?.toFixed(4) || 'N/A'}`
                        : (record.location?.name ||
                          (record.location?.city && record.location?.country
                            ? `${record.location.city}, ${record.location.country}`
                            : record.location?.city || record.location?.country || 'Unknown'))}
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
    </group>
  );
}

export default function AnalyticsPage() {
  const [history, setHistory] = React.useState<HistoryRecord[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [totalRecords, setTotalRecords] = React.useState(0);
  const [paginationModel, setPaginationModel] = React.useState({ page: 0, pageSize: 25 });
  const paginationModelRef = React.useRef(paginationModel);
  const fetchHistoryRef = React.useRef<typeof fetchHistory | null>(null);
  const [wsConnected, setWsConnected] = React.useState(false);
  const [dataModalOpen, setDataModalOpen] = React.useState(false);
  const [selectedRowData, setSelectedRowData] = React.useState<HistoryRecord | null>(null);
  const wsReconnectAttemptsRef = React.useRef(0);
  const wsRef = React.useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollingIntervalRef = React.useRef<ReturnType<typeof setInterval> | null>(null);
  const lastWebSocketFetchRef = React.useRef(0);
  const [snackbar, setSnackbar] = React.useState<{ open: boolean; message: string; severity: 'success' | 'error' | 'info' | 'warning' }>({
    open: false,
    message: '',
    severity: 'success',
  });
  const [availableModels, setAvailableModels] = React.useState<Array<{ model_name: string; training_date?: string; n_features?: number; accuracy?: number }>>([]);
  const [selectedModel, setSelectedModel] = React.useState<string>('');
  const [modelsLoading, setModelsLoading] = React.useState(false);
  const [availableDatasets, setAvailableDatasets] = React.useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = React.useState<string>('');
  const [datasetsLoading, setDatasetsLoading] = React.useState(false);
  const [mapView, setMapView] = React.useState<'2d' | '3d'>('3d');
  const [filterActive, setFilterActive] = React.useState<boolean | 'all'>('all');
  const [filterPrediction, setFilterPrediction] = React.useState<'all' | 'safe' | 'anomaly' | 'pending'>('all');
  const [clearDialogOpen, setClearDialogOpen] = React.useState(false);
  const [clearing, setClearing] = React.useState(false);

  const fetchHistory = React.useCallback(async (limit: number = 25, offset: number = 0) => {
    setLoading(true);
    try {
      // Analytics endpoints are automatically no-cache by the gateway
      const res = await fetch(`${USER_SERVICE_BASE}/history?limit=${limit}&offset=${offset}`, {
        method: 'GET',
      });
      
      // Check if response is ok before trying to parse JSON
      if (!res.ok) {
        let errorMessage = `HTTP ${res.status}: ${res.statusText}`;
        try {
          const errorJson = await res.json() as { detail?: string | { message?: string; error?: string } };
          if (errorJson.detail) {
            if (typeof errorJson.detail === 'string') {
              errorMessage = errorJson.detail;
            } else if (errorJson.detail.message) {
              errorMessage = errorJson.detail.message;
            } else if (errorJson.detail.error) {
              errorMessage = errorJson.detail.error;
            }
          }
        } catch (parseErr) {
          // If we can't parse the error, use the status text
          errorMessage = `HTTP ${res.status}: ${res.statusText}`;
        }
        setHistory([]);
        setTotalRecords(0);
        setSnackbar({ open: true, message: `Failed to fetch history: ${errorMessage}`, severity: 'error' });
        return;
      }
      
      const json = await res.json() as { 
        status?: string; 
        history?: HistoryRecord[]; 
        total_records?: number;
        total?: number;
        detail?: string;
      };
      
      if (res.ok && json.status === 'success') {
        if (json.history && Array.isArray(json.history)) {
          setHistory(json.history);
          setTotalRecords(json.total_records || json.total || json.history.length);
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
      
      // Provide more specific error messages
      let errorMessage = 'Failed to fetch history. ';
      if (err instanceof TypeError && err.message.includes('fetch')) {
        errorMessage += 'Network error - is the API Gateway running?';
      } else if (err instanceof Error) {
        errorMessage += err.message;
      } else {
        errorMessage += 'Is the User Service running?';
      }
      
      setSnackbar({ open: true, message: errorMessage, severity: 'error' });
    } finally {
      setLoading(false);
    }
  }, []);

  const clearHistory = React.useCallback(async () => {
    setClearing(true);
    try {
      const res = await fetch(`${USER_SERVICE_BASE}/history`, {
        method: 'DELETE',
      });
      
      if (!res.ok) {
        let errorMessage = `HTTP ${res.status}: ${res.statusText}`;
        try {
          const errorJson = await res.json() as { detail?: string };
          if (errorJson.detail) {
            errorMessage = errorJson.detail;
          }
        } catch (parseErr) {
          // If we can't parse the error, use the status text
          errorMessage = `HTTP ${res.status}: ${res.statusText}`;
        }
        setSnackbar({ open: true, message: `Failed to clear history: ${errorMessage}`, severity: 'error' });
        return;
      }
      
      const json = await res.json() as { 
        status?: string; 
        message?: string;
        records_deleted?: number;
      };
      
      if (res.ok && json.status === 'success') {
        const deletedCount = json.records_deleted || 0;
        setHistory([]);
        setTotalRecords(0);
        setSnackbar({ 
          open: true, 
          message: `Successfully cleared ${deletedCount} log${deletedCount !== 1 ? 's' : ''}`, 
          severity: 'success' 
        });
        // Refresh to show empty state
        const offset = paginationModel.page * paginationModel.pageSize;
        fetchHistory(paginationModel.pageSize, offset);
      } else {
        setSnackbar({ open: true, message: 'Failed to clear history', severity: 'error' });
      }
    } catch (err) {
      console.error('Failed to clear history:', err);
      let errorMessage = 'Failed to clear history. ';
      if (err instanceof TypeError && err.message.includes('fetch')) {
        errorMessage += 'Network error - is the User Service running?';
      } else if (err instanceof Error) {
        errorMessage += err.message;
      }
      setSnackbar({ open: true, message: errorMessage, severity: 'error' });
    } finally {
      setClearing(false);
      setClearDialogOpen(false);
    }
  }, [fetchHistory, paginationModel]);

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
      const ws = new WebSocket(`${WS_BASE}/ws/view-data`);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setWsConnected(true);
        wsReconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Ignore ping messages
          if (data.type === 'ping') {
            return;
          }
          
          console.log('WebSocket message received:', data);
          
          // Refresh history when new data arrives (keep current pagination)
          // But debounce to prevent rapid successive fetches (max once per second)
          const now = Date.now();
          const FETCH_DEBOUNCE_MS = 1000;
          if (now - lastWebSocketFetchRef.current >= FETCH_DEBOUNCE_MS) {
            lastWebSocketFetchRef.current = now;
            const currentPagination = paginationModelRef.current;
            const offset = currentPagination.page * currentPagination.pageSize;
            if (fetchHistoryRef.current) {
              fetchHistoryRef.current(currentPagination.pageSize, offset);
            }
          }
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
  }, []); // Empty deps - fetchHistory is stable via useCallback

  // Set the selected model in the user service
  const setModelInBackend = React.useCallback(async (modelName: string) => {
    try {
      // Analytics endpoints are automatically no-cache by the gateway
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

  // Set the selected dataset in the user service
  const setDatasetInBackend = React.useCallback(async (datasetName: string) => {
    try {
      const res = await fetch(`${USER_SERVICE_BASE}/set-dataset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dataset_name: datasetName }),
      });
      if (res.ok) {
        const json = await res.json();
        console.log(`Dataset set to ${datasetName}:`, json);
      } else {
        console.warn(`Failed to set dataset to ${datasetName}:`, res.statusText);
      }
    } catch (err) {
      console.error(`Error setting dataset to ${datasetName}:`, err);
    }
  }, []);

  // Fetch available datasets from Data Ingestion Service
  const fetchDatasets = React.useCallback(async () => {
    setDatasetsLoading(true);
    try {
      const res = await fetch(`${DATA_INGESTION_SERVICE_BASE}/tables`, {
        method: 'GET',
      });
      const json = (await res.json()) as {
        status?: string;
        tables?: string[];
        detail?: string;
      };

      if (res.ok && json.status === 'success' && json.tables) {
        // Extract dataset names from table names (format: csv_data_{dataset_name})
        const datasetNames = json.tables
          .filter((table) => table.startsWith('csv_data_'))
          .map((table) => table.replace(/^csv_data_/, ''));

        setAvailableDatasets(datasetNames);

        if (datasetNames.length === 0) {
          setSelectedDataset('');
          setDatasetsLoading(false);
          return;
        }

        // Use functional update to check current state
        setSelectedDataset((prev) => {
          // If current selection still exists, keep it
          if (prev && datasetNames.includes(prev)) {
            return prev;
          }
          
          // Otherwise, try to get from backend or use first available
          // We'll do this in a separate async call to avoid issues
          return prev || datasetNames[0];
        });

        // Try to get current dataset from backend
        try {
          const getRes = await fetch(`${USER_SERVICE_BASE}/get-dataset`, {
            method: 'GET',
          });
          const getJson = (await getRes.json()) as {
            status?: string;
            dataset_name?: string;
            detail?: string;
          };

          if (getRes.ok && getJson.status === 'success' && getJson.dataset_name) {
            // Use backend's current dataset if it exists in available datasets
            if (datasetNames.includes(getJson.dataset_name)) {
              setSelectedDataset(getJson.dataset_name);
            } else {
              // Backend dataset not in available list, use first available
              setSelectedDataset(datasetNames[0]);
            }
          } else {
            // No dataset from backend, use first available if none selected
            setSelectedDataset((prev) => prev || datasetNames[0]);
          }
        } catch (e) {
          console.warn('Failed to get current dataset from backend:', e);
          // Fallback to first available dataset if none selected
          setSelectedDataset((prev) => prev || datasetNames[0]);
        }
      } else {
        setAvailableDatasets([]);
        setSelectedDataset('');
        if (json.detail) {
          console.warn('Failed to fetch datasets:', json.detail);
        }
      }
    } catch (err) {
      console.error('Failed to fetch datasets:', err);
      setAvailableDatasets([]);
      setSelectedDataset('');
    } finally {
      setDatasetsLoading(false);
    }
  }, []); // Empty deps - only fetch once on mount

  // Use a ref to track if we've initialized the model to prevent loops
  const modelInitializedRef = React.useRef(false);
  const selectedModelRef = React.useRef<string>('');

  // Update ref when selectedModel changes
  React.useEffect(() => {
    selectedModelRef.current = selectedModel;
  }, [selectedModel]);

  // Fetch available models and set a sensible default if none is selected
  const fetchModels = React.useCallback(async () => {
    setModelsLoading(true);
    try {
      // Analytics endpoints are automatically no-cache by the gateway
      const res = await fetch(`${MODEL_API_BASE}/models`, {
        method: 'GET',
      });
      const json = (await res.json()) as {
        status?: string;
        models?: Array<{
          model_name: string;
          training_date?: string;
          n_features?: number;
          accuracy?: number;
        }>;
        total_models?: number;
        detail?: string;
      };

      if (res.ok && json.status === 'success') {
        if (json.models && Array.isArray(json.models) && json.models.length > 0) {
          setAvailableModels(json.models);

          // If no model is selected yet, try to use the backend's current model,
          // falling back to the first available model.
          // Only set initial model once to prevent loops
          if (!selectedModelRef.current && !modelInitializedRef.current) {
            let initialModel: string | null = null;

            try {
              // Analytics endpoints are automatically no-cache by the gateway
              const getRes = await fetch(`${USER_SERVICE_BASE}/get-model`, {
                method: 'GET',
              });
              const getJson = (await getRes.json()) as {
                status?: string;
                model_name?: string;
                detail?: string;
              };

              if (getRes.ok && getJson.status === 'success' && getJson.model_name) {
                initialModel = getJson.model_name;
              }
            } catch (e) {
              console.warn('Failed to get current model from backend:', e);
            }

            // If backend didn't return a model or it's missing, use first in list.
            if (!initialModel) {
              initialModel = json.models[0].model_name;
            }

            if (initialModel) {
              modelInitializedRef.current = true;
              setSelectedModel(initialModel);
            }
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
  }, []); // Remove selectedModel dependency to prevent loops

  // Sync selected model to backend when it changes
  React.useEffect(() => {
    if (selectedModel) {
      setModelInBackend(selectedModel);
    }
  }, [selectedModel, setModelInBackend]);

  // Sync selected dataset to backend when it changes
  React.useEffect(() => {
    if (selectedDataset) {
      setDatasetInBackend(selectedDataset);
    }
  }, [selectedDataset, setDatasetInBackend]);

  // Update refs when they change
  React.useEffect(() => {
    paginationModelRef.current = paginationModel;
  }, [paginationModel]);
  
  React.useEffect(() => {
    fetchHistoryRef.current = fetchHistory;
  }, [fetchHistory]);

  // Fetch history when pagination changes
  React.useEffect(() => {
    const offset = paginationModel.page * paginationModel.pageSize;
    fetchHistory(paginationModel.pageSize, offset);
  }, [paginationModel, fetchHistory]);

  // Initialize: fetch models and connect WebSocket (single persistent connection)
  // This effect should only run once on mount
  React.useEffect(() => {
    let isMounted = true;
    
    const initialize = async () => {
      await fetchModels();
      await fetchDatasets();
      if (isMounted) {
        connectWebSocket();
      }
    };
    
    initialize();

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
      isMounted = false;
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
        try {
          wsRef.current.close(1000, 'Component unmounting');
        } catch (e) {
          // Ignore errors when closing
        }
        wsRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty dependency array - only run once on mount

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
      width: 200,
      sortable: true,
      filterable: true,
      renderCell: (params: GridRenderCellParams<HistoryRecord>) => {
        const value = params.value as string | null | undefined;
        if (!value) return <Typography variant="body2" color="text.secondary">N/A</Typography>;
        try {
          const date = typeof value === 'string' ? new Date(value) : value;
          return <Typography variant="body2">{date.toLocaleString('en-US')}</Typography>;
        } catch {
          return <Typography variant="body2">{String(value)}</Typography>;
        }
      }
    },
    {
      field: 'utc_timestamp',
      headerName: 'UTC Timestamp',
      width: 200,
      sortable: true,
      filterable: true,
      valueGetter: (value: any, row: HistoryRecord) => {
        return row.utc_timestamp || row.timestamp || null;
      },
      renderCell: (params: GridRenderCellParams<HistoryRecord>) => {
        // valueGetter returns the value, so params.value contains the result
        const value = params.value as string | null | undefined;
        if (!value) return <Typography variant="body2" color="text.secondary">N/A</Typography>;
        try {
          const date = typeof value === 'string' ? new Date(value) : value;
          return <Typography variant="body2">{date.toLocaleString('en-US', { timeZone: 'UTC' })}</Typography>;
        } catch {
          return <Typography variant="body2">{String(value)}</Typography>;
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
        // Check if SSH location
        if (loc.ssh) {
          return `SSH: ${loc.latitude?.toFixed(4)}, ${loc.longitude?.toFixed(4)}`;
        }
        // Prefer name if available (TMU campus locations)
        if (loc.name) {
          return loc.name;
        }
        if (loc.city && loc.country) {
          return `${loc.city}, ${loc.country}`;
        }
        return loc.city || loc.country || 'N/A';
      },
      renderCell: (params: GridRenderCellParams<HistoryRecord>) => {
        const loc = params.row.location;
        if (!loc) return <Typography variant="body2" color="text.secondary">N/A</Typography>;
        // Check if SSH location
        if (loc.ssh) {
          return (
            <Stack direction="row" spacing={0.5} alignItems="center">
              <Chip label="SSH" size="small" color="warning" sx={{ height: 20, fontSize: '0.7rem' }} />
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                {loc.latitude?.toFixed(4)}, {loc.longitude?.toFixed(4)}
              </Typography>
            </Stack>
          );
        }
        // Prefer name if available (TMU campus locations)
        if (loc.name) {
          return <Typography variant="body2">{loc.name}</Typography>;
        }
        const displayText = loc.city && loc.country 
          ? `${loc.city}, ${loc.country}` 
          : (loc.city || loc.country || 'N/A');
        return <Typography variant="body2">{displayText}</Typography>;
      },
    },
    {
      field: 'prediction_results',
      headerName: 'Prediction',
      width: 200,
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
            attack_cat?: string | null;
            attack_cat_probabilities?: Record<string, number>;
          }>;
          timestamp?: string;
          model_name?: string;
        } | null;
        
        // Check if predResults exists and has predictions
        if (!predResults || !predResults.predictions || predResults.predictions.length === 0) {
          return <Chip label="Pending" color="default" size="small" />;
        }
        
        // Extract the first prediction from the predictions array
        const prediction = predResults.predictions[0];
        
        // prediction: 0 = safe, 1 = unsafe/anomaly
        const isAnomaly = prediction.prediction === 1;
        const label = prediction.label || (isAnomaly ? 'Anomaly' : 'Safe');
        const confidence = prediction.confidence !== undefined 
          ? (prediction.confidence * 100).toFixed(1) + '%' 
          : null;
        const probUnsafe = prediction.probability_unsafe !== undefined
          ? (prediction.probability_unsafe * 100).toFixed(1) + '%'
          : null;

        // Model name used for this prediction (added by backend)
        const modelName = predResults.model_name;
        
        // Attack category (only available for RFv1 models and unsafe predictions)
        const attackCat = prediction.attack_cat;
        
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
            {modelName && (
              <Typography variant="caption" color="text.secondary">
                Model: {modelName}
              </Typography>
            )}
          </Stack>
        );
      }
    },
    {
      field: 'attack_cat',
      headerName: 'Attack Category',
      width: 180,
      sortable: true,
      filterable: true,
      valueGetter: (value: any, row: HistoryRecord) => {
        const predResults = row.prediction_results;
        if (!predResults || !predResults.predictions || predResults.predictions.length === 0) return null;
        const prediction = predResults.predictions[0];
        return prediction.attack_cat || null;
      },
      renderCell: (params: GridRenderCellParams<HistoryRecord>) => {
        const predResults = (params.row as HistoryRecord).prediction_results;
        if (!predResults || !predResults.predictions || predResults.predictions.length === 0) {
          return <Typography variant="body2" color="text.secondary">-</Typography>;
        }
        
        const prediction = predResults.predictions[0];
        const attackCat = prediction.attack_cat;
        
        if (!attackCat || attackCat === 'Normal' || attackCat === null) {
          return <Typography variant="body2" color="text.secondary">-</Typography>;
        }
        
        return (
          <Chip
            label={attackCat}
            color="error"
            variant="outlined"
            size="small"
          />
        );
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
    {
      field: 'actions',
      headerName: 'Data',
      width: 100,
      sortable: false,
      filterable: false,
      renderCell: (params: GridRenderCellParams<HistoryRecord>) => {
        return (
          <IconButton
            size="small"
            onClick={() => {
              setSelectedRowData(params.row as HistoryRecord);
              setDataModalOpen(true);
            }}
            color="primary"
          >
            <VisibilityIcon fontSize="small" />
          </IconButton>
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
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel id="dataset-select-label">Dataset</InputLabel>
            <Select
              labelId="dataset-select-label"
              id="dataset-select"
              value={selectedDataset}
              label="Dataset"
              onChange={(e) => {
                const newDataset = e.target.value;
                setSelectedDataset(newDataset);
                setDatasetInBackend(newDataset);
              }}
              disabled={datasetsLoading || availableDatasets.length === 0}
            >
              {availableDatasets.map((dataset) => (
                <MenuItem key={dataset} value={dataset}>
                  {dataset}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
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
            onClick={() => {
              const offset = paginationModel.page * paginationModel.pageSize;
              fetchHistory(paginationModel.pageSize, offset);
            }}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="outlined"
            color="error"
            startIcon={clearing ? <CircularProgress size={16} color="inherit" /> : <DeleteOutlineIcon />}
            onClick={() => setClearDialogOpen(true)}
            disabled={loading || clearing || totalRecords === 0}
          >
            Clear Logs
          </Button>
        </Stack>
      </Stack>

      {/* Clear Confirmation Dialog */}
      <Dialog open={clearDialogOpen} onClose={() => !clearing && setClearDialogOpen(false)}>
        <DialogTitle>Clear All Logs?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to clear all {totalRecords} log{totalRecords !== 1 ? 's' : ''}? 
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearDialogOpen(false)} disabled={clearing}>
            Cancel
          </Button>
          <Button 
            onClick={clearHistory} 
            color="error" 
            variant="contained"
            disabled={clearing}
            startIcon={clearing ? <CircularProgress size={16} color="inherit" /> : <DeleteOutlineIcon />}
          >
            {clearing ? 'Clearing...' : 'Clear All'}
          </Button>
        </DialogActions>
      </Dialog>

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
                paginationModel={paginationModel}
                onPaginationModelChange={setPaginationModel}
                pageSizeOptions={[10, 25, 50, 100]}
                disableRowSelectionOnClick
                loading={loading}
                rowCount={totalRecords}
                paginationMode="server"
                filterMode="client"
                initialState={{ 
                  sorting: {
                    sortModel: [{ field: 'timestamp', sort: 'desc' }],
                  },
                }}
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
                disableColumnResize={false}
                columnVisibilityModel={{
                  // All columns visible by default
                }}
                sx={{
                  '& .MuiDataGrid-columnHeaders': {
                    backgroundColor: 'background.paper',
                  },
                  '& .MuiDataGrid-columnHeader': {
                    fontWeight: 600,
                  },
                }}
              />
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Data Preview Modal */}
      <Dialog
        open={dataModalOpen}
        onClose={() => setDataModalOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Stack direction="row" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">WebSocket Data Preview</Typography>
            <IconButton
              aria-label="close"
              onClick={() => setDataModalOpen(false)}
              size="small"
            >
              <CloseIcon />
            </IconButton>
          </Stack>
        </DialogTitle>
        <DialogContent>
          {selectedRowData && (
            <Stack spacing={2}>
              <Box>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Record ID: {selectedRowData.id}
                </Typography>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Network ID: {selectedRowData.network_id}
                </Typography>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Timestamp: {selectedRowData.timestamp ? new Date(selectedRowData.timestamp).toLocaleString('en-US', { timeZone: 'UTC' }) : 'N/A'}
                </Typography>
                {selectedRowData.session_start_time && (
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Session Start: {new Date(selectedRowData.session_start_time).toLocaleString('en-US', { timeZone: 'UTC' })}
                  </Typography>
                )}
                {selectedRowData.prediction_results &&
                  (selectedRowData.prediction_results as any).model_name && (
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Model: {(selectedRowData.prediction_results as any).model_name}
                    </Typography>
                  )}
                {selectedRowData.prediction_results?.predictions &&
                  selectedRowData.prediction_results.predictions.length > 0 && (
                    <>
                      {selectedRowData.prediction_results.predictions[0].attack_cat &&
                        selectedRowData.prediction_results.predictions[0].attack_cat !== 'Normal' &&
                        selectedRowData.prediction_results.predictions[0].attack_cat !== null && (
                          <Box sx={{ mt: 1, mb: 1 }}>
                            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                              Attack Category: <strong style={{ color: '#d32f2f' }}>{selectedRowData.prediction_results.predictions[0].attack_cat}</strong>
                            </Typography>
                            {selectedRowData.prediction_results.predictions[0].attack_cat_probabilities &&
                              Object.keys(selectedRowData.prediction_results.predictions[0].attack_cat_probabilities).length > 0 && (
                                <Box sx={{ mt: 1 }}>
                                  <Typography variant="caption" color="text.secondary" gutterBottom>
                                    Category Probabilities:
                                  </Typography>
                                  <Stack direction="row" spacing={1} sx={{ mt: 0.5, flexWrap: 'wrap' }}>
                                    {Object.entries(selectedRowData.prediction_results.predictions[0].attack_cat_probabilities)
                                      .sort(([, a], [, b]) => (b as number) - (a as number))
                                      .slice(0, 5)
                                      .map(([category, prob]) => (
                                        <Chip
                                          key={category}
                                          label={`${category}: ${((prob as number) * 100).toFixed(1)}%`}
                                          size="small"
                                          variant="outlined"
                                          sx={{
                                            fontSize: '0.65rem',
                                            height: 20,
                                            '& .MuiChip-label': { px: 0.75, py: 0.25 },
                                          }}
                                        />
                                      ))}
                                  </Stack>
                                </Box>
                              )}
                          </Box>
                        )}
                    </>
                  )}
              </Box>
              
              {selectedRowData.data && typeof selectedRowData.data === 'object' ? (
                <Box>
                  <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
                    Generated Data:
                  </Typography>
                  <Box
                    sx={{
                      bgcolor: 'background.default',
                      p: 2,
                      borderRadius: 1,
                      border: '1px solid',
                      borderColor: 'divider',
                      maxHeight: 500,
                      overflow: 'auto',
                    }}
                  >
                    <Stack spacing={1}>
                      {Object.entries(selectedRowData.data).map(([key, value]) => (
                        <Box
                          key={key}
                          sx={{
                            display: 'flex',
                            gap: 2,
                            py: 0.5,
                            borderBottom: '1px solid',
                            borderColor: 'divider',
                            '&:last-child': {
                              borderBottom: 'none',
                            },
                          }}
                        >
                          <Typography
                            variant="body2"
                            sx={{
                              fontFamily: 'monospace',
                              fontWeight: 600,
                              minWidth: 150,
                              color: 'primary.main',
                            }}
                          >
                            {key}:
                          </Typography>
                          <Typography
                            variant="body2"
                            sx={{
                              fontFamily: 'monospace',
                              flex: 1,
                              wordBreak: 'break-word',
                            }}
                          >
                            {String(value)}
                          </Typography>
                        </Box>
                      ))}
                    </Stack>
                  </Box>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No data available for this record.
                </Typography>
              )}
            </Stack>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDataModalOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* 3D Globe Card */}
      <Card variant="outlined" sx={{ mt: 2 }}>
        <CardContent>
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              {mapView === '3d' ? '3D SSH Locations Globe' : '2D Campus Locations Map'}
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
          
          {/* Filter Controls */}
          <Stack direction="row" spacing={2} sx={{ mb: 2, flexWrap: 'wrap' }}>
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel id="filter-active-label">Session Status</InputLabel>
              <Select
                labelId="filter-active-label"
                id="filter-active"
                value={filterActive}
                label="Session Status"
                onChange={(e) => setFilterActive(e.target.value as boolean | 'all')}
              >
                <MenuItem value="all">All Sessions</MenuItem>
                <MenuItem value={true}>Active Only</MenuItem>
                <MenuItem value={false}>Inactive Only</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel id="filter-prediction-label">Prediction Status</InputLabel>
              <Select
                labelId="filter-prediction-label"
                id="filter-prediction"
                value={filterPrediction}
                label="Prediction Status"
                onChange={(e) => setFilterPrediction(e.target.value as 'all' | 'safe' | 'anomaly' | 'pending')}
              >
                <MenuItem value="all">All Predictions</MenuItem>
                <MenuItem value="safe">Safe Only</MenuItem>
                <MenuItem value="anomaly">Anomaly Only</MenuItem>
                <MenuItem value="pending">Pending Only</MenuItem>
              </Select>
            </FormControl>
            
            <Button
              size="small"
              variant="outlined"
              onClick={() => {
                setFilterActive('all');
                setFilterPrediction('all');
              }}
            >
              Clear Filters
            </Button>
          </Stack>
          
          <Box sx={{ height: 500, width: '100%', position: 'relative', bgcolor: '#000', borderRadius: 1 }}>
            {(() => {
              // Extract and filter locations with valid coordinates
              let filteredHistory = history.filter(
                (record) =>
                  record.location &&
                  typeof record.location.latitude === 'number' &&
                  typeof record.location.longitude === 'number' &&
                  !isNaN(record.location.latitude) &&
                  !isNaN(record.location.longitude)
              );
              
              // Apply active/inactive filter
              if (filterActive !== 'all') {
                filteredHistory = filteredHistory.filter((record) => record.is_active === filterActive);
              }
              
              // Apply prediction status filter
              if (filterPrediction !== 'all') {
                filteredHistory = filteredHistory.filter((record) => {
                  const predResults = record.prediction_results;
                  if (!predResults || !predResults.predictions || predResults.predictions.length === 0) {
                    return filterPrediction === 'pending';
                  }
                  const prediction = predResults.predictions[0];
                  if (filterPrediction === 'safe') {
                    return prediction.prediction === 0;
                  } else if (filterPrediction === 'anomaly') {
                    return prediction.prediction === 1;
                  }
                  return false;
                });
              }
              
              // Separate SSH and non-SSH locations based on map view
              let locationsForView = filteredHistory;
              if (mapView === '3d') {
                // 3D map: only show SSH locations
                locationsForView = filteredHistory.filter(
                  (record) => record.location && record.location.ssh === true
                );
              } else {
                // 2D map: only show non-SSH locations
                locationsForView = filteredHistory.filter(
                  (record) => record.location && record.location.ssh !== true
                );
              }
              
              const locationsWithCoords = locationsForView.map((record) => ({
                record,
                lat: record.location!.latitude!,
                lon: record.location!.longitude!,
              }));

              if (locationsWithCoords.length === 0) {
                return (
                  <Stack alignItems="center" justifyContent="center" sx={{ height: '100%' }}>
                    <Typography variant="body2" color="text.secondary">
                      {history.length === 0
                        ? 'No location data available. Sessions will appear on the map once they include coordinates.'
                        : mapView === '3d'
                        ? 'No SSH sessions found. SSH sessions will appear on the 3D globe.'
                        : 'No campus sessions found. Campus sessions will appear on the 2D map.'}
                    </Typography>
                  </Stack>
                );
              }

              if (mapView === '3d') {
                return (
                  <ErrorBoundary fallbackMessage="Failed to load the 3D globe. Your browser may not support WebGL, or the texture failed to load. Try switching to 2D Map view.">
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
                  </ErrorBoundary>
                );
              } else {
                // 2D map using Leaflet, focused on TMU campus area
                const TMU_CENTER: [number, number] = [43.6577, -79.3788];
                
                return (
                  <Box sx={{ height: '100%', width: '100%', position: 'relative', zIndex: 0 }}>
                    <MapContainer
                      key={`map-${locationsWithCoords.length}`}
                      center={TMU_CENTER}
                      zoom={locationsWithCoords.length > 0 ? 15 : 16}
                      style={{ height: '100%', width: '100%', borderRadius: 8, zIndex: 0 }}
                      scrollWheelZoom={true}
                      zoomControl={true}
                    >
                      <TileLayer
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                      />
                      <FitBounds locations={locationsWithCoords.map(l => ({ lat: l.lat, lon: l.lon }))} />
                      {locationsWithCoords.map(({ record, lat, lon }) => {
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

                        const color =
                          status === 'Anomaly'
                            ? '#d32f2f'
                            : status === 'Safe'
                            ? '#2e7d32'
                            : '#757575';

                        const loc = record.location;
                        const label = loc?.ssh
                          ? `SSH Connection (${lat.toFixed(4)}, ${lon.toFixed(4)})`
                          : (loc?.name ||
                            (loc?.city && loc?.country
                              ? `${loc.city}, ${loc.country}`
                              : loc?.city || loc?.country || 'Unknown location'));

                        return (
                          <CircleMarker
                            key={record.id}
                            center={[lat, lon]}
                            radius={8}
                            pathOptions={{ 
                              color, 
                              fillColor: color, 
                              fillOpacity: 0.7,
                              weight: 2,
                              opacity: 0.9
                            }}
                          >
                            <Popup>
                              <Stack spacing={0.5} sx={{ minWidth: 200 }}>
                                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                  Session #{record.id}
                                </Typography>
                                <Typography variant="body2" sx={{ fontSize: '0.875rem' }}>
                                  {label}
                                </Typography>
                                <Typography variant="caption" sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>
                                  Lat: {lat.toFixed(5)}, Lon: {lon.toFixed(5)}
                                </Typography>
                                <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                                  <Chip
                                    label={status}
                                    color={isAnomaly ? 'error' : status === 'Safe' ? 'success' : 'default'}
                                    size="small"
                                    sx={{ height: 24, fontSize: '0.75rem' }}
                                  />
                                  <Chip
                                    label={record.is_active ? 'Active' : 'Inactive'}
                                    color={record.is_active ? 'success' : 'default'}
                                    size="small"
                                    variant="outlined"
                                    sx={{ height: 24, fontSize: '0.75rem' }}
                                  />
                                </Stack>
                              </Stack>
                            </Popup>
                          </CircleMarker>
                        );
                      })}
                    </MapContainer>
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
