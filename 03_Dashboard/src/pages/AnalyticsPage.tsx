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
import Slider from '@mui/material/Slider';
import { DataGrid, GridColDef, GridRenderCellParams } from '@mui/x-data-grid';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import WifiIcon from '@mui/icons-material/Wifi';
import WifiOffIcon from '@mui/icons-material/WifiOff';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CloseIcon from '@mui/icons-material/Close';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import ExpandLessRoundedIcon from '@mui/icons-material/ExpandLessRounded';
import ExpandMoreRoundedIcon from '@mui/icons-material/ExpandMoreRounded';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Sphere, useTexture } from '@react-three/drei';
import * as THREE from 'three';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import maplibregl from 'maplibre-gl';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'maplibre-gl/dist/maplibre-gl.css';
import { ErrorBoundary } from '../App';
import { jsPDF } from 'jspdf';
import 'jspdf-autotable';
import FileDownloadIcon from '@mui/icons-material/FileDownload';

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

type PredictionStatus = 'unsafe' | 'safe' | 'pending';

type PredictionLike = {
  prediction?: number;
  label?: string;
};

function normalizePredictionLabel(label?: string): string {
  return (label || '').trim().toLowerCase();
}

function getPredictionStatus(prediction: PredictionLike | null | undefined): PredictionStatus {
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

function SessionPopupContent({
  record,
  lat,
  lon,
  label,
  status,
}: {
  record: HistoryRecord;
  lat: number;
  lon: number;
  label: string;
  status: PredictionStatus;
}) {
  return (
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
          label={toStatusLabel(status)}
          color={getStatusChipColor(status)}
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
  );
}

function toStatusLabel(status: PredictionStatus): string {
  if (status === 'unsafe') return 'Unsafe';
  if (status === 'safe') return 'Safe';
  return 'Pending';
}

function getStatusHexColor(status: PredictionStatus): string {
  if (status === 'unsafe') return '#d32f2f';
  if (status === 'safe') return '#2e7d32';
  return '#757575';
}

function getStatusChipColor(status: PredictionStatus): 'error' | 'success' | 'default' {
  if (status === 'unsafe') return 'error';
  if (status === 'safe') return 'success';
  return 'default';
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
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
function FitBounds({ locations, maxZoom = 17 }: { locations: Array<{ lat: number; lon: number }>; maxZoom?: number }) {
  const map = useMap();
  
  React.useEffect(() => {
    if (locations.length > 0) {
      const bounds = locations.reduce(
        (acc, { lat, lon }) => {
          return acc.extend([lat, lon]);
        },
        new L.LatLngBounds([locations[0].lat, locations[0].lon], [locations[0].lat, locations[0].lon])
      );
      map.fitBounds(bounds, { padding: [20, 20], maxZoom });
    }
  }, [locations, map, maxZoom]);
  
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
        const status = getPredictionStatus(prediction);
        const color = getStatusHexColor(status);

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
                      label={toStatusLabel(status)}
                      color={getStatusChipColor(status)}
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

function MapV2ThreeDBuildings({ locations }: { locations: Array<{ record: HistoryRecord; lat: number; lon: number }> }) {
  const DEFAULT_MAPV2_BEARING = -16;
  const mapContainerRef = React.useRef<HTMLDivElement | null>(null);
  const mapRef = React.useRef<maplibregl.Map | null>(null);
  const markersRef = React.useRef<maplibregl.Marker[]>([]);
  const activePopupRef = React.useRef<maplibregl.Popup | null>(null);
  const [cameraPitch, setCameraPitch] = React.useState(62);
  const [cameraBearing, setCameraBearing] = React.useState(DEFAULT_MAPV2_BEARING);
  const [cameraZoom, setCameraZoom] = React.useState(15.2);
  const [rotateEnabled, setRotateEnabled] = React.useState(true);
  const [controlsExpanded, setControlsExpanded] = React.useState(false);

  const clearMarkers = React.useCallback(() => {
    if (activePopupRef.current) {
      activePopupRef.current.remove();
      activePopupRef.current = null;
    }
    markersRef.current.forEach((marker) => marker.remove());
    markersRef.current = [];
  }, []);

  React.useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: 'https://tiles.openfreemap.org/styles/liberty',
      center: [-79.3788, 43.6577],
      zoom: 15.2,
      pitch: 62,
      bearing: DEFAULT_MAPV2_BEARING,
      maxZoom: 19,
    });

    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl(), 'top-right');

    const syncCameraState = () => {
      setCameraPitch(Number(map.getPitch().toFixed(1)));
      setCameraBearing(Number(map.getBearing().toFixed(1)));
      setCameraZoom(Number(map.getZoom().toFixed(2)));
    };

    const closeActivePopup = () => {
      if (activePopupRef.current) {
        activePopupRef.current.remove();
        activePopupRef.current = null;
      }
    };

    map.on('move', syncCameraState);
    map.on('click', closeActivePopup);

    const add3DBuildings = () => {
      try {
        if (!map.getStyle()) return;

        const keepLabelTokens = [
          'road',
          'street',
          'highway',
          'route',
          'place',
          'city',
          'town',
          'village',
          'district',
          'neighbourhood',
          'neighborhood',
          'country',
          'state',
          'building',
          'address',
          'housenumber',
          'house_number',
        ];

        const hideLabelTokens = [
          'poi',
          'shop',
          'store',
          'cafe',
          'coffee',
          'restaurant',
          'food',
          'drink',
          'bar',
          'pub',
          'fast_food',
          'fuel',
          'pharmacy',
          'hospital',
          'museum',
          'tourism',
          'attraction',
          'transit',
          'station',
          'rail',
          'subway',
          'metro',
          'bus',
          'tram',
          'ferry',
          'airport',
        ];

        const styleLayers = map.getStyle().layers ?? [];
        for (const layer of styleLayers) {
          const idLower = layer.id.toLowerCase();
          const sourceLayerLower = String((layer as any)['source-layer'] ?? '').toLowerCase();
          const combined = `${idLower} ${sourceLayerLower}`;

          if (sourceLayerLower.includes('poi')) {
            map.setLayoutProperty(layer.id, 'visibility', 'none');
            continue;
          }

          if (layer.type !== 'symbol') continue;

          const shouldKeep = keepLabelTokens.some((token) => combined.includes(token));
          const shouldHide = hideLabelTokens.some((token) => combined.includes(token));

          if (shouldHide && !shouldKeep) {
            map.setLayoutProperty(layer.id, 'visibility', 'none');
          }
        }

        map.setLight({
          anchor: 'viewport',
          position: [1.15, 210, 82],
          color: '#f9fcff',
          intensity: 0.62,
        } as any);

        // Option 2: soft atmosphere/haze to improve depth without extra UI.
        const setFog = (map as any).setFog;
        if (typeof setFog === 'function') {
          setFog.call(map, {
            color: '#eaf2ff',
            'high-color': '#f7fbff',
            'horizon-blend': 0.22,
            range: [0.7, 7.5],
            'space-color': '#dce8ff',
            'star-intensity': 0,
          });
        }

        const building3dId = map.getLayer('building-3d') ? 'building-3d' : null;

        if (building3dId) {
          const landmarkNames = [
            'kerr hall',
            'student learning centre',
            'slc',
            'ted rogers school of management',
            'trsm',
            'george vari engineering and computing centre',
            'rac recreation and athletics centre',
            'oakham house',
            'sally horsfall eaton centre',
            'library building',
          ];

          const landmarkMatchExpr = [
            'in',
            ['downcase', ['coalesce', ['get', 'name'], '']],
            ['literal', landmarkNames],
          ] as any;

          // Option 3: emphasize important landmarks via warmer highlight tone.
          map.setPaintProperty(building3dId, 'fill-extrusion-color', [
            'case',
            landmarkMatchExpr,
            '#e9d9b6',
            ['interpolate',
              ['linear'],
              ['coalesce', ['to-number', ['get', 'render_height']], 0],
              0,
              '#d7e5f6',
              120,
              '#b9cfe8',
              300,
              '#9fbde1',
            ],
          ] as any);
          map.setPaintProperty(building3dId, 'fill-extrusion-opacity', 0.96 as any);

          map.setPaintProperty(building3dId, 'fill-extrusion-height', [
            'case',
            landmarkMatchExpr,
            ['*', ['coalesce', ['to-number', ['get', 'render_height']], ['to-number', ['get', 'height']], 0], 1.06],
            ['coalesce', ['to-number', ['get', 'render_height']], ['to-number', ['get', 'height']], 0],
          ] as any);

          // Option 4: roofline/edge contrast using a lightweight outline layer.
          const buildingLayer = map.getLayer(building3dId) as any;
          const outlineId = 'mapv2-building-outline';
          if (buildingLayer?.source && buildingLayer['source-layer'] && !map.getLayer(outlineId)) {
            map.addLayer({
              id: outlineId,
              type: 'line',
              source: buildingLayer.source,
              'source-layer': buildingLayer['source-layer'],
              minzoom: 14,
              paint: {
                'line-color': '#8fa6c6',
                'line-width': [
                  'interpolate',
                  ['linear'],
                  ['zoom'],
                  14,
                  0.25,
                  18,
                  1.1,
                ],
                'line-opacity': 0.58,
              },
            } as any);
          }
        }

        // Keep green areas richer, but no artificial green dots.
        for (const layer of styleLayers) {
          const idLower = layer.id.toLowerCase();
          const sourceLayerLower = String((layer as any)['source-layer'] ?? '').toLowerCase();
          const combined = `${idLower} ${sourceLayerLower}`;

          const isGreenArea =
            combined.includes('park') ||
            combined.includes('grass') ||
            combined.includes('wood') ||
            combined.includes('forest') ||
            combined.includes('green') ||
            combined.includes('landcover') ||
            combined.includes('landuse');

          if (!isGreenArea) continue;

          if (layer.type === 'fill') {
            map.setPaintProperty(layer.id, 'fill-color', '#cbeecd');
            map.setPaintProperty(layer.id, 'fill-opacity', 0.84);
          }

          if (layer.type === 'line') {
            map.setPaintProperty(layer.id, 'line-color', '#99d7a0');
            map.setPaintProperty(layer.id, 'line-opacity', 0.72);
          }
        }

      } catch (err) {
        console.warn('Could not add 3D buildings layer for Map V2:', err);
      }
    };

    map.on('load', add3DBuildings);

    return () => {
      map.off('move', syncCameraState);
      map.off('click', closeActivePopup);
      clearMarkers();
      map.remove();
      mapRef.current = null;
    };
  }, [clearMarkers, DEFAULT_MAPV2_BEARING]);

  const setCameraView = React.useCallback((next: { pitch?: number; bearing?: number; zoom?: number; duration?: number }) => {
    const map = mapRef.current;
    if (!map) return;

    map.easeTo({
      pitch: next.pitch ?? map.getPitch(),
      bearing: next.bearing ?? map.getBearing(),
      zoom: next.zoom ?? map.getZoom(),
      duration: next.duration ?? 450,
    });
  }, []);

  const handlePitchChange = React.useCallback((_event: Event, value: number | number[]) => {
    const nextPitch = Array.isArray(value) ? value[0] : value;
    setCameraPitch(nextPitch);
    setCameraView({ pitch: nextPitch, duration: 0 });
  }, [setCameraView]);

  const handleBearingChange = React.useCallback((_event: Event, value: number | number[]) => {
    const nextBearing = Array.isArray(value) ? value[0] : value;
    setCameraBearing(nextBearing);
    setCameraView({ bearing: nextBearing, duration: 0 });
  }, [setCameraView]);

  const handleZoomChange = React.useCallback((_event: Event, value: number | number[]) => {
    const nextZoom = Array.isArray(value) ? value[0] : value;
    setCameraZoom(nextZoom);
    setCameraView({ zoom: nextZoom, duration: 0 });
  }, [setCameraView]);

  const toggleRotate = React.useCallback(() => {
    const map = mapRef.current;
    if (!map) return;

    setRotateEnabled((prev) => {
      const next = !prev;
      if (next) {
        map.dragRotate.enable();
        map.touchZoomRotate.enableRotation();
      } else {
        map.dragRotate.disable();
        map.touchZoomRotate.disableRotation();
      }
      return next;
    });
  }, []);

  React.useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    clearMarkers();

    if (locations.length === 0) return;

    const bounds = new maplibregl.LngLatBounds();

    locations.forEach(({ record, lat, lon }) => {
      bounds.extend([lon, lat]);

      const prediction = record.prediction_results?.predictions?.[0] ?? null;
      const status = getPredictionStatus(prediction);
      const color = getStatusHexColor(status);

      const loc = record.location;
      const locationLabel = loc?.ssh
        ? `SSH Connection (${lat.toFixed(4)}, ${lon.toFixed(4)})`
        : (loc?.name ||
          (loc?.city && loc?.country
            ? `${loc.city}, ${loc.country}`
            : loc?.city || loc?.country || 'Unknown location'));

      const markerEl = document.createElement('button');
      markerEl.type = 'button';
      markerEl.setAttribute('aria-label', `View details for session ${record.id}`);
      markerEl.title = `Session ${record.id}`;
      markerEl.style.width = '22px';
      markerEl.style.height = '22px';
      markerEl.style.borderRadius = '999px';
      markerEl.style.border = `2px solid #ffffff`;
      markerEl.style.background = color;
      markerEl.style.boxShadow = `0 1px 0 rgba(255,255,255,0.65), 0 0 0 6px ${color}2e, 0 8px 20px ${color}75`;
      markerEl.style.cursor = 'pointer';
      markerEl.style.padding = '0';

      const safeLocationLabel = escapeHtml(locationLabel);
      const safeStatus = escapeHtml(toStatusLabel(status));
      const safeActive = record.is_active ? 'Active' : 'Inactive';
      const statusChipBg = status === 'unsafe' ? '#d32f2f' : status === 'safe' ? '#2e7d32' : '#9e9e9e';
      const activeBg = record.is_active ? '#2e7d32' : '#f8fafc';
      const activeColor = record.is_active ? '#ffffff' : '#14532d';
      const activeBorder = record.is_active ? '#2e7d32' : '#86efac';

      const popupHtml = `
        <div style="min-width: 400px; max-width: 600px; color: #111827; font-family: Inter, system-ui, sans-serif; line-height: 1.35;">
          <div style="font-size: 0.95rem; font-weight: 700; margin-bottom: 6px; color: #374151;">Session #${record.id}</div>
          <div style="font-size: 0.875rem; color: #4b5563; margin-bottom: 8px; word-break: break-word;">${safeLocationLabel}</div>
          <div style="font-size: 0.78rem; color: #94a3b8; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin-bottom: 10px;">Lat: ${lat.toFixed(5)}, Lon: ${lon.toFixed(5)}</div>
          <div style="display: flex; gap: 8px; align-items: center; margin-top: 2px;">
            <span style="display: inline-flex; align-items: center; justify-content: center; height: 24px; padding: 0 10px; border-radius: 999px; font-size: 0.76rem; font-weight: 700; color: #ffffff; background: ${statusChipBg};">${safeStatus}</span>
            <span style="display: inline-flex; align-items: center; justify-content: center; height: 24px; padding: 0 10px; border-radius: 999px; font-size: 0.76rem; font-weight: 600; color: ${activeColor}; background: ${activeBg}; border: 1px solid ${activeBorder};">${safeActive}</span>
          </div>
        </div>
      `;

      const popup = new maplibregl.Popup({ offset: 18, closeOnClick: false, closeButton: true }).setHTML(popupHtml);
      popup.on('close', () => {
        if (activePopupRef.current === popup) {
          activePopupRef.current = null;
        }
      });

      const marker = new maplibregl.Marker({ element: markerEl, anchor: 'center' })
        .setLngLat([lon, lat])
        .setPopup(popup)
        .addTo(map);

      const openPopupForMarker = () => {
        if (activePopupRef.current && activePopupRef.current !== popup) {
          activePopupRef.current.remove();
        }
        popup.setLngLat([lon, lat]).addTo(map);
        activePopupRef.current = popup;
      };

      markerEl.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        openPopupForMarker();
      });
      markerEl.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          event.stopPropagation();
          openPopupForMarker();
        }
      });
      markersRef.current.push(marker);
    });

    map.fitBounds(bounds, {
      padding: 48,
      maxZoom: 17,
      duration: 700,
      pitch: 62,
      bearing: DEFAULT_MAPV2_BEARING,
    });
  }, [locations, clearMarkers, DEFAULT_MAPV2_BEARING]);

  return (
    <Box
      sx={{
        height: '100%',
        width: '100%',
        borderRadius: 1,
        overflow: 'hidden',
        position: 'relative',
        bgcolor: '#d8e3ef',
        '& .maplibregl-ctrl-group': {
          bgcolor: 'rgba(255, 255, 255, 0.92)',
          border: '1px solid rgba(148, 163, 184, 0.45)',
          boxShadow: '0 6px 18px rgba(15, 23, 42, 0.15)',
        },
        '& .maplibregl-ctrl button': {
          color: '#0f172a',
        },
        '& .maplibregl-popup-content': {
          bgcolor: '#f3f4f6',
          border: '1px solid #d1d5db',
          borderRadius: '12px',
          p: '12px',
          boxShadow: '0 10px 24px rgba(15, 23, 42, 0.22)',
          minWidth: '400px',
          maxWidth: '600px',
        },
        '& .maplibregl-popup-close-button': {
          color: '#9ca3af',
          fontSize: '18px',
          lineHeight: 1,
          padding: '8px 10px',
          border: 'none',
          background: 'transparent',
          right: 0,
          top: 0,
        },
        '& .maplibregl-popup-close-button:hover': {
          color: '#6b7280',
          background: 'transparent',
        },
        '& .maplibregl-popup-tip': {
          borderTopColor: '#f3f4f6',
          borderBottomColor: '#f3f4f6',
        },
      }}
    >
      <Box ref={mapContainerRef} sx={{ height: '100%', width: '100%' }} />
      <Box
        sx={{
          position: 'absolute',
          left: 12,
          top: 12,
          zIndex: 5,
          width: { xs: 'calc(100% - 24px)', sm: 290 },
          bgcolor: 'rgba(5, 8, 12, 0.88)',
          border: '1px solid rgba(255,255,255,0.16)',
          borderRadius: 2,
          boxShadow: '0 10px 24px rgba(2, 6, 12, 0.45)',
          backdropFilter: 'blur(6px)',
          p: 1.25,
        }}
      >
        <Stack spacing={1}>
          <Stack direction="row" alignItems="center" justifyContent="space-between">
            <Typography sx={{ fontSize: '0.78rem', fontWeight: 800, color: '#ffffff' }}>
              Camera Controls
            </Typography>
            <IconButton
              size="small"
              onClick={() => setControlsExpanded((prev) => !prev)}
              sx={{
                color: '#ffffff',
                bgcolor: '#000000',
                border: '1px solid rgba(255,255,255,0.26)',
                width: 24,
                height: 24,
                '&:hover': { bgcolor: '#111111' },
              }}
            >
              {controlsExpanded ? <ExpandLessRoundedIcon fontSize="small" /> : <ExpandMoreRoundedIcon fontSize="small" />}
            </IconButton>
          </Stack>

          {controlsExpanded && (
            <>

          <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
            <Button size="small" variant="outlined" onClick={() => setCameraView({ pitch: 0, bearing: DEFAULT_MAPV2_BEARING, duration: 550 })}>
              Top-Down 2D
            </Button>
            <Button size="small" variant="outlined" onClick={() => setCameraView({ pitch: 62, bearing: DEFAULT_MAPV2_BEARING, duration: 550 })}>
              Angled 3D
            </Button>
            <Button size="small" variant="outlined" onClick={() => setCameraView({ pitch: 78, bearing: cameraBearing, duration: 550 })}>
              Max Tilt
            </Button>
          </Stack>

          <Button size="small" variant={rotateEnabled ? 'contained' : 'outlined'} onClick={toggleRotate}>
            {rotateEnabled ? 'Rotate Drag: ON' : 'Rotate Drag: OFF'}
          </Button>

          <Typography sx={{ fontSize: '0.7rem', color: '#e2e8f0', fontWeight: 700 }}>
            Pitch ({cameraPitch.toFixed(1)} deg)
          </Typography>
          <Slider size="small" value={cameraPitch} min={0} max={80} step={1} onChange={handlePitchChange} />

          <Typography sx={{ fontSize: '0.7rem', color: '#e2e8f0', fontWeight: 700 }}>
            Bearing ({cameraBearing.toFixed(1)} deg)
          </Typography>
          <Slider size="small" value={cameraBearing} min={-180} max={180} step={1} onChange={handleBearingChange} />

          <Typography sx={{ fontSize: '0.7rem', color: '#e2e8f0', fontWeight: 700 }}>
            Zoom ({cameraZoom.toFixed(2)})
          </Typography>
          <Slider size="small" value={cameraZoom} min={10} max={19} step={0.1} onChange={handleZoomChange} />
            </>
          )}
        </Stack>
      </Box>
    </Box>
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
  const [mapView, setMapView] = React.useState<'2d' | '3d' | 'mapv2'>('3d');
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

  // Export table to PDF
  const exportToPDF = React.useCallback(async () => {
    try {
      setLoading(true);
      setSnackbar({ open: true, message: 'Fetching all records...', severity: 'info' });

      // Fetch ALL records (not just current page)
      const res = await fetch(`${USER_SERVICE_BASE}/history?limit=10000&offset=0`, {
        method: 'GET',
      });

      if (!res.ok) {
        let errorMessage = `HTTP ${res.status}: ${res.statusText}`;
        try {
          const errorJson = await res.json() as { detail?: string };
          if (errorJson.detail) {
            errorMessage = errorJson.detail;
          }
        } catch (parseErr) {
          // ignore
        }
        setSnackbar({ open: true, message: `Failed to fetch records: ${errorMessage}`, severity: 'error' });
        return;
      }

      const json = await res.json() as { 
        status?: string;
        history?: HistoryRecord[];
        total_records?: number;
        total?: number;
      };

      if (json.status !== 'success' || !json.history || json.history.length === 0) {
        setSnackbar({ open: true, message: 'No data to export', severity: 'warning' });
        return;
      }

      const allRecords = json.history;
      setSnackbar({ open: true, message: `Exporting ${allRecords.length} records to PDF...`, severity: 'info' });

      // Create PDF with proper text (selectable and copyable)
      const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'mm',
        format: 'a4',
      }) as any;

      // Add title
      pdf.setFontSize(14);
      pdf.setTextColor(0, 0, 0);
      pdf.text('Analytics Sessions Report', 15, 15);
      pdf.setFontSize(10);
      pdf.setTextColor(80, 80, 80);
      pdf.text(`Generated on: ${new Date().toLocaleString('en-US')}`, 15, 22);
      pdf.text(`Total Records: ${allRecords.length}`, 15, 28);

      // Prepare table data
      const tableData = allRecords.map((record) => {
        // Location
        let location = 'N/A';
        if (record.location) {
          if (record.location.ssh) {
            location = `SSH: ${record.location.latitude?.toFixed(4)}, ${record.location.longitude?.toFixed(4)}`;
          } else if (record.location.name) {
            location = record.location.name;
          } else if (record.location.city && record.location.country) {
            location = `${record.location.city}, ${record.location.country}`;
          } else {
            location = record.location.city || record.location.country || 'N/A';
          }
        }

        // Prediction
        let prediction = 'Pending';
        if (record.prediction_results?.predictions && record.prediction_results.predictions.length > 0) {
          const pred = record.prediction_results.predictions[0];
          const status = getPredictionStatus(pred);
          const label = toStatusLabel(status);
          const percent = 
            status === 'unsafe'
              ? formatPercent(pred.probability_unsafe) || formatPercent(pred.confidence)
              : status === 'safe'
              ? formatPercent(pred.probability_safe) || formatPercent(pred.confidence)  
              : null;
          prediction = percent ? `${label} (${percent})` : label;
        }

        // Attack Category
        let attackCat = '-';
        if (record.prediction_results?.predictions && record.prediction_results.predictions.length > 0) {
          const pred = record.prediction_results.predictions[0];
          if (pred.attack_cat && pred.attack_cat !== 'Normal' && pred.attack_cat !== null) {
            attackCat = pred.attack_cat;
          }
        }

        // Timestamp
        const timestamp = record.timestamp ? new Date(record.timestamp).toLocaleString('en-US') : 'N/A';

        return [
          String(record.id || 'N/A'),
          String(record.network_id || 'N/A'),
          timestamp,
          String(record.user_id || 'N/A'),
          String(record.os || 'N/A'),
          String(record.browser || 'N/A'),
          location,
          prediction,
          attackCat,
        ];
      });

      // Add table using autoTable plugin
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      
      (pdf as any).autoTable({
        head: [['ID', 'Network ID', 'Timestamp', 'User ID', 'OS', 'Browser', 'Location', 'Prediction', 'Attack Category']],
        body: tableData,
        startY: 35,
        margin: { top: 35, right: 10, bottom: 10, left: 10 },
        pageWidth: pageWidth,
        pageHeight: pageHeight,
        columnStyles: {
          0: { cellWidth: 15 },  // ID
          1: { cellWidth: 40 },  // Network ID
          2: { cellWidth: 45 },  // Timestamp
          3: { cellWidth: 15 },  // User ID
          4: { cellWidth: 20 },  // OS
          5: { cellWidth: 20 },  // Browser
          6: { cellWidth: 35 },  // Location
          7: { cellWidth: 30 },  // Prediction
          8: { cellWidth: 30 },  // Attack Category
        },
        headStyles: {
          fillColor: [229, 231, 235],
          textColor: [0, 0, 0],
          fontStyle: 'bold',
          halign: 'left',
          fontSize: 10,
        },
        bodyStyles: {
          textColor: [0, 0, 0],
          fontSize: 9,
        },
        alternateRowStyles: {
          fillColor: [243, 244, 246],
        },
        didDrawPage: (data: any) => {
          // Re-add title to each page
          const pageSize = pdf.internal.pageSize;
          const pageHeight = pageSize.getHeight();
          const pageWidth = pageSize.getWidth();
          
          pdf.setFontSize(10);
          pdf.setTextColor(80, 80, 80);
          pdf.text(`Page ${data.pageNumber}`, pageWidth - 30, pageHeight - 10);
        },
      });

      // Save the PDF
      pdf.save(`Analytics_Report_${new Date().toISOString().split('T')[0]}.pdf`);

      setSnackbar({ open: true, message: `PDF exported successfully with ${allRecords.length} records`, severity: 'success' });
    } catch (err) {
      console.error('Failed to export PDF:', err);
      let errorMessage = 'Failed to export PDF. ';
      if (err instanceof TypeError && err.message.includes('fetch')) {
        errorMessage += 'Network error - is the User Service running?';
      } else if (err instanceof Error) {
        errorMessage += err.message;
      }
      setSnackbar({ open: true, message: errorMessage, severity: 'error' });
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
        return toStatusLabel(getPredictionStatus(prediction));
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
        const status = getPredictionStatus(prediction);
        const label = toStatusLabel(status);
        const percent =
          status === 'unsafe'
            ? formatPercent(prediction.probability_unsafe) || formatPercent(prediction.confidence)
            : status === 'safe'
            ? formatPercent(prediction.probability_safe) || formatPercent(prediction.confidence)
            : null;

        // Model name used for this prediction (added by backend)
        const modelName = predResults.model_name;
        
        // Attack category (only available for RFv1 models and unsafe predictions)
        const attackCat = prediction.attack_cat;
        
        return (
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Chip 
              label={percent ? `${label} (${percent})` : label}
              color={getStatusChipColor(status)}
              size="small" 
            />
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
            startIcon={<FileDownloadIcon />}
            onClick={exportToPDF}
            disabled={loading || history.length === 0}
          >
            Export to PDF
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
              {mapView === '3d' ? '3D SSH Locations Globe' : mapView === 'mapv2' ? '3D Map' : '2D Campus Locations Map'}
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
              <Button
                size="small"
                variant={mapView === 'mapv2' ? 'contained' : 'outlined'}
                onClick={() => setMapView('mapv2')}
              >
                3D Map
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
                  const status = getPredictionStatus(prediction);
                  if (filterPrediction === 'safe') {
                    return status === 'safe';
                  } else if (filterPrediction === 'anomaly') {
                    return status === 'unsafe';
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
                // 2D/MapV2: only show non-SSH locations
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
              } else if (mapView === 'mapv2') {
                return <MapV2ThreeDBuildings locations={locationsWithCoords} />;
              } else {
                // 2D map using Leaflet, focused on TMU campus area
                const TMU_CENTER: [number, number] = [43.6577, -79.3788];
                
                return (
                  <Box sx={{ height: '100%', width: '100%', position: 'relative', zIndex: 0 }}>
                    <MapContainer
                      key={`map-${mapView}-${locationsWithCoords.length}`}
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
                        const status = getPredictionStatus(prediction);
                        const color = getStatusHexColor(status);

                        const loc = record.location;
                        const label = loc?.ssh
                          ? `SSH Connection (${lat.toFixed(4)}, ${lon.toFixed(4)})`
                          : (loc?.name ||
                            (loc?.city && loc?.country
                              ? `${loc.city}, ${loc.country}`
                              : loc?.city || loc?.country || 'Unknown location'));

                        return (
                          <React.Fragment key={record.id}>
                            <CircleMarker
                              center={[lat, lon]}
                              radius={8}
                              pathOptions={{ 
                                color, 
                                fillColor: color, 
                                fillOpacity: 0.7,
                                weight: 2,
                                opacity: 0.95
                              }}
                            >
                              <Popup>
                                <SessionPopupContent
                                  record={record}
                                  lat={lat}
                                  lon={lon}
                                  label={label}
                                  status={status}
                                />
                              </Popup>
                            </CircleMarker>
                          </React.Fragment>
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
