import * as React from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import FormControl from '@mui/material/FormControl';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import Grid from '@mui/material/Grid';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import TextField from '@mui/material/TextField';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import UploadFileRoundedIcon from '@mui/icons-material/UploadFileRounded';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import AddRoundedIcon from '@mui/icons-material/AddRounded';
import DeleteSweepRoundedIcon from '@mui/icons-material/DeleteSweepRounded';
import PsychologyRoundedIcon from '@mui/icons-material/PsychologyRounded';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import { DataGrid } from '@mui/x-data-grid';

/* Mock data & types */
const DEFAULT_FEATURE_COLUMNS = ['duration', 'protocol', 'bytes_sent', 'bytes_recv', 'packets', 'label'];
const MOCK_MODEL_TYPES = [
  'Isolation Forest (Anomaly)',
  'One-Class SVM (Anomaly)',
  'Random Forest (Classification)',
  'Logistic Regression (Classification)',
  'XGBoost',
];

const createMockRows = (count: number) =>
  Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    duration: (Math.random() * 100).toFixed(2),
    protocol: ['tcp', 'udp', 'icmp'][Math.floor(Math.random() * 3)],
    bytes_sent: Math.floor(Math.random() * 10000),
    bytes_recv: Math.floor(Math.random() * 5000),
    packets: Math.floor(Math.random() * 100),
    label: ['normal', 'anomaly'][Math.floor(Math.random() * 2)],
  }));

const MOCK_INITIAL_ROWS = createMockRows(12);

const API_BASE = 'http://localhost:8000';

export default function ModelPage() {
  const [datasetName, setDatasetName] = React.useState('');
  const [datasetNameError, setDatasetNameError] = React.useState('');
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [uploading, setUploading] = React.useState(false);
  const [datasets, setDatasets] = React.useState<{ id: string; name: string }[]>([]);
  const [rows, setRows] = React.useState<Record<string, unknown>[]>([]);
  const [viewLimit, setViewLimit] = React.useState(1000);
  const [viewLoading, setViewLoading] = React.useState(false);
  const [viewTotalRows, setViewTotalRows] = React.useState<number | null>(null);
  const [filterMode, setFilterMode] = React.useState<'all' | 'training' | 'testing'>('all');
  const [searchQuery, setSearchQuery] = React.useState('');
  const [validating, setValidating] = React.useState(false);
  const [insertText, setInsertText] = React.useState('');
  const [clearConfirmOpen, setClearConfirmOpen] = React.useState(false);
  const [clearLoading, setClearLoading] = React.useState(false);
  const [insertLoading, setInsertLoading] = React.useState(false);
  const [modelName, setModelName] = React.useState('');
  const [modelNameError, setModelNameError] = React.useState('');
  const [selectedDatasetId, setSelectedDatasetId] = React.useState('');
  const [selectedFeatures, setSelectedFeatures] = React.useState<string[]>([]);
  const [modelType, setModelType] = React.useState('Random Forest');
  const [training, setTraining] = React.useState(false);
  const [metrics, setMetrics] = React.useState<any>(null);
  const [predictionResults, setPredictionResults] = React.useState<any[]>([]);
  const [predicting, setPredicting] = React.useState(false);
  const [predictionInput, setPredictionInput] = React.useState('');


  const [snackbar, setSnackbar] = React.useState<{ open: boolean; message: string; severity: 'success' | 'error' | 'info' | 'warning' }>({
    open: false,
    message: '',
    severity: 'success',
  });
  const [apiHealth, setApiHealth] = React.useState<'healthy' | 'unhealthy' | 'loading' | null>(null);
  const [apiHealthDetail, setApiHealthDetail] = React.useState<{ service?: string; database?: string; timestamp?: string } | null>(null);

  const filteredRows = React.useMemo(() => {
    let result = rows;
    if (filterMode === 'training') result = result.slice(0, Math.ceil(result.length * 0.8));
    else if (filterMode === 'testing') result = result.slice(Math.ceil(result.length * 0.8));
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter((r) => Object.values(r).some((v) => String(v).toLowerCase().includes(q)));
    }
    return result;
  }, [rows, filterMode, searchQuery]);

  const columns = React.useMemo(() => {
    if (filteredRows.length === 0) return [];
    return Object.keys(filteredRows[0] as object)
      .filter((k) => k !== 'id')
      .map((k) => ({ field: k, headerName: k, flex: 1, minWidth: 100 }));
  }, [filteredRows]);

  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleAttachClick = () => {
    setDatasetNameError('');
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    e.target.value = '';
  };

  const handleUploadToBackend = async () => {
    if (!selectedFile) {
      setSnackbar({ open: true, message: 'Attach a CSV file first.', severity: 'info' });
      return;
    }
    setUploading(true);
    setDatasetNameError('');
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const headers: Record<string, string> = {};
      if (datasetName.trim()) {
        headers['X-Database-Name'] = datasetName.trim();
      }
      const res = await fetch(`${API_BASE}/new`, {
        method: 'POST',
        headers,
        body: formData,
      });
      const responseText = await res.text();
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const errBody = JSON.parse(responseText) as { detail?: unknown };
          detail = Array.isArray(errBody.detail)
            ? (errBody.detail as { msg?: string }[]).map((d) => d.msg ?? '').join('; ')
            : (errBody.detail as string) ?? detail;
        } catch {
          if (responseText) detail = responseText;
        }
        setSnackbar({ open: true, message: `Upload failed: ${detail}`, severity: 'error' });
        setUploading(false);
        return;
      }
      let message = 'CSV uploaded successfully.';
      try {
        const result = JSON.parse(responseText) as string | { message?: string };
        message = typeof result === 'string' ? result : result.message ?? message;
      } catch {
        if (responseText) message = responseText;
      }
      setSnackbar({ open: true, message, severity: 'success' });
      setSelectedFile(null);
      if (datasetName.trim()) {
        const id = `ds-${Date.now()}`;
        setDatasets((d) => [...d, { id, name: datasetName.trim() }]);
        setSelectedDatasetId(id);
      }
      fetchViewData(viewLimit, 0);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Network error. Is the backend running at http://localhost:8000?';
      setSnackbar({ open: true, message, severity: 'error' });
    } finally {
      setUploading(false);
    }
  };

  const fetchViewData = React.useCallback(
    async (limit: number, offset: number) => {
      setViewLoading(true);
      setViewTotalRows(null);
      try {
        const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
        const headers: Record<string, string> = {};
        if (datasetName.trim()) headers['X-Database-Name'] = datasetName.trim();
        const res = await fetch(`${API_BASE}/view?${params}`, { headers });
        const json = (await res.json()) as {
          status?: string;
          data?: { id: number; upload_timestamp?: string; data: Record<string, unknown>; T?: unknown }[];
          total_rows?: number;
          returned_rows?: number;
        };
        if (!res.ok) {
          const detail = (json as { detail?: string | { msg?: string }[] }).detail;
          const msg = Array.isArray(detail) ? detail.map((d) => d.msg ?? '').join('; ') : String(detail ?? res.statusText);
          setSnackbar({ open: true, message: `View data failed: ${msg}`, severity: 'error' });
          setViewLoading(false);
          return;
        }
        const raw = json.data ?? [];
        const gridRows: Record<string, unknown>[] = raw.map((item) => ({
          id: item.id,
          ...item.data,
          ...(item.upload_timestamp != null && { upload_timestamp: item.upload_timestamp }),
          ...(item.T != null && { T: item.T }),
        }));
        setRows(gridRows);
        if (typeof json.total_rows === 'number') setViewTotalRows(json.total_rows);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load view data.';
        setSnackbar({ open: true, message, severity: 'error' });
      } finally {
        setViewLoading(false);
      }
    },
    [datasetName]
  );

  React.useEffect(() => {
    fetchViewData(viewLimit, 0);
  }, [viewLimit, fetchViewData]);

  const fetchApiHealth = React.useCallback(async (silent = false) => {
    if (!silent) {
      setApiHealth('loading');
      setApiHealthDetail(null);
    }
    try {
      const headers: Record<string, string> = {};
      if (datasetName.trim()) headers['X-Database-Name'] = datasetName.trim();
      const res = await fetch(`${API_BASE}/health`, { headers });
      const json = (await res.json()) as { status?: string; service?: string; database?: string; timestamp?: string };
      if (res.ok && json.status === 'healthy') {
        setApiHealth('healthy');
        setApiHealthDetail({
          service: json.service,
          database: json.database,
          timestamp: json.timestamp,
        });
      } else {
        setApiHealth('unhealthy');
      }
    } catch {
      setApiHealth('unhealthy');
    }
  }, [datasetName]);

  React.useEffect(() => {
    fetchApiHealth();
    const interval = setInterval(() => fetchApiHealth(true), 15000);
    return () => clearInterval(interval);
  }, [fetchApiHealth]);

  const handleViewLimitChange = (newLimit: number) => {
    setViewLimit(newLimit);
  };

  const [validationResult, setValidationResult] = React.useState<{ message: string; severity: 'success' | 'warning' } | null>(null);

  const handleRevalidate = async () => {
    setValidating(true);
    setValidationResult(null);
    try {
      const headers: Record<string, string> = {};
      if (datasetName.trim()) headers['X-Database-Name'] = datasetName.trim();
      const res = await fetch(`${API_BASE}/validate`, { method: 'PUT', headers });
      const text = await res.text();
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const json = JSON.parse(text) as { detail?: string };
          detail = json.detail ?? detail;
        } catch {
          if (text) detail = text;
        }
        setValidationResult({ message: `Validation failed: ${detail}`, severity: 'warning' });
        setSnackbar({ open: true, message: `Validation failed: ${detail}`, severity: 'error' });
        setValidating(false);
        return;
      }
      let message = 'Validation completed.';
      try {
        const json = JSON.parse(text) as {
          message?: string;
          total_rows?: number;
          training_rows?: number;
          testing_rows?: number;
          training_percentage?: number;
          testing_percentage?: number;
        };
        if (json.message) message = json.message;
        if (
          typeof json.training_rows === 'number' &&
          typeof json.testing_rows === 'number'
        ) {
          message = `Validation: ✅ ${json.training_rows} training (${json.training_percentage ?? '—'}%), ${json.testing_rows} testing (${json.testing_percentage ?? '—'}%)`;
        } else if (json.total_rows === 0) {
          message = 'No rows to validate.';
        }
      } catch {
        if (text) message = text;
      }
      setValidationResult({ message, severity: 'success' });
      setSnackbar({ open: true, message, severity: 'success' });
      fetchViewData(viewLimit, 0);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to validate dataset.';
      setValidationResult({ message: msg, severity: 'warning' });
      setSnackbar({ open: true, message: msg, severity: 'error' });
    } finally {
      setValidating(false);
    }
  };

  const handleInsert = async () => {
    if (!insertText.trim()) {
      setSnackbar({ open: true, message: 'Paste a CSV row first.', severity: 'info' });
      return;
    }
    const cols = rows.length > 0 ? Object.keys(rows[0] as object).filter((k) => k !== 'id' && k !== 'upload_timestamp' && k !== 'T') : DEFAULT_FEATURE_COLUMNS;
    if (rows.length === 0) {
      setSnackbar({ open: true, message: 'Load view data first so columns are known, or paste a CSV row with columns: ' + cols.join(', '), severity: 'info' });
      return;
    }
    const parts = insertText.trim().split(',').map((p) => p.trim());
    if (parts.length < cols.length) {
      setSnackbar({ open: true, message: `Invalid format. Use: ${cols.join(', ')}`, severity: 'error' });
      return;
    }
    const data: Record<string, unknown> = {};
    cols.forEach((c, i) => { data[c] = parts[i] ?? ''; });
    setInsertLoading(true);
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (datasetName.trim()) headers['X-Database-Name'] = datasetName.trim();
      const res = await fetch(`${API_BASE}/insert`, {
        method: 'POST',
        headers,
        body: JSON.stringify(data),
      });
      const text = await res.text();
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const json = JSON.parse(text) as { detail?: string | { msg?: string }[] };
          detail = Array.isArray(json.detail) ? (json.detail as { msg?: string }[]).map((d) => d.msg ?? '').join('; ') : String(json.detail ?? detail);
        } catch {
          if (text) detail = text;
        }
        setSnackbar({ open: true, message: `Insert failed: ${detail}`, severity: 'error' });
        setInsertLoading(false);
        return;
      }
      setInsertText('');
      setSnackbar({ open: true, message: 'Row inserted successfully.', severity: 'success' });
      fetchViewData(viewLimit, 0);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to insert row.';
      setSnackbar({ open: true, message: msg, severity: 'error' });
    } finally {
      setInsertLoading(false);
    }
  };

  const handleClearConfirm = async () => {
    setClearLoading(true);
    try {
      const headers: Record<string, string> = {};
      if (datasetName.trim()) headers['X-Database-Name'] = datasetName.trim();
      const res = await fetch(`${API_BASE}/clear`, { method: 'POST', headers });
      const text = await res.text();
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const json = JSON.parse(text) as { detail?: string };
          detail = json.detail ?? detail;
        } catch {
          if (text) detail = text;
        }
        setSnackbar({ open: true, message: `Clear failed: ${detail}`, severity: 'error' });
        setClearLoading(false);
        return;
      }
      let message = 'Database table cleared.';
      try {
        const json = JSON.parse(text) as { rows_deleted?: number; message?: string };
        if (typeof json.rows_deleted === 'number') message = `Database cleared. ${json.rows_deleted} row(s) deleted.`;
        else if (json.message) message = json.message;
      } catch {
        if (text) message = text;
      }
      setRows([]);
      setMetrics(null);
      setValidationResult(null);
      setViewTotalRows(0);
      setClearConfirmOpen(false);
      setSnackbar({ open: true, message, severity: 'success' });
      fetchViewData(viewLimit, 0);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to clear database.';
      setSnackbar({ open: true, message: msg, severity: 'error' });
    } finally {
      setClearLoading(false);
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setMetrics(null);

    try {
      const payload = {
        model_name: `${modelType.replace(/ /g, '')}_${new Date().getTime()}`,
        dataset_name: datasetName.trim() || undefined,
        features: [], // Backend auto-selects
        model_type: modelType,
      };

      const res = await fetch(`${API_BASE}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Training failed');
      }

      setMetrics(data);
      setSnackbar({ open: true, message: `${modelType} trained successfully`, severity: 'success' });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to train model.';
      setSnackbar({ open: true, message: msg, severity: 'error' });
    } finally {
      setTraining(false);
    }
  };

  const handlePredict = async (useTestData: boolean = true) => {
    if (!metrics?.model_path) {
      setSnackbar({ open: true, message: 'Train a model first to get a model path.', severity: 'warning' });
      return;
    }

    setPredicting(true);
    try {
      let dataToPredict: any[] = [];

      if (useTestData) {
        // Fetch testing data from backend
        const params = new URLSearchParams({ limit: '100', offset: '0' });
        const headers: Record<string, string> = {};
        if (datasetName.trim()) headers['X-Database-Name'] = datasetName.trim();
        const res = await fetch(`${API_BASE}/testing?${params}`, { headers });
        const json = await res.json();
        if (!res.ok) throw new Error(json.detail || 'Failed to fetch testing data');
        dataToPredict = (json.data || []).map((item: any) => item.data);
      } else {
        // Parse from local input field
        if (!predictionInput.trim()) {
          setSnackbar({ open: true, message: 'Paste data rows for prediction.', severity: 'info' });
          setPredicting(false);
          return;
        }
        // Simplified CSV parsing for the single input
        const cols = metrics.features.split(', ');
        const parts = predictionInput.trim().split(',').map(p => p.trim());
        const obj: any = {};
        cols.forEach((c: string, i: number) => { obj[c] = parts[i] || 0; });
        dataToPredict = [obj];
      }

      if (dataToPredict.length === 0) {
        throw new Error('No data found for prediction.');
      }

      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: metrics.model_path,
          data: dataToPredict
        })
      });

      const result = await res.json();
      if (!res.ok) throw new Error(result.detail || 'Prediction failed');

      setPredictionResults(result.results);
      setSnackbar({ open: true, message: `Successfully generated ${result.results.length} predictions.`, severity: 'success' });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to run prediction.';
      setSnackbar({ open: true, message: msg, severity: 'error' });
    } finally {
      setPredicting(false);
    }
  };

  const handleFeatureToggle = (f: string) => () => {
    setSelectedFeatures((prev) =>
      prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]
    );
  };

  const featureColumns = React.useMemo(() => {
    if (rows.length === 0) return DEFAULT_FEATURE_COLUMNS;
    const keys = Object.keys(rows[0] as object).filter((k) => k !== 'id');
    return keys.length > 0 ? keys : DEFAULT_FEATURE_COLUMNS;
  }, [rows]);

  const canTrain = modelName.trim() && selectedDatasetId && selectedFeatures.length > 0 && modelType;

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      {/* Page Header */}
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2.5 }}>
        <Box
          sx={(theme) => ({
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 52,
            height: 52,
            borderRadius: 2,
            background: `linear-gradient(145deg, ${theme.palette.primary.main}28 0%, ${theme.palette.primary.dark}12 100%)`,
            border: '2px solid',
            borderColor: 'primary.main',
            position: 'relative',
            overflow: 'visible',
            transformOrigin: 'center',
            '@keyframes pulseGlow': {
              '0%, 100%': {
                boxShadow: `0 0 0 3px ${theme.palette.primary.main}35`,
              },
              '50%': {
                boxShadow: `0 0 0 10px ${theme.palette.primary.main}22, 0 0 28px ${theme.palette.primary.main}18`,
              },
            },
            '@keyframes breathe': {
              '0%, 100%': { transform: 'scale(1)' },
              '50%': { transform: 'scale(1.05)' },
            },
            animation: 'pulseGlow 2.2s ease-in-out infinite, breathe 2.2s ease-in-out infinite',
            '@media (prefers-reduced-motion: reduce)': {
              animation: 'none',
            },
          })}
        >
          <PsychologyRoundedIcon
            sx={(theme) => ({
              fontSize: 28,
              color: 'primary.main',
              '@keyframes iconGlow': {
                '0%, 100%': { opacity: 1, filter: 'brightness(1)' },
                '50%': { opacity: 0.92, filter: 'brightness(1.08)' },
              },
              animation: 'iconGlow 2.2s ease-in-out infinite',
              '@media (prefers-reduced-motion: reduce)': { animation: 'none' },
            })}
          />
        </Box>
        <Stack spacing={0.25}>
          <Typography component="h1" variant="h5" sx={{ fontWeight: 600, letterSpacing: '-0.02em', lineHeight: 1.3 }}>
            Model Handling
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.45 }}>
            Campus IoT anomaly detection — upload a dataset, validate it, then configure and train a model.
          </Typography>
        </Stack>
      </Stack>

      <Stack spacing={3}>
        {/* API Health (above Dataset Setup) */}
        <Stack
          direction="row"
          alignItems="center"
          spacing={1.5}
          sx={{
            px: 2,
            py: 1.25,
            borderRadius: 1.5,
            bgcolor: 'action.hover',
            border: '1px solid',
            borderColor: 'divider',
          }}
        >
          {apiHealth === 'loading' && (
            <CircularProgress size={16} sx={{ color: 'text.secondary' }} />
          )}
          {apiHealth === 'healthy' && (
            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'success.main' }} />
          )}
          {apiHealth === 'unhealthy' && (
            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'error.main' }} />
          )}
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            API: {apiHealth === 'loading' ? 'Checking…' : apiHealth === 'healthy' ? 'Healthy' : 'Unreachable'}
          </Typography>
          {apiHealth === 'healthy' && apiHealthDetail?.service && (
            <Typography variant="caption" color="text.secondary">
              {apiHealthDetail.service}
              {apiHealthDetail.database ? ` · DB: ${apiHealthDetail.database}` : ''}
            </Typography>
          )}
        </Stack>

        {/* Section A: Dataset Setup */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderLeft: `4px solid ${theme.palette.info.main}`,
            borderTopLeftRadius: 0,
            borderBottomLeftRadius: 0,
            boxShadow: 'none',
            transition: 'box-shadow 0.2s ease',
            '&:hover': { boxShadow: theme.shadows[1] },
          })}
        >
          <CardContent>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: 'info.main', letterSpacing: '0.03em' }}>
              Dataset Setup & Upload
            </Typography>
            <Stack spacing={2}>
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mb: 0.75 }}>
                  Dataset Name (optional – sent as X-Database-Name header)
                </Typography>
                <TextField
                  fullWidth
                  placeholder="e.g., campus_iot_logs_v1"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  error={!!datasetNameError}
                  helperText={datasetNameError}
                  size="small"
                />
              </Box>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
              <Stack direction="row" alignItems="center" spacing={2} flexWrap="wrap" useFlexGap>
                <Button
                  variant="outlined"
                  color="info"
                  onClick={handleAttachClick}
                  sx={{ flexShrink: 0 }}
                >
                  Attach file
                </Button>
                {selectedFile ? (
                  <Stack direction="row" alignItems="center" spacing={1}>
                    <Chip
                      label={selectedFile.name}
                      size="small"
                      onDelete={() => setSelectedFile(null)}
                      color="info"
                      variant="outlined"
                    />
                  </Stack>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No file attached
                  </Typography>
                )}
              </Stack>
              <Button
                variant="contained"
                color="info"
                startIcon={uploading ? <CircularProgress size={16} color="inherit" /> : <UploadFileRoundedIcon />}
                onClick={handleUploadToBackend}
                disabled={uploading || !selectedFile}
                sx={(theme) => ({
                  alignSelf: 'flex-start',
                  color: `${theme.palette.info.contrastText || '#fff'} !important`,
                  border: 'none !important',
                  boxShadow: 'none !important',
                  outline: 'none !important',
                  '&:focus, &:focus-visible': { outline: 'none !important', boxShadow: 'none !important', border: 'none !important' },
                  '& .MuiSvgIcon-root': { color: 'inherit' },
                })}
              >
                {uploading ? 'Uploading…' : 'Upload CSV'}
              </Button>
            </Stack>
          </CardContent>
        </Card>

        {/* Section B: Data Viewer */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderLeft: '4px solid hsl(173, 58%, 42%)',
            borderTopLeftRadius: 0,
            borderBottomLeftRadius: 0,
            boxShadow: 'none',
            transition: 'box-shadow 0.2s ease',
            '&:hover': { boxShadow: theme.shadows[1] },
          })}
        >
          <CardContent>
            <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems="center" sx={{ mb: 2, gap: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'hsl(173, 58%, 32%)', letterSpacing: '0.03em' }}>
                View Data
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap alignItems="center">
                <FormControl size="small" sx={{ minWidth: 140 }}>
                  <InputLabel id="rows-label">Rows to show</InputLabel>
                  <Select
                    labelId="rows-label"
                    value={viewLimit}
                    label="Rows to show"
                    onChange={(e) => handleViewLimitChange(Number(e.target.value))}
                    disabled={viewLoading}
                  >
                    <MenuItem value={500}>500</MenuItem>
                    <MenuItem value={1000}>1000</MenuItem>
                    <MenuItem value={2000}>2000</MenuItem>
                    <MenuItem value={5000}>5000</MenuItem>
                    <MenuItem value={10000}>10,000</MenuItem>
                  </Select>
                </FormControl>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={viewLoading ? <CircularProgress size={14} color="inherit" /> : <RefreshRoundedIcon />}
                  onClick={() => fetchViewData(viewLimit, 0)}
                  disabled={viewLoading}
                >
                  Refresh
                </Button>
                <FormControl size="small" sx={{ minWidth: 130 }}>
                  <InputLabel id="filter-label">Filter</InputLabel>
                  <Select
                    labelId="filter-label"
                    value={filterMode}
                    label="Filter"
                    onChange={(e) => setFilterMode(e.target.value as typeof filterMode)}
                  >
                    <MenuItem value="all">All</MenuItem>
                    <MenuItem value="training">Training</MenuItem>
                    <MenuItem value="testing">Testing</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  size="small"
                  placeholder="Search rows…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  slotProps={{ htmlInput: { 'aria-label': 'Search' } }}
                  sx={{ width: 180 }}
                />
              </Stack>
            </Stack>
            {viewTotalRows != null && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Showing {rows.length} of {viewTotalRows} rows from backend
              </Typography>
            )}
            <Box sx={{ height: 280, width: '100%', borderRadius: 1, overflow: 'hidden', border: '1px solid', borderColor: 'divider' }}>
              {viewLoading && rows.length === 0 ? (
                <Stack alignItems="center" justifyContent="center" sx={{ height: '100%', color: 'text.secondary' }}>
                  <CircularProgress size={32} sx={{ mb: 1 }} />
                  <Typography variant="body2">Loading view data…</Typography>
                </Stack>
              ) : filteredRows.length > 0 ? (
                <DataGrid
                  rows={filteredRows}
                  columns={columns}
                  initialState={{ pagination: { paginationModel: { pageSize: 100 } } }}
                  pageSizeOptions={[25, 50, 100]}
                  disableColumnResize
                  density="compact"
                />
              ) : (
                <Stack alignItems="center" justifyContent="center" sx={{ height: '100%', color: 'text.secondary' }}>
                  <SearchRoundedIcon sx={{ fontSize: 48, mb: 1, opacity: 0.5 }} />
                  <Typography variant="body2">
                    {viewLoading ? 'Loading…' : 'No data. Upload a CSV or check backend at http://localhost:8000.'}
                  </Typography>
                </Stack>
              )}
            </Box>
          </CardContent>
        </Card>

        {/* Section C: Dataset Actions */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderLeft: '4px solid hsl(280, 65%, 55%)',
            borderTopLeftRadius: 0,
            borderBottomLeftRadius: 0,
            boxShadow: 'none',
            transition: 'box-shadow 0.2s ease',
            '&:hover': { boxShadow: theme.shadows[1] },
          })}
        >
          <CardContent>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: 'hsl(280, 65%, 45%)', letterSpacing: '0.03em' }}>
              Dataset Actions
            </Typography>
            <Stack spacing={2}>
              {/* C1) Revalidate Dataset Button */}
              <Box>
                <Button
                  variant="contained"
                  color="warning"
                  startIcon={validating ? <CircularProgress size={16} color="inherit" /> : <RefreshRoundedIcon />}
                  onClick={handleRevalidate}
                  disabled={validating}
                  sx={(theme) => ({
                    alignSelf: 'flex-start',
                    color: `${theme.palette.warning.contrastText || '#1a1a1a'} !important`,
                    border: 'none !important',
                    boxShadow: 'none !important',
                    outline: 'none !important',
                    '&:focus, &:focus-visible': { outline: 'none !important', boxShadow: 'none !important', border: 'none !important' },
                    '& .MuiSvgIcon-root': { color: 'inherit' },
                  })}
                >
                  {validating ? 'Validating…' : 'Revalidate dataset'}
                </Button>
                {validationResult && (
                  <Alert
                    severity={validationResult.severity}
                    sx={{ mt: 1.5 }}
                    onClose={() => setValidationResult(null)}
                  >
                    {validationResult.message}
                  </Alert>
                )}
              </Box>

              {/* C2) Insert New Data */}
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mb: 0.75 }}>
                  Insert New Data
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                  Paste a CSV row matching columns: {rows.length > 0 ? Object.keys(rows[0] as object).filter((k) => k !== 'id').join(', ') : 'duration, protocol, bytes_sent, bytes_recv, packets, label'}
                </Typography>
                <Stack direction="row" spacing={1}>
                  <TextField
                    fullWidth
                    multiline
                    maxRows={3}
                    placeholder="Paste CSV row…"
                    value={insertText}
                    onChange={(e) => setInsertText(e.target.value)}
                    size="small"
                  />
                  <Button
                    variant="contained"
                    color="success"
                    startIcon={insertLoading ? <CircularProgress size={16} color="inherit" /> : <AddRoundedIcon />}
                    onClick={handleInsert}
                    disabled={insertLoading}
                    sx={(theme) => ({
                      alignSelf: 'flex-end',
                      color: `${theme.palette.success.contrastText || '#fff'} !important`,
                      border: 'none !important',
                      boxShadow: 'none !important',
                      outline: 'none !important',
                      '&:focus, &:focus-visible': { outline: 'none !important', boxShadow: 'none !important', border: 'none !important' },
                      '& .MuiSvgIcon-root': { color: 'inherit' },
                    })}
                  >
                    {insertLoading ? 'Inserting…' : 'Add Row'}
                  </Button>
                </Stack>
              </Box>

              {/* C3) Clear Database Table */}
              <Button
                variant="contained"
                color="error"
                startIcon={<DeleteSweepRoundedIcon />}
                onClick={() => rows.length === 0 ? setSnackbar({ open: true, message: 'Nothing to clear.', severity: 'info' }) : setClearConfirmOpen(true)}
                sx={(theme) => ({
                  alignSelf: 'flex-start',
                  color: `${theme.palette.error.contrastText || '#fff'} !important`,
                  border: 'none !important',
                  boxShadow: 'none !important',
                  outline: 'none !important',
                  '&:focus, &:focus-visible': { outline: 'none !important', boxShadow: 'none !important', border: 'none !important' },
                  '& .MuiSvgIcon-root': { color: 'inherit' },
                })}
              >
                Clear Database Table
              </Button>
            </Stack>
          </CardContent>
        </Card>

        {/* Section D: Model Configuration */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderLeft: '4px solid hsl(199, 89%, 48%)',
            borderTopLeftRadius: 0,
            borderBottomLeftRadius: 0,
            boxShadow: 'none',
            transition: 'box-shadow 0.2s ease',
            '&:hover': { boxShadow: theme.shadows[1] },
          })}
        >
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 2, flexWrap: 'wrap', gap: 0.5 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'hsl(199, 89%, 38%)', letterSpacing: '0.03em' }}>
                Model Configuration & Training
              </Typography>
            </Stack>
            <Stack spacing={2}>
              <Box>
                <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2 }}>
                  Start a new training session. The backend will automatically select relevant features.
                </Typography>
                <TextField
                  select
                  fullWidth
                  size="small"
                  label="Select Model Architecture"
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  SelectProps={{
                    displayEmpty: true,
                  }}
                >
                  <MenuItem value="Random Forest">Random Forest (rfV1)</MenuItem>
                  <MenuItem value="Isolation Forest">Isolation Forest</MenuItem>
                  <MenuItem value="Autoencoder">Autoencoder (MLP)</MenuItem>
                </TextField>
              </Box>
              <Button
                variant="contained"
                size="large"
                startIcon={training ? <CircularProgress size={20} color="inherit" /> : <PsychologyRoundedIcon />}
                onClick={handleTrain}
                disabled={training}
                sx={{
                  alignSelf: 'flex-start',
                  mt: 1,
                  backgroundColor: 'hsl(295, 65%, 52%) !important',
                  backgroundImage: 'none !important',
                  color: '#fff !important',
                  border: 'none !important',
                  boxShadow: 'none !important',
                  outline: 'none !important',
                  '&:hover:not(:disabled)': {
                    backgroundColor: 'hsl(295, 65%, 45%) !important',
                    backgroundImage: 'none !important',
                  },
                  '&:focus, &:focus-visible': { outline: 'none !important', boxShadow: 'none !important', border: 'none !important' },
                  '& .MuiSvgIcon-root': { color: 'inherit' },
                }}
              >
                {training ? 'Training…' : 'Train Model'}
              </Button>
            </Stack>
          </CardContent>
        </Card>

        {/* Section E: Metrics */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderLeft: '4px solid hsl(142, 76%, 48%)',
            borderTopLeftRadius: 0,
            borderBottomLeftRadius: 0,
            boxShadow: 'none',
            transition: 'box-shadow 0.2s ease',
            '&:hover': { boxShadow: theme.shadows[1] },
          })}
        >
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 2, flexWrap: 'wrap', gap: 0.5 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'hsl(142, 76%, 38%)', letterSpacing: '0.03em' }}>
                Model KPIs and metrics
              </Typography>
            </Stack>
            {metrics ? (
              <Stack spacing={2}>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  <Chip label={`Status: ${metrics.status}`} color="success" size="small" />
                  <Chip label={`Dataset: ${metrics.dataset}`} size="small" variant="outlined" />
                  <Chip label={`Features: ${metrics.features}`} size="small" variant="outlined" />
                </Stack>
                <Grid container spacing={2}>
                  {metrics.accuracy != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">Accuracy</Typography>
                        <Typography variant="h6">{(Number(metrics.accuracy) * 100).toFixed(1)}%</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.precision != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">Precision</Typography>
                        <Typography variant="h6">{(Number(metrics.precision) * 100).toFixed(1)}%</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.recall != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">Recall</Typography>
                        <Typography variant="h6">{(Number(metrics.recall) * 100).toFixed(1)}%</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.f1 != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">F1</Typography>
                        <Typography variant="h6">{(Number(metrics.f1) * 100).toFixed(1)}%</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.anomalyRate != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">Anomaly Rate</Typography>
                        <Typography variant="h6">{String(metrics.anomalyRate)}</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.flagged != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">Flagged</Typography>
                        <Typography variant="h6">{String(metrics.flagged)}</Typography>
                      </Box>
                    </Grid>
                  )}
                </Grid>
                <Typography variant="caption" color="text.secondary">
                  Trained: {new Date(String(metrics.timestamp)).toLocaleString()}
                </Typography>
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Train a model to see metrics here.
              </Typography>
            )}
          </CardContent>
        </Card>

        {/* Section F: Model Inference (Prediction) */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderLeft: '4px solid hsl(48, 89%, 50%)',
            borderTopLeftRadius: 0,
            borderBottomLeftRadius: 0,
            boxShadow: 'none',
            transition: 'box-shadow 0.2s ease',
            '&:hover': { boxShadow: theme.shadows[1] },
          })}
        >
          <CardContent>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: 'hsl(48, 89%, 40%)', letterSpacing: '0.03em' }}>
              Model Inference (In-browser Prediction)
            </Typography>
            <Stack spacing={2}>
              <Typography variant="body2" color="text.secondary">
                Use your trained <strong>{metrics ? metrics.model_type : 'model'}</strong> to classify netflow traffic.
              </Typography>

              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  color="warning"
                  onClick={() => handlePredict(true)}
                  disabled={predicting || !metrics}
                  startIcon={predicting ? <CircularProgress size={16} /> : <SearchRoundedIcon />}
                >
                  Run on Testing Set
                </Button>
                <Box sx={{ flexGrow: 1 }} />
              </Stack>

              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mb: 0.75 }}>
                  Manual Prediction (CSV row)
                </Typography>
                <Stack direction="row" spacing={1}>
                  <TextField
                    fullWidth
                    size="small"
                    placeholder="Paste flow data here..."
                    value={predictionInput}
                    onChange={(e) => setPredictionInput(e.target.value)}
                  />
                  <Button
                    variant="outlined"
                    onClick={() => handlePredict(false)}
                    disabled={predicting || !metrics || !predictionInput}
                  >
                    Predict
                  </Button>
                </Stack>
              </Box>

              {predictionResults.length > 0 && (
                <Box sx={{ mt: 2, p: 2, borderRadius: 1, bgcolor: 'action.hover', border: '1px solid', borderColor: 'divider' }}>
                  <Typography variant="subtitle2" gutterBottom>Prediction Results (Recent)</Typography>
                  <Stack spacing={1}>
                    {predictionResults.slice(0, 5).map((res, idx) => (
                      <Stack key={idx} direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2">Sample #{res.index + 1}</Typography>
                        <Chip
                          label={res.label.toUpperCase()}
                          size="small"
                          color={res.label === 'anomaly' ? 'error' : 'success'}
                          sx={{ fontWeight: 700 }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          {(res.confidence * 100).toFixed(1)}% Confidence
                        </Typography>
                      </Stack>
                    ))}
                    {predictionResults.length > 5 && (
                      <Typography variant="caption" color="text.secondary" align="center">
                        + {predictionResults.length - 5} more results in buffer
                      </Typography>
                    )}
                  </Stack>
                </Box>
              )}
            </Stack>
          </CardContent>
        </Card>
      </Stack>

      {/* Clear Confirm Dialog */}
      <Dialog open={clearConfirmOpen} onClose={() => !clearLoading && setClearConfirmOpen(false)}>
        <DialogTitle>Clear Database Table?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This will delete all rows for this dataset. Continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearConfirmOpen(false)} disabled={clearLoading}>Cancel</Button>
          <Button onClick={handleClearConfirm} color="error" variant="contained" disabled={clearLoading} startIcon={clearLoading ? <CircularProgress size={16} color="inherit" /> : undefined}>
            {clearLoading ? 'Clearing…' : 'Confirm'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
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
