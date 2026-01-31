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
  const [modelType, setModelType] = React.useState('');
  const [training, setTraining] = React.useState(false);
  const [metrics, setMetrics] = React.useState<Record<string, unknown> | null>(null);
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
        const res = await fetch(`${API_BASE}/api/view?${params}`, { headers });
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
      const res = await fetch(`${API_BASE}/api/health`, { headers });
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
    const interval = setInterval(() => fetchApiHealth(true), 2000);
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

  const handleTrain = () => {
    if (!modelName.trim()) {
      setModelNameError('Model name is required');
      return;
    }
    if (!selectedDatasetId || selectedFeatures.length === 0 || !modelType) {
      setSnackbar({ open: true, message: 'Please select dataset, at least one feature, and model type', severity: 'error' });
      return;
    }
    setModelNameError('');
    setTraining(true);
    setTimeout(() => {
      setTraining(false);
      setMetrics({
        accuracy: 0.94,
        precision: 0.92,
        recall: 0.89,
        f1: 0.9,
        anomalyRate: modelType.includes('Anomaly') ? '2.3%' : null,
        flagged: modelType.includes('Anomaly') ? 23 : null,
        status: 'success',
        timestamp: new Date().toISOString(),
        dataset: datasets.find((d) => d.id === selectedDatasetId)?.name,
        features: selectedFeatures.join(', '),
      });
      setSnackbar({ open: true, message: 'Model trained successfully', severity: 'success' });
    }, 2500);
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
      <Stack sx={{ mb: 3 }}>
        <Typography component="h1" variant="h5" sx={{ fontWeight: 600 }}>
          Model Handling
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Upload a dataset, validate it, then configure and train a model.
        </Typography>
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
                borderRadius: 1,
                bgcolor: 'action.hover',
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
              })}
            >
              <CardContent>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: 'info.main' }}>
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
                borderLeft: `4px solid ${theme.palette.primary.main}`,
                borderTopLeftRadius: 0,
                borderBottomLeftRadius: 0,
              })}
            >
              <CardContent>
                <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems="center" sx={{ mb: 2, gap: 2 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'primary.main' }}>
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
                <Box sx={{ height: 280, width: '100%' }}>
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
              sx={{
                borderLeft: '4px solid hsl(280, 65%, 55%)',
                borderTopLeftRadius: 0,
                borderBottomLeftRadius: 0,
              }}
            >
              <CardContent>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: 'hsl(280, 65%, 45%)' }}>
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
              sx={{
                borderLeft: '4px solid hsl(199, 89%, 48%)',
                borderTopLeftRadius: 0,
                borderBottomLeftRadius: 0,
              }}
            >
              <CardContent>
                <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'hsl(199, 89%, 38%)' }}>
                    Model Configuration & Training
                  </Typography>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'red' }}>
                    (UNDER CONSTRUCTION)
                  </Typography>
                </Stack>
                <Stack spacing={2}>
                  <Box>
                    <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mb: 0.75 }}>
                      Model Name
                    </Typography>
                    <TextField
                      fullWidth
                      placeholder="e.g., anomaly_rf_v1"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      error={!!modelNameError}
                      helperText={modelNameError}
                      size="small"
                    />
                  </Box>
                  <Box>
                    <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mb: 0.75 }}>
                      Select Dataset to train Model
                    </Typography>
                    <TextField
                      select
                      fullWidth
                      size="small"
                      value={selectedDatasetId}
                      onChange={(e) => setSelectedDatasetId(e.target.value)}
                      SelectProps={{
                      displayEmpty: true,
                      renderValue: (v) => {
                        const d = datasets.find((x) => x.id === v);
                        return d ? d.name : 'None';
                      },
                      sx: {
                        '& .MuiSelect-select': {
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        },
                      },
                    }}
                  >
                    <MenuItem value="">None</MenuItem>
                    {datasets.map((d) => (
                      <MenuItem key={d.id} value={d.id}>{d.name}</MenuItem>
                    ))}
                  </TextField>
                  </Box>
                  <Box>
                    <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mb: 0.75 }}>
                      Choose Datafields to Model
                    </Typography>
                    <FormGroup row>
                      {featureColumns.map((f) => (
                        <FormControlLabel
                          key={f}
                          control={
                            <Checkbox
                              checked={selectedFeatures.includes(f)}
                              onChange={handleFeatureToggle(f)}
                              size="small"
                            />
                          }
                          label={f}
                        />
                      ))}
                    </FormGroup>
                    <Button size="small" color="inherit" onClick={() => setSelectedFeatures(featureColumns)} sx={{ mt: 0.5, color: 'text.secondary' }}>
                      Select all
                    </Button>
                  </Box>
                  <Box>
                    <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mb: 0.75 }}>
                      Select Model Type
                    </Typography>
                    <TextField
                      select
                      fullWidth
                      size="small"
                      value={modelType}
                      onChange={(e) => setModelType(e.target.value)}
                      SelectProps={{
                      displayEmpty: true,
                      renderValue: (v: unknown) => (v ? String(v) : 'Choose model type'),
                      sx: {
                        '& .MuiSelect-select': {
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        },
                      },
                    }}
                  >
                    <MenuItem value="">Choose model type</MenuItem>
                    {MOCK_MODEL_TYPES.map((t) => (
                      <MenuItem key={t} value={t}>{t}</MenuItem>
                    ))}
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
              sx={{
                borderLeft: '4px solid hsl(142, 76%, 48%)',
                borderTopLeftRadius: 0,
                borderBottomLeftRadius: 0,
              }}
            >
              <CardContent>
                <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'hsl(142, 76%, 38%)' }}>
                    Model KPIs and metrics
                  </Typography>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'red' }}>
                    (UNDER CONSTRUCTION)
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
                          <Typography variant="caption" color="text.secondary">Accuracy</Typography>
                          <Typography variant="h6">{(Number(metrics.accuracy) * 100).toFixed(1)}%</Typography>
                        </Grid>
                      )}
                      {metrics.precision != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Typography variant="caption" color="text.secondary">Precision</Typography>
                          <Typography variant="h6">{(Number(metrics.precision) * 100).toFixed(1)}%</Typography>
                        </Grid>
                      )}
                      {metrics.recall != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Typography variant="caption" color="text.secondary">Recall</Typography>
                          <Typography variant="h6">{(Number(metrics.recall) * 100).toFixed(1)}%</Typography>
                        </Grid>
                      )}
                      {metrics.f1 != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Typography variant="caption" color="text.secondary">F1</Typography>
                          <Typography variant="h6">{(Number(metrics.f1) * 100).toFixed(1)}%</Typography>
                        </Grid>
                      )}
                      {metrics.anomalyRate != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Typography variant="caption" color="text.secondary">Anomaly Rate</Typography>
                          <Typography variant="h6">{String(metrics.anomalyRate)}</Typography>
                        </Grid>
                      )}
                      {metrics.flagged != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Typography variant="caption" color="text.secondary">Flagged</Typography>
                          <Typography variant="h6">{String(metrics.flagged)}</Typography>
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
