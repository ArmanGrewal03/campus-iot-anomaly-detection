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

export default function ModelPage() {
  const [datasetName, setDatasetName] = React.useState('');
  const [datasetNameError, setDatasetNameError] = React.useState('');
  const [uploading, setUploading] = React.useState(false);
  const [datasets, setDatasets] = React.useState<{ id: string; name: string }[]>([]);
  const [rows, setRows] = React.useState<Record<string, unknown>[]>([]);
  const [filterMode, setFilterMode] = React.useState<'all' | 'training' | 'testing'>('all');
  const [searchQuery, setSearchQuery] = React.useState('');
  const [validating, setValidating] = React.useState(false);
  const [insertText, setInsertText] = React.useState('');
  const [clearConfirmOpen, setClearConfirmOpen] = React.useState(false);
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

  const handleUploadClick = () => {
    if (!datasetName.trim()) {
      setDatasetNameError('Dataset name is required');
      setSnackbar({ open: true, message: 'Enter a dataset name first.', severity: 'info' });
      return;
    }
    setDatasetNameError('');
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !datasetName.trim()) return;
    setUploading(true);
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const text = ev.target?.result as string;
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',').map((h) => h.trim());
        const parsed = lines.slice(1).map((line, i) => {
          const values = line.split(',').map((v) => v.trim());
          const row: Record<string, unknown> = { id: i + 1 };
          headers.forEach((h, j) => { row[h] = values[j] ?? ''; });
          return row;
        });
        const id = `ds-${Date.now()}`;
        setDatasets((d) => [...d, { id, name: datasetName.trim() }]);
        setSelectedDatasetId(id);
        setRows(parsed.length > 0 ? parsed : MOCK_INITIAL_ROWS);
        setSnackbar({ open: true, message: `Uploaded ${parsed.length || MOCK_INITIAL_ROWS.length} rows`, severity: 'success' });
      } catch {
        setRows(MOCK_INITIAL_ROWS);
        setDatasets((d) => [...d, { id: `ds-${Date.now()}`, name: datasetName.trim() }]);
        setSelectedDatasetId(`ds-${Date.now()}`);
        setSnackbar({ open: true, message: 'Parsing failed. Using sample data.', severity: 'info' });
      }
      setUploading(false);
    };
    reader.onerror = () => {
      setSnackbar({ open: true, message: 'Failed to read file', severity: 'error' });
      setUploading(false);
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const [validationResult, setValidationResult] = React.useState<{ message: string; severity: 'success' | 'warning' } | null>(null);

  const handleRevalidate = () => {
    if (rows.length === 0) {
      setSnackbar({ open: true, message: 'No dataset loaded. Upload a CSV first.', severity: 'info' });
      return;
    }
    setValidating(true);
    setValidationResult(null);
    setTimeout(() => {
      setValidating(false);
      // Simulate validation outcome (frontend only)
      const passed = Math.random() > 0.2;
      if (passed) {
        setValidationResult({ message: 'Validation: ✅ Passed', severity: 'success' });
        setSnackbar({ open: true, message: 'Validation: ✅ Passed', severity: 'success' });
      } else {
        const rejected = Math.floor(Math.random() * 20) + 1;
        const msg = `⚠ ${rejected} rows rejected`;
        setValidationResult({ message: msg, severity: 'warning' });
        setSnackbar({ open: true, message: msg, severity: 'warning' });
      }
    }, 800);
  };

  const handleInsert = () => {
    if (!insertText.trim()) {
      setSnackbar({ open: true, message: 'Paste a CSV row first.', severity: 'info' });
      return;
    }
    if (rows.length === 0) {
      setSnackbar({ open: true, message: 'No dataset loaded. Upload a CSV first.', severity: 'info' });
      return;
    }
    const parts = insertText.trim().split(',').map((p) => p.trim());
    const cols = rows.length > 0 ? Object.keys(rows[0] as object).filter((k) => k !== 'id') : DEFAULT_FEATURE_COLUMNS;
    if (parts.length >= cols.length) {
      const newRow: Record<string, unknown> = { id: rows.length + 1 };
      cols.forEach((c, i) => { newRow[c] = parts[i] ?? ''; });
      setRows((r) => [...r, newRow]);
      setInsertText('');
      setSnackbar({ open: true, message: 'Row added successfully', severity: 'success' });
    } else {
      setSnackbar({ open: true, message: `Invalid format. Use: ${cols.join(', ')}`, severity: 'error' });
    }
  };

  const handleClearConfirm = () => {
    setRows([]);
    setMetrics(null);
    setValidationResult(null);
    setClearConfirmOpen(false);
    setSnackbar({ open: true, message: 'Database table cleared', severity: 'info' });
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
                      Dataset Name
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
                  <Button
                    variant="contained"
                    color="info"
                    startIcon={uploading ? <CircularProgress size={16} color="inherit" /> : <UploadFileRoundedIcon />}
                    onClick={handleUploadClick}
                    disabled={uploading}
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
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
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
                <Box sx={{ height: 280, width: '100%' }}>
                  {filteredRows.length > 0 ? (
                    <DataGrid
                      rows={filteredRows}
                      columns={columns}
                      initialState={{ pagination: { paginationModel: { pageSize: 5 } } }}
                      pageSizeOptions={[5, 10, 25]}
                      disableColumnResize
                      density="compact"
                    />
                  ) : (
                    <Stack alignItems="center" justifyContent="center" sx={{ height: '100%', color: 'text.secondary' }}>
                      <SearchRoundedIcon sx={{ fontSize: 48, mb: 1, opacity: 0.5 }} />
                      <Typography variant="body2">No dataset loaded yet. Upload a CSV to view data.</Typography>
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
                        startIcon={<AddRoundedIcon />}
                        onClick={handleInsert}
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
                        Add Row
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
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: 'hsl(199, 89%, 38%)' }}>
                  Model Configuration & Training
                </Typography>
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
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: 'hsl(142, 76%, 38%)' }}>
                  Model KPIs and metrics
                </Typography>
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
      <Dialog open={clearConfirmOpen} onClose={() => setClearConfirmOpen(false)}>
        <DialogTitle>Clear Database Table?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This will delete all rows for this dataset. Continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearConfirmOpen(false)}>Cancel</Button>
          <Button onClick={handleClearConfirm} color="error" variant="contained">
            Confirm
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
