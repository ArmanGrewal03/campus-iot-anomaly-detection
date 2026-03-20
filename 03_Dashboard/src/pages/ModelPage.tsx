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
import Autocomplete from '@mui/material/Autocomplete';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import UploadFileRoundedIcon from '@mui/icons-material/UploadFileRounded';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import AddRoundedIcon from '@mui/icons-material/AddRounded';
import DeleteSweepRoundedIcon from '@mui/icons-material/DeleteSweepRounded';
import PsychologyRoundedIcon from '@mui/icons-material/PsychologyRounded';
import SearchRoundedIcon from '@mui/icons-material/SearchRounded';
import ChevronLeftRoundedIcon from '@mui/icons-material/ChevronLeftRounded';
import ChevronRightRoundedIcon from '@mui/icons-material/ChevronRightRounded';
import { DataGrid } from '@mui/x-data-grid';
import { LineChart } from '@mui/x-charts/LineChart';

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

const GATEWAY_BASE = 'http://127.0.0.1:8003'; // API Gateway
const API_BASE = `${GATEWAY_BASE}`; // Data Ingestion Service via Gateway
const MODEL_API_BASE = `${GATEWAY_BASE}`; // Model Service via Gateway

export default function ModelPage() {
  const playCompletionSound = React.useCallback(() => {
    try {
      const AudioContextClass =
        window.AudioContext ||
        ((window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext);
      if (!AudioContextClass) return;
      const audioCtx = new AudioContextClass();
      const now = audioCtx.currentTime;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.setValueAtTime(1175, now + 0.12);
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(0.12, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.28);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.3);
    } catch (err) {
      console.debug('Audio notification unavailable:', err);
    }
  }, []);

  const [datasetName, setDatasetName] = React.useState('');
  const [datasetNameError, setDatasetNameError] = React.useState('');
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [uploading, setUploading] = React.useState(false);
  const [rows, setRows] = React.useState<Record<string, unknown>[]>([]);
  const [viewLoading, setViewLoading] = React.useState(false);
  const [viewTotalRows, setViewTotalRows] = React.useState<number | null>(null);
  const [filterMode, setFilterMode] = React.useState<'all' | 'training' | 'testing'>('all');
  const [searchQuery, setSearchQuery] = React.useState('');
  const [paginationModel, setPaginationModel] = React.useState({ page: 0, pageSize: 100 });
  const [validating, setValidating] = React.useState(false);
  const [insertText, setInsertText] = React.useState('');
  const [clearConfirmOpen, setClearConfirmOpen] = React.useState(false);
  const [clearLoading, setClearLoading] = React.useState(false);
  const [insertLoading, setInsertLoading] = React.useState(false);
  const [modelName, setModelName] = React.useState('');
  const [modelNameError, setModelNameError] = React.useState('');
  /** Single source of truth: selected dataset name (used in View Data, Dataset Actions, Training, Stats, Test) */
  const [selectedDataset, setSelectedDataset] = React.useState<string>('');
  const [selectedFeatures, setSelectedFeatures] = React.useState<string[]>([]);
  const [modelType, setModelType] = React.useState('');
  const [training, setTraining] = React.useState(false);
  const [metrics, setMetrics] = React.useState<any>(null);
  const [predictionResults, setPredictionResults] = React.useState<any[]>([]);
  const [predicting, setPredicting] = React.useState(false);
  const [predictionInput, setPredictionInput] = React.useState('');
  const [datasetStats, setDatasetStats] = React.useState<any>(null);
  const [typeStats, setTypeStats] = React.useState<any>(null);
  const [statsLoading, setStatsLoading] = React.useState(false);
  const [availableDatasets, setAvailableDatasets] = React.useState<string[]>([]);
  const [datasetsLoading, setDatasetsLoading] = React.useState(false);
  const [selectedValidateDataset, setSelectedValidateDataset] = React.useState<string>('');
  const [label0Percent, setLabel0Percent] = React.useState<string>('');
  const [label1Percent, setLabel1Percent] = React.useState<string>('');
  const [trainingPercent, setTrainingPercent] = React.useState<string>('80');
  const [testingPercent, setTestingPercent] = React.useState<string>('20');
  const [selectedStatsDataset, setSelectedStatsDataset] = React.useState<string>('');
  const [availableFields, setAvailableFields] = React.useState<string[]>([]);
  const [fieldsLoading, setFieldsLoading] = React.useState(false);
  const [availableModels, setAvailableModels] = React.useState<any[]>([]);
  const [modelsLoading, setModelsLoading] = React.useState(false);
  const [selectedTestModel, setSelectedTestModel] = React.useState<string>('');
  const [testing, setTesting] = React.useState(false);
  const [testResults, setTestResults] = React.useState<any>(null);
  const [modelStatuses, setModelStatuses] = React.useState<Record<string, any>>({});
  const [modelMetrics, setModelMetrics] = React.useState<Record<string, any>>({});
  const [currentModelIndex, setCurrentModelIndex] = React.useState(0);
  const [modelDetailsLoading, setModelDetailsLoading] = React.useState(false);
  const [modelTypes, setModelTypes] = React.useState<Array<{ model_type: string; path: string; files: Array<{ name: string; size: number; modified: string }> }>>([]);
  const [modelTypesLoading, setModelTypesLoading] = React.useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = React.useState(false);
  const [deleting, setDeleting] = React.useState(false);
  const [modelToDelete, setModelToDelete] = React.useState<string | null>(null);


  const [snackbar, setSnackbar] = React.useState<{ open: boolean; message: string; severity: 'success' | 'error' | 'info' | 'warning' }>({
    open: false,
    message: '',
    severity: 'success',
  });
  const [apiHealth, setApiHealth] = React.useState<'healthy' | 'unhealthy' | 'loading' | null>(null);
  const [apiHealthDetail, setApiHealthDetail] = React.useState<{ service?: string; database?: string; timestamp?: string } | null>(null);

  // Client-side search filtering (backend doesn't support search)
  // Note: This only searches within the current page of data
  const filteredRows = React.useMemo(() => {
    // For server-side pagination, we don't filter client-side
    // The search should be handled server-side, but for now we'll only filter if there's a search query
    // and we'll adjust rowCount accordingly
    if (!searchQuery) return rows;
      const q = searchQuery.toLowerCase();
    return rows.filter((r) => Object.values(r).some((v) => String(v).toLowerCase().includes(q)));
  }, [rows, searchQuery]);

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
    if (uploading) return; // Prevent double-submit
    setUploading(true);
    setDatasetNameError('');
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const headers: Record<string, string> = {};
      const nameToUse = datasetName.trim();
      if (!nameToUse) {
        setSnackbar({ open: true, message: 'Enter a dataset name before uploading.', severity: 'warning' });
        setUploading(false);
        return;
      }
      headers['dataset_name'] = nameToUse;
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
      setPaginationModel({ page: 0, pageSize: paginationModel.pageSize });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Network error. Is the backend running at http://localhost:8000?';
      setSnackbar({ open: true, message, severity: 'error' });
    } finally {
      setUploading(false);
    }
  };

  const fetchTables = React.useCallback(async () => {
    setDatasetsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/tables`);
      const json = await res.json() as { status?: string; tables?: string[] };
      if (res.ok && json.status === 'success' && json.tables) {
        // Extract dataset names from table names (format: csv_data_{dataset_name})
        const datasetNames = json.tables
          .filter(table => table.startsWith('csv_data_'))
          .map(table => table.replace(/^csv_data_/, ''));
        setAvailableDatasets(datasetNames);
        // Keep current selection if it still exists; otherwise select first (one selection for all sections)
        setSelectedDataset((prev) => {
          if (datasetNames.length === 0) return '';
          if (prev && datasetNames.includes(prev)) return prev;
          return datasetNames[0];
        });
      }
    } catch (err) {
      console.error('Failed to fetch tables:', err);
      setSnackbar({ open: true, message: 'Failed to fetch dataset list.', severity: 'error' });
    } finally {
      setDatasetsLoading(false);
    }
  }, []);

  React.useEffect(() => {
    fetchTables();
  }, [fetchTables]);

  const fetchFields = React.useCallback(async (datasetName: string) => {
    if (!datasetName.trim()) {
      setAvailableFields([]);
      return;
    }
    setFieldsLoading(true);
    try {
      const headers: Record<string, string> = {
        'dataset_name': datasetName.trim(),
      };
      const res = await fetch(`${API_BASE}/fields`, { headers });
      const json = await res.json() as { status?: string; fields?: string[]; detail?: string };
      if (res.ok && json.status === 'success' && json.fields) {
        // Filter out label, id, attack_cat as they shouldn't be included as features
        const filteredFields = json.fields.filter(
          field => !['label', 'id', 'attack_cat'].includes(field.toLowerCase())
        );
        setAvailableFields(filteredFields);
      } else {
        setAvailableFields([]);
        if (json.detail) {
          setSnackbar({ open: true, message: `Failed to fetch fields: ${json.detail}`, severity: 'warning' });
        }
      }
    } catch (err) {
      console.error('Failed to fetch fields:', err);
      setAvailableFields([]);
      setSnackbar({ open: true, message: 'Failed to fetch field list.', severity: 'error' });
    } finally {
      setFieldsLoading(false);
    }
  }, []);

  // Fetch fields when dataset changes
  React.useEffect(() => {
    if (selectedDataset) {
      fetchFields(selectedDataset);
    } else {
      setAvailableFields([]);
    }
  }, [selectedDataset, fetchFields]);

  const fetchDatasetStats = React.useCallback(async (datasetName: string) => {
    if (!datasetName.trim()) {
      setDatasetStats(null);
      setTypeStats(null);
      return;
    }
    setStatsLoading(true);
    try {
      const headers: Record<string, string> = {
        'dataset_name': datasetName.trim(),
      };
      
      // Fetch both stats and type-stats in parallel
      const [statsRes, typeStatsRes] = await Promise.all([
        fetch(`${API_BASE}/stats`, { headers }),
        fetch(`${API_BASE}/type-stats`, { headers }),
      ]);

      const statsJson = await statsRes.json() as any;
      const typeStatsJson = await typeStatsRes.json() as any;

      if (statsRes.ok && !statsJson.error) {
        setDatasetStats(statsJson);
      } else {
        setDatasetStats(null);
      }

      if (typeStatsRes.ok && typeStatsJson.type_distribution) {
        setTypeStats(typeStatsJson);
      } else {
        setTypeStats(null);
      }
    } catch (err) {
      console.error('Failed to fetch dataset stats:', err);
      setDatasetStats(null);
      setTypeStats(null);
    } finally {
      setStatsLoading(false);
    }
  }, []);

  // Fetch stats when dataset changes or when metrics are updated
  React.useEffect(() => {
    const datasetToUse = metrics?.dataset || selectedDataset;
    if (datasetToUse) {
      fetchDatasetStats(datasetToUse);
    }
  }, [selectedDataset, metrics?.dataset, fetchDatasetStats]);

  const fetchModels = React.useCallback(async () => {
    setModelsLoading(true);
    try {
      const res = await fetch(`${MODEL_API_BASE}/models`);
      const json = await res.json() as { status?: string; models?: any[]; total_models?: number; detail?: string };
      
      console.log('Models API response:', json);
      console.log('Response status:', res.ok, 'JSON status:', json.status, 'Models array:', json.models);
      
      if (res.ok && json.status === 'success') {
        if (json.models && Array.isArray(json.models) && json.models.length > 0) {
          console.log(`Setting ${json.models.length} models:`, json.models);
          setAvailableModels(json.models);
        } else {
          console.warn('Models array is empty or invalid:', json.models);
          setAvailableModels([]);
        }
      } else {
        console.warn('Failed to fetch models - response not OK or status not success:', {
          resOk: res.ok,
          status: json.status,
          detail: json.detail
        });
        setAvailableModels([]);
        if (json.detail) {
          console.warn('Error detail:', json.detail);
        }
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
      setAvailableModels([]);
    } finally {
      setModelsLoading(false);
    }
  }, []);

  React.useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Fetch model types
  const fetchModelTypes = React.useCallback(async () => {
    setModelTypesLoading(true);
    try {
      const res = await fetch(`${MODEL_API_BASE}/model-types`);
      const json = await res.json() as { 
        status?: string; 
        model_types?: Array<{ model_type: string; path: string; files: Array<{ name: string; size: number; modified: string }> }>; 
        total_model_types?: number;
        detail?: string;
      };
      
      if (res.ok && json.status === 'success') {
        if (json.model_types && Array.isArray(json.model_types)) {
          setModelTypes(json.model_types);
          // Set first model type as default if none selected
          if (!modelType && json.model_types.length > 0) {
            setModelType(json.model_types[0].model_type);
          }
        } else {
          setModelTypes([]);
        }
      } else {
        setModelTypes([]);
        if (json.detail) {
          console.warn('Failed to fetch model types:', json.detail);
        }
      }
    } catch (err) {
      console.error('Failed to fetch model types:', err);
      setModelTypes([]);
    } finally {
      setModelTypesLoading(false);
    }
  }, []);

  React.useEffect(() => {
    fetchModelTypes();
  }, [fetchModelTypes]);

  // Update modelType when modelTypes are loaded
  React.useEffect(() => {
    if (modelTypes.length > 0 && !modelType) {
      setModelType(modelTypes[0].model_type);
    }
  }, [modelTypes, modelType]);

  // Fetch status and metrics for all models
  const fetchModelDetails = React.useCallback(async (modelName: string) => {
    if (!modelName) return;
    
    try {
      const headers: Record<string, string> = {
        'model_name': modelName,
      };

      const [statusRes, metricsRes] = await Promise.all([
        fetch(`${MODEL_API_BASE}/model/status`, { headers }),
        fetch(`${MODEL_API_BASE}/model/metrics`, { headers }).catch(() => null), // Metrics might not exist
      ]);

      const statusJson = await statusRes.json();
      if (statusRes.ok) {
        setModelStatuses((prev) => ({ ...prev, [modelName]: statusJson }));
      }

      if (metricsRes && metricsRes.ok) {
        const metricsJson = await metricsRes.json();
        // Debug logging for loss_history
        if (metricsJson.training_params?.loss_history) {
          console.log(`Model ${modelName} has loss_history:`, {
            length: metricsJson.training_params.loss_history.length,
            firstItem: metricsJson.training_params.loss_history[0],
            lastItem: metricsJson.training_params.loss_history[metricsJson.training_params.loss_history.length - 1]
          });
        } else {
          console.log(`Model ${modelName} does not have loss_history in training_params`);
        }
        setModelMetrics((prev) => ({ ...prev, [modelName]: metricsJson }));
      }
    } catch (err) {
      console.error(`Failed to fetch details for model ${modelName}:`, err);
    }
  }, []);

  // Fetch details for all models when models list changes
  React.useEffect(() => {
    if (availableModels.length > 0) {
      setModelDetailsLoading(true);
      Promise.all(availableModels.map((model) => fetchModelDetails(model.model_name)))
        .finally(() => setModelDetailsLoading(false));
    }
  }, [availableModels, fetchModelDetails]);

  const currentModel = availableModels.length > 0 && currentModelIndex < availableModels.length 
    ? availableModels[currentModelIndex] 
    : null;
  const currentModelStatus = currentModel ? modelStatuses[currentModel.model_name] : null;
  const currentModelMetrics = currentModel ? modelMetrics[currentModel.model_name] : null;

  const handlePreviousModel = () => {
    setCurrentModelIndex((prev) => (prev > 0 ? prev - 1 : availableModels.length - 1));
  };

  const handleNextModel = () => {
    setCurrentModelIndex((prev) => (prev < availableModels.length - 1 ? prev + 1 : 0));
  };

  const handleDeleteClick = () => {
    if (currentModel) {
      setModelToDelete(currentModel.model_name);
      setDeleteConfirmOpen(true);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!modelToDelete) return;
    
    setDeleting(true);
    try {
      const res = await fetch(`${MODEL_API_BASE}/models/${encodeURIComponent(modelToDelete)}`, {
        method: 'DELETE',
      });
      
      const json = await res.json() as { status?: string; message?: string; detail?: string };
      
      if (res.ok && json.status === 'success') {
        setSnackbar({
          open: true,
          message: json.message || `Model "${modelToDelete}" deleted successfully`,
          severity: 'success',
        });
        
        // Refresh models list
        await fetchModels();
        
        // Adjust current index if needed
        if (currentModelIndex >= availableModels.length - 1 && currentModelIndex > 0) {
          setCurrentModelIndex(currentModelIndex - 1);
        } else if (availableModels.length === 1) {
          setCurrentModelIndex(0);
        }
      } else {
        throw new Error(json.detail || json.message || 'Failed to delete model');
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to delete model.';
      setSnackbar({ open: true, message: msg, severity: 'error' });
    } finally {
      setDeleting(false);
      setDeleteConfirmOpen(false);
      setModelToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteConfirmOpen(false);
    setModelToDelete(null);
  };

  const handleTest = async () => {
    if (!selectedTestModel.trim()) {
      setSnackbar({ open: true, message: 'Please select a model to test.', severity: 'warning' });
      return;
    }

    const testDataset = selectedDataset.trim();
    if (!testDataset) {
      setSnackbar({ open: true, message: 'Please select a dataset to test on.', severity: 'warning' });
      return;
    }

    setTesting(true);
    setTestResults(null);
    const testStartMs = performance.now();

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'model_name': selectedTestModel.trim(),
      };

      // dataset_name is optional for /test endpoint
      if (testDataset) {
        headers['dataset_name'] = testDataset;
      }

      const res = await fetch(`${MODEL_API_BASE}/test`, {
        method: 'POST',
        headers,
        body: JSON.stringify({}),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Testing failed');
      }

      setTestResults(data);
      playCompletionSound();
      const testDurationSeconds =
        typeof data.testing_duration_seconds === 'number'
          ? data.testing_duration_seconds
          : Number(((performance.now() - testStartMs) / 1000).toFixed(3));
      setSnackbar({ 
        open: true, 
        message: `Model "${selectedTestModel.trim()}" tested successfully on dataset "${testDataset}" in ${testDurationSeconds.toFixed(2)}s`, 
        severity: 'success' 
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to test model.';
      setSnackbar({ open: true, message: msg, severity: 'error' });
    } finally {
      setTesting(false);
    }
  };

  const fetchViewData = React.useCallback(
    async (limit: number, offset: number) => {
      if (!selectedDataset.trim()) {
        setSnackbar({ open: true, message: 'Please select a dataset to view.', severity: 'warning' });
        return;
      }
      setViewLoading(true);
      // Don't reset total_rows - keep it to maintain pagination state
      try {
        const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
        const headers: Record<string, string> = {};
        headers['dataset_name'] = selectedDataset.trim();
        
        // Determine which endpoint to call based on filterMode
        // Note: These are GET endpoints from Data Ingestion Service, not POST /train from Model Service
        let endpoint = '/view';
        if (filterMode === 'training') {
          endpoint = '/training';  // GET endpoint from Data Ingestion Service
        } else if (filterMode === 'testing') {
          endpoint = '/testing';  // GET endpoint from Data Ingestion Service
        }
        
        const res = await fetch(`${API_BASE}${endpoint}?${params}`, { headers });
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
    [selectedDataset, filterMode]
  );

  // Track previous dataset and filter to detect changes (for resetting pagination)
  const prevDatasetRef = React.useRef<string | null>(null);
  const prevFilterRef = React.useRef<string | null>(null);
  
  // Reset to page 0 ONLY when dataset or filter actually changes (not on pagination changes)
  React.useEffect(() => {
    if (!selectedDataset) {
      // Initialize refs on first render when dataset is not selected
      if (prevDatasetRef.current === null) {
        prevDatasetRef.current = '';
        prevFilterRef.current = 'all';
      }
      return;
    }
    
    // Initialize refs on first render with dataset
    if (prevDatasetRef.current === null) {
      prevDatasetRef.current = selectedDataset;
      prevFilterRef.current = filterMode;
      return; // Don't reset on initial load
    }
    
    const datasetChanged = prevDatasetRef.current !== selectedDataset;
    const filterChanged = prevFilterRef.current !== filterMode;
    
    if (datasetChanged || filterChanged) {
      prevDatasetRef.current = selectedDataset;
      prevFilterRef.current = filterMode;
      // Reset to page 0 when dataset or filter changes
      setPaginationModel(prev => ({ page: 0, pageSize: prev.pageSize }));
    }
  }, [selectedDataset, filterMode]);
  
  // Fetch data when pagination, dataset, or filter changes
  React.useEffect(() => {
    if (selectedDataset) {
      const offset = paginationModel.page * paginationModel.pageSize;
      fetchViewData(paginationModel.pageSize, offset);
    }
    // fetchViewData is stable (useCallback with selectedDataset and filterMode dependencies)
    // We exclude it from deps to avoid unnecessary re-renders, but it will use latest values
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paginationModel.page, paginationModel.pageSize, selectedDataset, filterMode]);

  const fetchApiHealth = React.useCallback(async (silent = false) => {
    if (!silent) {
      setApiHealth('loading');
      setApiHealthDetail(null);
    }
    try {
      // /health endpoint doesn't require headers
      const res = await fetch(`${API_BASE}/health`);
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
    const interval = setInterval(() => fetchApiHealth(true), 60000); // Check every 60 seconds (1 minute)
    return () => clearInterval(interval);
  }, [fetchApiHealth]);

  // Removed handleViewLimitChange - using DataGrid pagination instead

  const [validationResult, setValidationResult] = React.useState<{ message: string; severity: 'success' | 'warning' } | null>(null);

  const handleRevalidate = async () => {
    if (!selectedDataset.trim()) {
      setSnackbar({ open: true, message: 'Please select a dataset to validate.', severity: 'warning' });
      return;
    }

    setValidating(true);
    setValidationResult(null);
    try {
      const headers: Record<string, string> = {
        'dataset_name': selectedDataset.trim(),
      };
      
      // Add optional percentage headers if provided
      if (label0Percent.trim()) {
        headers['X-Label-0-Percent'] = label0Percent.trim();
      }
      if (label1Percent.trim()) {
        headers['X-Label-1-Percent'] = label1Percent.trim();
      }
      if (trainingPercent.trim()) {
        headers['X-Training-Percent'] = trainingPercent.trim();
      }
      if (testingPercent.trim()) {
        headers['X-Testing-Percent'] = testingPercent.trim();
      }
      
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
          label_0_rows?: number;
          label_1_rows?: number;
          label_0_percentage?: number;
          label_1_percentage?: number;
        };
        if (json.message) message = json.message;
        if (
          typeof json.training_rows === 'number' &&
          typeof json.testing_rows === 'number'
        ) {
          let labelInfo = '';
          if (typeof json.label_0_rows === 'number' && typeof json.label_1_rows === 'number') {
            labelInfo = ` | Labels: ${json.label_0_rows} (${json.label_0_percentage ?? '—'}%) = 0, ${json.label_1_rows} (${json.label_1_percentage ?? '—'}%) = 1`;
          }
          message = `Validation: ✅ ${json.training_rows} training (${json.training_percentage ?? '—'}%), ${json.testing_rows} testing (${json.testing_percentage ?? '—'}%)${labelInfo}`;
        } else if (json.total_rows === 0) {
          message = 'No rows to validate.';
        }
      } catch {
        if (text) message = text;
      }
      setValidationResult({ message, severity: 'success' });
      setSnackbar({ open: true, message, severity: 'success' });
      setPaginationModel({ page: 0, pageSize: paginationModel.pageSize });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to validate dataset.';
      setValidationResult({ message: msg, severity: 'warning' });
      setSnackbar({ open: true, message: msg, severity: 'error' });
    } finally {
      setValidating(false);
    }
  };

  const handleInsert = async () => {
    // /insert endpoint was removed from backend
    setSnackbar({ open: true, message: 'Insert endpoint is no longer available. Please upload data via CSV file upload.', severity: 'info' });
  };

  const handleClearConfirm = async () => {
    setClearLoading(true);
    try {
      const headers: Record<string, string> = {};
      if (selectedDataset.trim()) headers['dataset_name'] = selectedDataset.trim();
      const res = await fetch(`${API_BASE}/clear`, { method: 'DELETE', headers });
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
      setPaginationModel({ page: 0, pageSize: paginationModel.pageSize });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to clear database.';
      setSnackbar({ open: true, message: msg, severity: 'error' });
    } finally {
      setClearLoading(false);
    }
  };

  const handleTrain = async () => {
    if (!modelName.trim()) {
      setSnackbar({ open: true, message: 'Please enter a model name.', severity: 'warning' });
      return;
    }
    
    if (!modelType) {
      setSnackbar({ open: true, message: 'Please select a model architecture.', severity: 'warning' });
      return;
    }
    
    if (!selectedDataset.trim()) {
      setSnackbar({ open: true, message: 'Please select a dataset to train on.', severity: 'warning' });
      return;
    }

    setTraining(true);
    setMetrics(null);
    const trainStartMs = performance.now();

    try {
      // TrainRequest body structure
      const payload: any = {
        n_estimators: 100,
        max_depth: null,
        random_state: 42,
        model_type: modelType,  // Include the selected model architecture
      };
      
      // Only include include_fields if fields are selected
      if (selectedFeatures.length > 0) {
        payload.include_fields = selectedFeatures;
      }
      // exclude_fields is optional, don't include if not needed

      // Headers: dataset_name and model_name are required
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'dataset_name': selectedDataset.trim(),
        'model_name': modelName.trim(),
      };

      const res = await fetch(`${MODEL_API_BASE}/train`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Training failed');
      }

      setMetrics(data);
      playCompletionSound();
      // Refresh models list after training
      fetchModels();
      const modelTypeDisplay = modelType || data.model_type || 'model';
      const trainingDurationSeconds =
        typeof data.training_duration_seconds === 'number'
          ? data.training_duration_seconds
          : Number(((performance.now() - trainStartMs) / 1000).toFixed(3));
      setSnackbar({ 
        open: true, 
        message: `Model "${modelName.trim()}" (${modelTypeDisplay}) trained successfully on dataset "${selectedDataset.trim()}" in ${trainingDurationSeconds.toFixed(2)}s`, 
        severity: 'success' 
      });
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
        if (selectedDataset.trim()) headers['dataset_name'] = selectedDataset.trim();
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

      if (!modelName.trim()) {
        throw new Error('Model name is required for prediction.');
      }

      // PredictRequest body structure
      const payload = {
        data: dataToPredict
      };

      // Headers: model_name is required
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'model_name': modelName.trim(),
      };

      const res = await fetch(`${MODEL_API_BASE}/predict`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload)
      });

      const result = await res.json();
      if (!res.ok) throw new Error(result.detail || 'Prediction failed');

      // Debug: log the raw response to see attack_cat
      console.log('Prediction response:', result);
      if (result.predictions && result.predictions.length > 0) {
        console.log('First prediction:', result.predictions[0]);
        console.log('attack_cat in first prediction:', result.predictions[0]?.attack_cat);
      }

      // Transform backend response to match frontend expectations
      const transformedResults = (result.predictions || []).map((pred: any, idx: number) => {
        // prediction is now a percentage (0-100), not binary 0/1
        const riskPercentage = typeof pred.prediction === 'number' ? pred.prediction : (pred.prediction === 1 ? 100 : 0);
        const isAnomaly = riskPercentage >= 50; // Consider >= 50% as anomaly
        
        const transformed = {
          index: idx,
          label: pred.label || (isAnomaly ? 'anomaly' : 'normal'),
          confidence: pred.confidence || pred.probability_safe || 0,
          prediction: riskPercentage, // Risk percentage (0-100)
          probability_safe: pred.probability_safe,
          probability_unsafe: pred.probability_unsafe,
          attack_cat: pred.attack_cat || null,
          attack_cat_probabilities: pred.attack_cat_probabilities || {},
        };
        // Debug: log attack_cat for anomalies
        if (isAnomaly) {
          console.log(`Sample ${idx}: attack_cat =`, transformed.attack_cat);
        }
        return transformed;
      });
      setPredictionResults(transformedResults);
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

  const canTrain = modelName.trim() && selectedDataset.trim() && selectedFeatures.length > 0 && modelType;

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
                  Dataset Name (sent as dataset_name header)
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
                <FormControl size="small" sx={{ minWidth: 180 }}>
                  <InputLabel id="dataset-label">Dataset</InputLabel>
                  <Select
                    labelId="dataset-label"
                    value={selectedDataset}
                    label="Dataset"
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    disabled={datasetsLoading || availableDatasets.length === 0}
                  >
                    {availableDatasets.map((ds) => (
                      <MenuItem key={ds} value={ds}>{ds}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                {/* Removed "Rows to show" dropdown - using DataGrid pagination instead */}
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={datasetsLoading ? <CircularProgress size={14} color="inherit" /> : <RefreshRoundedIcon />}
                  onClick={fetchTables}
                  disabled={datasetsLoading}
                  title="Refresh dataset list"
                >
                  Refresh Datasets
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={viewLoading ? <CircularProgress size={14} color="inherit" /> : <RefreshRoundedIcon />}
                  onClick={() => {
                    const offset = paginationModel.page * paginationModel.pageSize;
                    fetchViewData(paginationModel.pageSize, offset);
                  }}
                  disabled={viewLoading || !selectedDataset}
                >
                  Refresh Data
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
                Showing {rows.length} of {viewTotalRows} rows (Page {paginationModel.page + 1} of {Math.ceil(viewTotalRows / paginationModel.pageSize)})
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
                  getRowId={(row) => row.id as string}
                  paginationModel={paginationModel}
                  onPaginationModelChange={setPaginationModel}
                  pageSizeOptions={[25, 50, 100, 200, 500, 1000]}
                  paginationMode="server"
                  rowCount={viewTotalRows || 0}
                  disableColumnResize
                  density="compact"
                  loading={viewLoading}
                  keepNonExistentRowsSelected
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
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mb: 0.75 }}>
                  Revalidate Dataset
                </Typography>
                <Stack direction="row" spacing={1} sx={{ mb: 1.5 }}>
                  <FormControl size="small" sx={{ minWidth: 200 }}>
                    <InputLabel id="validate-dataset-label">Dataset to Validate</InputLabel>
                    <Select
                      labelId="validate-dataset-label"
                      value={selectedDataset}
                      label="Dataset to Validate"
                      onChange={(e) => setSelectedDataset(e.target.value)}
                      disabled={datasetsLoading || availableDatasets.length === 0}
                    >
                      {availableDatasets.map((ds) => (
                        <MenuItem key={ds} value={ds}>{ds}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <Button
                    variant="contained"
                    color="warning"
                    startIcon={validating ? <CircularProgress size={16} color="inherit" /> : <RefreshRoundedIcon />}
                    onClick={handleRevalidate}
                    disabled={validating || !selectedDataset}
                    sx={(theme) => ({
                      color: `${theme.palette.warning.contrastText || '#1a1a1a'} !important`,
                      border: 'none !important',
                      boxShadow: 'none !important',
                      outline: 'none !important',
                      '&:focus, &:focus-visible': { outline: 'none !important', boxShadow: 'none !important', border: 'none !important' },
                      '& .MuiSvgIcon-root': { color: 'inherit' },
                    })}
                  >
                    {validating ? 'Validating…' : 'Revalidate'}
                  </Button>
                </Stack>
                
                {/* Label Distribution Controls */}
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mt: 2, mb: 1 }}>
                  Label Distribution (Optional)
                </Typography>
                <Grid container spacing={1} sx={{ mb: 1.5 }}>
                  <Grid item xs={6} sm={3}>
                    <TextField
                      size="small"
                      label="Label 0 (%)"
                      type="number"
                      value={label0Percent}
                      onChange={(e) => {
                        const val = e.target.value;
                        if (val === '' || (parseFloat(val) >= 0 && parseFloat(val) <= 100)) {
                          setLabel0Percent(val);
                          // Auto-calculate label 1 if both are being set
                          if (val && label1Percent) {
                            const remaining = 100 - parseFloat(val);
                            if (remaining >= 0 && remaining <= 100) {
                              setLabel1Percent(remaining.toFixed(1));
                            }
                          }
                        }
                      }}
                      inputProps={{ min: 0, max: 100, step: 0.1 }}
                      helperText="% labeled as 0"
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <TextField
                      size="small"
                      label="Label 1 (%)"
                      type="number"
                      value={label1Percent}
                      onChange={(e) => {
                        const val = e.target.value;
                        if (val === '' || (parseFloat(val) >= 0 && parseFloat(val) <= 100)) {
                          setLabel1Percent(val);
                          // Auto-calculate label 0 if both are being set
                          if (val && label0Percent) {
                            const remaining = 100 - parseFloat(val);
                            if (remaining >= 0 && remaining <= 100) {
                              setLabel0Percent(remaining.toFixed(1));
                            }
                          }
                        }
                      }}
                      inputProps={{ min: 0, max: 100, step: 0.1 }}
                      helperText="% labeled as 1"
                      fullWidth
                    />
                  </Grid>
                </Grid>
                
                {/* Training/Testing Split Controls */}
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mt: 1, mb: 1 }}>
                  Training/Testing Split
                </Typography>
                <Grid container spacing={1} sx={{ mb: 1.5 }}>
                  <Grid item xs={6} sm={3}>
                    <TextField
                      size="small"
                      label="Training (%)"
                      type="number"
                      value={trainingPercent}
                      onChange={(e) => {
                        const val = e.target.value;
                        if (val === '' || (parseFloat(val) >= 0 && parseFloat(val) <= 100)) {
                          setTrainingPercent(val);
                          // Auto-calculate testing if both are being set
                          if (val && testingPercent) {
                            const remaining = 100 - parseFloat(val);
                            if (remaining >= 0 && remaining <= 100) {
                              setTestingPercent(remaining.toFixed(1));
                            }
                          }
                        }
                      }}
                      inputProps={{ min: 0, max: 100, step: 0.1 }}
                      helperText="% for training"
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <TextField
                      size="small"
                      label="Testing (%)"
                      type="number"
                      value={testingPercent}
                      onChange={(e) => {
                        const val = e.target.value;
                        if (val === '' || (parseFloat(val) >= 0 && parseFloat(val) <= 100)) {
                          setTestingPercent(val);
                          // Auto-calculate training if both are being set
                          if (val && trainingPercent) {
                            const remaining = 100 - parseFloat(val);
                            if (remaining >= 0 && remaining <= 100) {
                              setTrainingPercent(remaining.toFixed(1));
                            }
                          }
                        }
                      }}
                      inputProps={{ min: 0, max: 100, step: 0.1 }}
                      helperText="% for testing"
                      fullWidth
                    />
                  </Grid>
                </Grid>
                
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
                  Start a new training session. Select a dataset and fields to include in training.
                </Typography>
                <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                  <InputLabel id="train-dataset-label">Dataset for Training</InputLabel>
                  <Select
                    labelId="train-dataset-label"
                    value={selectedDataset || ''}
                    label="Dataset for Training"
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    disabled={datasetsLoading || availableDatasets.length === 0}
                  >
                    {availableDatasets.map((ds) => (
                      <MenuItem key={ds} value={ds}>{ds}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Autocomplete
                  freeSolo
                  options={modelTypes.map((type) => type.model_type)}
                  value={modelName}
                  onInputChange={(_, newValue) => setModelName(newValue)}
                  loading={modelTypesLoading}
                  renderInput={(params) => (
                <TextField
                      {...params}
                  fullWidth
                  size="small"
                  label="Model Name"
                      placeholder="e.g., AEv1, IFv1, RFv1, or custom name"
                  error={!!modelNameError}
                      helperText={modelNameError || 'Required: Unique name for this model. Select from available types or enter custom name.'}
                      InputProps={{
                        ...params.InputProps,
                        endAdornment: (
                          <React.Fragment>
                            {modelTypesLoading ? <CircularProgress color="inherit" size={20} /> : null}
                            {params.InputProps.endAdornment}
                          </React.Fragment>
                        ),
                      }}
                    />
                  )}
                  sx={{ mb: 2 }}
                />
                <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                  <InputLabel id="model-architecture-label">Select Model Architecture</InputLabel>
                  <Select
                    labelId="model-architecture-label"
                    id="model-architecture-select"
                  value={modelType}
                    label="Select Model Architecture"
                  onChange={(e) => setModelType(e.target.value)}
                    disabled={modelTypesLoading || modelTypes.length === 0}
                  >
                    {modelTypes.length > 0 ? (
                      modelTypes.map((type) => (
                        <MenuItem key={type.model_type} value={type.model_type}>
                          {type.model_type}
                          {type.files && type.files.length > 0 && (
                            <Typography component="span" variant="caption" sx={{ ml: 1, color: 'text.secondary' }}>
                              ({type.files.length} file{type.files.length !== 1 ? 's' : ''})
                            </Typography>
                          )}
                        </MenuItem>
                      ))
                    ) : (
                      <MenuItem disabled value="">
                        {modelTypesLoading ? 'Loading model types...' : 'No model types available'}
                      </MenuItem>
                    )}
                  </Select>
                </FormControl>
              </Box>
              
              {/* Field Selection */}
              <Box>
                <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1, fontWeight: 500 }}>
                  Select Fields to Include in Training
                </Typography>
                {fieldsLoading ? (
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ py: 2 }}>
                    <CircularProgress size={16} />
                    <Typography variant="caption" color="text.secondary">
                      Loading fields...
                    </Typography>
                  </Stack>
                ) : availableFields.length === 0 ? (
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', py: 1 }}>
                    {selectedDataset 
                      ? 'No fields available. Make sure a dataset is selected and contains data.'
                      : 'Select a dataset to view available fields.'}
                  </Typography>
                ) : (
                  <FormGroup>
                    <Box
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
                        gap: 1,
                        maxHeight: 300,
                        overflowY: 'auto',
                        p: 1,
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        bgcolor: 'action.hover',
                      }}
                    >
                      {availableFields.map((field) => (
                        <FormControlLabel
                          key={field}
                          control={
                            <Checkbox
                              checked={selectedFeatures.includes(field)}
                              onChange={() => {
                                setSelectedFeatures((prev) =>
                                  prev.includes(field)
                                    ? prev.filter((f) => f !== field)
                                    : [...prev, field]
                                );
                              }}
                              size="small"
                            />
                          }
                          label={
                            <Typography variant="body2" sx={{ fontSize: '0.875rem' }}>
                              {field}
                            </Typography>
                          }
                        />
                      ))}
                    </Box>
                    <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => setSelectedFeatures([...availableFields])}
                        disabled={selectedFeatures.length === availableFields.length}
                      >
                        Select All
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => setSelectedFeatures([])}
                        disabled={selectedFeatures.length === 0}
                      >
                        Clear All
                      </Button>
                      <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center', ml: 'auto' }}>
                        {selectedFeatures.length} of {availableFields.length} selected
                      </Typography>
                    </Stack>
                  </FormGroup>
                )}
              </Box>
              <Button
                variant="contained"
                size="large"
                startIcon={training ? <CircularProgress size={20} color="inherit" /> : <PsychologyRoundedIcon />}
                onClick={handleTrain}
                disabled={training || !modelName.trim() || !modelType || !selectedDataset.trim()}
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
                Dataset KPIs and metrics
              </Typography>
            </Stack>
            {metrics ? (
              <Stack spacing={2}>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  <Chip label={`Status: ${metrics.status}`} color="success" size="small" />
                  <Chip label={`Dataset: ${metrics.dataset_name || metrics.dataset || 'N/A'}`} size="small" variant="outlined" />
                  <Chip label={`Model: ${metrics.model_name || 'N/A'}`} size="small" variant="outlined" />
                  <Chip label={`Type: ${metrics.model_type || 'N/A'}`} size="small" variant="outlined" />
                  {(metrics.training_duration_seconds != null || metrics.training_params?.training_duration_seconds != null) && (
                    <Chip
                      label={`Train Time: ${Number(metrics.training_duration_seconds ?? metrics.training_params?.training_duration_seconds).toFixed(2)}s`}
                      size="small"
                      variant="outlined"
                    />
                  )}
                  {metrics.features && (
                    <Chip label={`Features: ${typeof metrics.features === 'string' ? metrics.features : metrics.n_features || 'N/A'}`} size="small" variant="outlined" />
                  )}
                </Stack>
                
                {/* Loss History Graph */}
                {(() => {
                  // Check for loss_history in multiple possible locations
                  const lossHistory = metrics?.loss_history || metrics?.training_params?.loss_history;
                  
                  // Debug logging
                  if (metrics && !lossHistory) {
                    console.log('Loss history not found. Metrics keys:', Object.keys(metrics));
                    console.log('metrics.loss_history:', metrics.loss_history);
                    console.log('metrics.training_params:', metrics.training_params);
                  }
                  
                  if (!lossHistory || !Array.isArray(lossHistory) || lossHistory.length === 0) {
                    return null;
                  }
                  
                  const iterations = lossHistory.map((item: any) => {
                    // Handle both object format {iteration: X, loss: Y} and array format
                    if (typeof item === 'object' && item !== null) {
                      return Number(item.iteration || item[0]);
                    }
                    return Number(item);
                  }).filter((v: number) => !isNaN(v));
                  
                  const losses = lossHistory.map((item: any) => {
                    // Handle both object format {iteration: X, loss: Y} and array format
                    if (typeof item === 'object' && item !== null) {
                      return Number(item.loss || item[1]);
                    }
                    return Number(item);
                  }).filter((v: number) => !isNaN(v));
                  
                  if (iterations.length === 0 || losses.length === 0 || iterations.length !== losses.length) {
                    console.warn('Loss history data format issue:', { iterations, losses, lossHistory });
                    return null;
                  }
                  
                  return (
                    <Card variant="outlined" sx={{ mt: 2, mb: 2 }}>
                      <CardContent>
                        <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
                          Training Loss History
                        </Typography>
                        <Box sx={{ width: '100%', height: 300 }}>
                          <LineChart
                            xAxis={[{
                              data: iterations,
                              label: 'Iteration',
                              scaleType: 'linear',
                            }]}
                            yAxis={[{
                              label: 'Loss (MSE)',
                              scaleType: 'linear',
                            }]}
                            series={[{
                              data: losses,
                              label: 'Loss',
                              color: '#1976d2',
                              curve: 'monotone',
                              showMark: true,
                            }]}
                            width={undefined}
                            height={300}
                            margin={{ top: 20, right: 20, bottom: 40, left: 50 }}
                            grid={{ vertical: true, horizontal: true }}
                          />
                        </Box>
                      </CardContent>
                    </Card>
                  );
                })()}
                
                {/* Show message if loss history is not available (for non-AEv1 models) */}
                {metrics && !metrics.loss_history && !metrics.training_params?.loss_history && metrics.model_type === 'AEv1' && (
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1, fontStyle: 'italic' }}>
                    Loss history not available for this model.
                  </Typography>
                )}
                
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
                  {(metrics.f1_score != null || metrics.f1 != null) && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">F1 Score</Typography>
                        <Typography variant="h6">{((Number(metrics.f1_score || metrics.f1)) * 100).toFixed(1)}%</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.specificity != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">Specificity</Typography>
                        <Typography variant="h6">{(Number(metrics.specificity) * 100).toFixed(1)}%</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.sensitivity != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">Sensitivity</Typography>
                        <Typography variant="h6">{(Number(metrics.sensitivity) * 100).toFixed(1)}%</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.npv != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">NPV</Typography>
                        <Typography variant="h6">{(Number(metrics.npv) * 100).toFixed(1)}%</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.mcc != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">MCC</Typography>
                        <Typography variant="h6">{Number(metrics.mcc).toFixed(3)}</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.roc_auc != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">ROC AUC</Typography>
                        <Typography variant="h6">{Number(metrics.roc_auc).toFixed(3)}</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.pr_auc != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">PR AUC</Typography>
                        <Typography variant="h6">{Number(metrics.pr_auc).toFixed(3)}</Typography>
                      </Box>
                    </Grid>
                  )}
                  {metrics.total_support != null && (
                    <Grid size={{ xs: 6, sm: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                        <Typography variant="caption" color="text.secondary" display="block">Total Samples</Typography>
                        <Typography variant="h6">{Number(metrics.total_support).toLocaleString()}</Typography>
                        {metrics.support_0 != null && metrics.support_1 != null && (
                          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                            Class 0: {Number(metrics.support_0).toLocaleString()}, Class 1: {Number(metrics.support_1).toLocaleString()}
                          </Typography>
                        )}
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
                
                {/* Dataset Statistics */}
                {(datasetStats || typeStats) && (
                  <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                    <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>
                      Dataset Statistics
                    </Typography>
                    {statsLoading ? (
                      <Stack direction="row" spacing={1} alignItems="center">
                        <CircularProgress size={16} />
                        <Typography variant="caption" color="text.secondary">
                          Loading stats...
                        </Typography>
                      </Stack>
                    ) : (
                      <Grid container spacing={2}>
                        {datasetStats && (
                          <>
                            {datasetStats.total_records != null && (
                              <Grid size={{ xs: 6, sm: 3 }}>
                                <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                                  <Typography variant="caption" color="text.secondary" display="block">Total Records</Typography>
                                  <Typography variant="h6">{datasetStats.total_records.toLocaleString()}</Typography>
                                </Box>
                              </Grid>
                            )}
                            {datasetStats.training_records != null && (
                              <Grid size={{ xs: 6, sm: 3 }}>
                                <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                                  <Typography variant="caption" color="text.secondary" display="block">Training Records</Typography>
                                  <Typography variant="h6">{datasetStats.training_records.toLocaleString()}</Typography>
                                  {datasetStats.training_percentage != null && (
                                    <Typography variant="caption" color="text.secondary">
                                      ({datasetStats.training_percentage}%)
                                    </Typography>
                                  )}
                                </Box>
                              </Grid>
                            )}
                            {datasetStats.testing_records != null && (
                              <Grid size={{ xs: 6, sm: 3 }}>
                                <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                                  <Typography variant="caption" color="text.secondary" display="block">Testing Records</Typography>
                                  <Typography variant="h6">{datasetStats.testing_records.toLocaleString()}</Typography>
                                  {datasetStats.testing_percentage != null && (
                                    <Typography variant="caption" color="text.secondary">
                                      ({datasetStats.testing_percentage}%)
                                    </Typography>
                                  )}
                                </Box>
                              </Grid>
                            )}
                          </>
                        )}
                        {typeStats && typeStats.type_distribution && Object.keys(typeStats.type_distribution).length > 0 && (
                          <Grid size={{ xs: 12 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                                Type Distribution
                              </Typography>
                              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                {Object.entries(typeStats.type_distribution).map(([type, count]: [string, any]) => (
                                  <Chip
                                    key={type}
                                    label={`${type}: ${count.toLocaleString()}`}
                                    size="small"
                                    variant="outlined"
                                  />
                                ))}
                              </Stack>
                            </Box>
                          </Grid>
                        )}
                      </Grid>
                    )}
                  </Box>
                )}

                <Typography variant="caption" color="text.secondary">
                  Trained: {new Date(String(metrics.timestamp)).toLocaleString()}
                </Typography>
              </Stack>
            ) : (
              <Stack spacing={2}>
                <Box>
                  <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1.5 }}>
                    Select a dataset to view statistics and metrics.
                  </Typography>
                  <FormControl size="small" sx={{ minWidth: 250 }}>
                    <InputLabel id="stats-dataset-label">Select Dataset</InputLabel>
                    <Select
                      labelId="stats-dataset-label"
                      value={selectedDataset}
                      label="Select Dataset"
                      onChange={(e) => setSelectedDataset(e.target.value)}
                      disabled={datasetsLoading || availableDatasets.length === 0}
                    >
                      {availableDatasets.map((ds) => (
                        <MenuItem key={ds} value={ds}>{ds}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>
                {/* Show dataset stats even without model metrics */}
                {(datasetStats || typeStats) && selectedDataset && (
                  <Box sx={{ pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                    <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>
                      Dataset Statistics
                    </Typography>
                    {statsLoading ? (
                      <Stack direction="row" spacing={1} alignItems="center">
                        <CircularProgress size={16} />
                        <Typography variant="caption" color="text.secondary">
                          Loading stats...
                        </Typography>
                      </Stack>
                    ) : (
                      <Grid container spacing={2}>
                        {datasetStats && (
                          <>
                            {datasetStats.total_records != null && (
                              <Grid size={{ xs: 6, sm: 3 }}>
                                <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                                  <Typography variant="caption" color="text.secondary" display="block">Total Records</Typography>
                                  <Typography variant="h6">{datasetStats.total_records.toLocaleString()}</Typography>
                                </Box>
                              </Grid>
                            )}
                            {datasetStats.training_records != null && (
                              <Grid size={{ xs: 6, sm: 3 }}>
                                <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                                  <Typography variant="caption" color="text.secondary" display="block">Training Records</Typography>
                                  <Typography variant="h6">{datasetStats.training_records.toLocaleString()}</Typography>
                                  {datasetStats.training_percentage != null && (
                                    <Typography variant="caption" color="text.secondary">
                                      ({datasetStats.training_percentage}%)
                                    </Typography>
                                  )}
                                </Box>
                              </Grid>
                            )}
                            {datasetStats.testing_records != null && (
                              <Grid size={{ xs: 6, sm: 3 }}>
                                <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                                  <Typography variant="caption" color="text.secondary" display="block">Testing Records</Typography>
                                  <Typography variant="h6">{datasetStats.testing_records.toLocaleString()}</Typography>
                                  {datasetStats.testing_percentage != null && (
                                    <Typography variant="caption" color="text.secondary">
                                      ({datasetStats.testing_percentage}%)
                                    </Typography>
                                  )}
                                </Box>
                              </Grid>
                            )}
                          </>
                        )}
                        {typeStats && typeStats.type_distribution && Object.keys(typeStats.type_distribution).length > 0 && (
                          <Grid size={{ xs: 12 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                                Type Distribution
                              </Typography>
                              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                {Object.entries(typeStats.type_distribution).map(([type, count]: [string, any]) => (
                                  <Chip
                                    key={type}
                                    label={`${type}: ${count.toLocaleString()}`}
                                    size="small"
                                    variant="outlined"
                                  />
                                ))}
                              </Stack>
                            </Box>
                          </Grid>
                        )}
                      </Grid>
                    )}
                  </Box>
                )}
                {!selectedDataset && availableDatasets.length > 0 && (
                  <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic', mt: 1 }}>
                    Select a dataset above to view its statistics.
                  </Typography>
                )}
                {availableDatasets.length === 0 && (
                  <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic', mt: 1 }}>
                    No datasets available. Upload a dataset first.
                  </Typography>
                )}
              </Stack>
            )}
          </CardContent>
        </Card>

        {/* Section E.5: Model Testing */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderLeft: '4px solid hsl(210, 90%, 50%)',
            borderTopLeftRadius: 0,
            borderBottomLeftRadius: 0,
            boxShadow: 'none',
            transition: 'box-shadow 0.2s ease',
            '&:hover': { boxShadow: theme.shadows[1] },
          })}
        >
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 2, flexWrap: 'wrap', gap: 0.5 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'hsl(210, 90%, 40%)', letterSpacing: '0.03em' }}>
                Model Testing
              </Typography>
            </Stack>
            <Stack spacing={2}>
              <Box>
                <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2 }}>
                  Test a trained model on a dataset to evaluate its performance.
                </Typography>
                <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                  <InputLabel id="test-dataset-label">Dataset for Testing</InputLabel>
                  <Select
                    labelId="test-dataset-label"
                    value={selectedDataset}
                    label="Dataset for Testing"
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    disabled={datasetsLoading || availableDatasets.length === 0}
                  >
                    {availableDatasets.map((ds) => (
                      <MenuItem key={ds} value={ds}>{ds}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                  <InputLabel id="test-model-label">Model to Test</InputLabel>
                  <Select
                    labelId="test-model-label"
                    value={selectedTestModel}
                    label="Model to Test"
                    onChange={(e) => setSelectedTestModel(e.target.value)}
                    disabled={modelsLoading || !availableModels || availableModels.length === 0}
                  >
                    {modelsLoading && (
                      <MenuItem disabled>
                        <Stack direction="row" spacing={1} alignItems="center">
                          <CircularProgress size={16} />
                          <Typography variant="body2">Loading models...</Typography>
                        </Stack>
                      </MenuItem>
                    )}
                    {!modelsLoading && availableModels.length === 0 && (
                      <MenuItem disabled>No models available</MenuItem>
                    )}
                    {availableModels.map((model) => (
                      <MenuItem key={model.model_name} value={model.model_name}>
                        <Stack>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {model.model_name}
                          </Typography>
                          <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                            {model.accuracy != null && (
                              <Typography variant="caption" color="text.secondary">
                                Accuracy: {(model.accuracy * 100).toFixed(1)}%
                              </Typography>
                            )}
                            {model.n_features != null && (
                              <Typography variant="caption" color="text.secondary">
                                • {model.n_features} features
                              </Typography>
                            )}
                            {model.training_date && (
                              <Typography variant="caption" color="text.secondary">
                                • {new Date(model.training_date).toLocaleDateString()}
                              </Typography>
                            )}
                          </Stack>
                        </Stack>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Button
                  variant="contained"
                  color="info"
                  size="large"
                  startIcon={testing ? <CircularProgress size={20} color="inherit" /> : <PsychologyRoundedIcon />}
                  onClick={handleTest}
                  disabled={testing || !selectedTestModel.trim() || !selectedDataset.trim()}
                  sx={{
                    alignSelf: 'flex-start',
                    mt: 1,
                  }}
                >
                  {testing ? 'Testing…' : 'Test Model'}
                </Button>
              </Box>

              {/* Test Results */}
              {testResults && (
                <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                  <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>
                    Test Results
                  </Typography>
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 2 }}>
                    <Chip label={`Status: ${testResults.status}`} color="success" size="small" />
                    <Chip label={`Samples: ${testResults.testing_samples?.toLocaleString() || 'N/A'}`} size="small" variant="outlined" />
                    {testResults.testing_duration_seconds != null && (
                      <Chip label={`Test Time: ${Number(testResults.testing_duration_seconds).toFixed(2)}s`} size="small" variant="outlined" />
                    )}
                  </Stack>
                  {testResults.metrics && (
                    <Grid container spacing={2}>
                      {testResults.metrics.accuracy != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Accuracy</Typography>
                            <Typography variant="h6">{(Number(testResults.metrics.accuracy) * 100).toFixed(1)}%</Typography>
                          </Box>
                        </Grid>
                      )}
                      {testResults.metrics.precision != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Precision</Typography>
                            <Typography variant="h6">{(Number(testResults.metrics.precision) * 100).toFixed(1)}%</Typography>
                          </Box>
                        </Grid>
                      )}
                      {testResults.metrics.recall != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Recall</Typography>
                            <Typography variant="h6">{(Number(testResults.metrics.recall) * 100).toFixed(1)}%</Typography>
                          </Box>
                        </Grid>
                      )}
                      {(testResults.metrics.f1_score != null || testResults.metrics.f1 != null) && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">F1 Score</Typography>
                            <Typography variant="h6">{((Number(testResults.metrics.f1_score || testResults.metrics.f1)) * 100).toFixed(1)}%</Typography>
                          </Box>
                        </Grid>
                      )}
                      {testResults.metrics.specificity != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Specificity</Typography>
                            <Typography variant="h6">{(Number(testResults.metrics.specificity) * 100).toFixed(1)}%</Typography>
                          </Box>
                        </Grid>
                      )}
                      {testResults.metrics.sensitivity != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Sensitivity</Typography>
                            <Typography variant="h6">{(Number(testResults.metrics.sensitivity) * 100).toFixed(1)}%</Typography>
                          </Box>
                        </Grid>
                      )}
                      {testResults.metrics.mcc != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">MCC</Typography>
                            <Typography variant="h6">{Number(testResults.metrics.mcc).toFixed(3)}</Typography>
                          </Box>
                        </Grid>
                      )}
                      {testResults.metrics.roc_auc != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">ROC AUC</Typography>
                            <Typography variant="h6">{Number(testResults.metrics.roc_auc).toFixed(3)}</Typography>
                          </Box>
                        </Grid>
                      )}
                      {testResults.metrics.pr_auc != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">PR AUC</Typography>
                            <Typography variant="h6">{Number(testResults.metrics.pr_auc).toFixed(3)}</Typography>
                          </Box>
                        </Grid>
                      )}
                      {testResults.metrics.total_support != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Total Samples</Typography>
                            <Typography variant="h6">{Number(testResults.metrics.total_support).toLocaleString()}</Typography>
                            {testResults.metrics.support_0 != null && testResults.metrics.support_1 != null && (
                              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                                Class 0: {Number(testResults.metrics.support_0).toLocaleString()}, Class 1: {Number(testResults.metrics.support_1).toLocaleString()}
                              </Typography>
                            )}
                          </Box>
                        </Grid>
                      )}
                      {testResults.metrics.f1 != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">F1 Score</Typography>
                            <Typography variant="h6">{(Number(testResults.metrics.f1) * 100).toFixed(1)}%</Typography>
                          </Box>
                        </Grid>
                      )}
                    </Grid>
                  )}
                  {testResults.timestamp && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                      Tested: {new Date(testResults.timestamp).toLocaleString()}
                    </Typography>
                  )}
                </Box>
              )}
            </Stack>
          </CardContent>
        </Card>

        {/* Section E.6: Model Data Carousel */}
        {availableModels.length > 0 && (
          <Card
            variant="outlined"
            sx={(theme) => ({
              borderLeft: '4px solid hsl(260, 80%, 50%)',
              borderTopLeftRadius: 0,
              borderBottomLeftRadius: 0,
              boxShadow: 'none',
              transition: 'box-shadow 0.2s ease',
              '&:hover': { boxShadow: theme.shadows[1] },
            })}
          >
            <CardContent>
              <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'hsl(260, 80%, 40%)', letterSpacing: '0.03em' }}>
                  Model Data Overview
                </Typography>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Typography variant="caption" color="text.secondary">
                    {currentModelIndex + 1} of {availableModels.length}
                  </Typography>
                  <Button
                    size="small"
                    onClick={handlePreviousModel}
                    disabled={availableModels.length <= 1 || modelDetailsLoading}
                    startIcon={<ChevronLeftRoundedIcon />}
                  >
                    Previous
                  </Button>
                  <Button
                    size="small"
                    onClick={handleNextModel}
                    disabled={availableModels.length <= 1 || modelDetailsLoading}
                    endIcon={<ChevronRightRoundedIcon />}
                  >
                    Next
                  </Button>
                  <Button
                    size="small"
                    color="error"
                    onClick={handleDeleteClick}
                    disabled={modelDetailsLoading || !currentModel}
                    startIcon={<DeleteSweepRoundedIcon />}
                    sx={{ ml: 1 }}
                  >
                    Delete
                  </Button>
                </Stack>
              </Stack>

              {modelDetailsLoading ? (
                <Stack alignItems="center" justifyContent="center" sx={{ py: 4 }}>
                  <CircularProgress />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    Loading model details...
                  </Typography>
                </Stack>
              ) : currentModel ? (
                <Stack spacing={3}>
                  {/* Model Info from /models */}
                  <Box>
                    <Typography variant="h6" sx={{ mb: 1.5, fontWeight: 600 }}>
                      {currentModel.model_name}
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid size={{ xs: 6, sm: 3 }}>
                        <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                          <Typography variant="caption" color="text.secondary" display="block">Model File</Typography>
                          <Typography variant="body2">{currentModel.model_file}</Typography>
                        </Box>
                      </Grid>
                      {currentModel.accuracy != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Accuracy</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {(currentModel.accuracy * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        </Grid>
                      )}
                      {currentModel.n_features != null && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Features</Typography>
                            <Typography variant="body2">{currentModel.n_features}</Typography>
                          </Box>
                        </Grid>
                      )}
                      {currentModel.training_date && (
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Training Date</Typography>
                            <Typography variant="body2">
                              {new Date(currentModel.training_date).toLocaleDateString()}
                            </Typography>
                          </Box>
                        </Grid>
                      )}
                    </Grid>
                  </Box>

                  {/* Model Status from /model/status */}
                  {currentModelStatus && (
                    <Box sx={{ pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                      <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>
                        Model Status
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid size={{ xs: 6, sm: 3 }}>
                          <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Status</Typography>
                            <Chip
                              label={currentModelStatus.status === 'trained' ? 'Trained' : 'Not Trained'}
                              color={currentModelStatus.status === 'trained' ? 'success' : 'default'}
                              size="small"
                              sx={{ mt: 0.5 }}
                            />
                          </Box>
                        </Grid>
                        {currentModelStatus.model_type && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">Model Type</Typography>
                              <Typography variant="body2">{currentModelStatus.model_type}</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelStatus.n_features != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">Features</Typography>
                              <Typography variant="body2">{currentModelStatus.n_features}</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelStatus.last_test_date && currentModelStatus.last_test_date !== 'Not tested yet' && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">Last Test</Typography>
                              <Typography variant="body2">
                                {new Date(currentModelStatus.last_test_date).toLocaleDateString()}
                              </Typography>
                            </Box>
                          </Grid>
                        )}
                      </Grid>
                    </Box>
                  )}

                  {/* Loss History Graph - Show at top if available */}
                  {(() => {
                    // Check for loss_history in training_params
                    const lossHistory = currentModelMetrics?.training_params?.loss_history;
                    
                    if (!lossHistory || !Array.isArray(lossHistory) || lossHistory.length === 0) {
                      return null;
                    }
                    
                    const iterations = lossHistory.map((item: any) => {
                      // Handle both object format {iteration: X, loss: Y} and array format
                      if (typeof item === 'object' && item !== null) {
                        return Number(item.iteration || item[0]);
                      }
                      return Number(item);
                    }).filter((v: number) => !isNaN(v));
                    
                    const losses = lossHistory.map((item: any) => {
                      // Handle both object format {iteration: X, loss: Y} and array format
                      if (typeof item === 'object' && item !== null) {
                        return Number(item.loss || item[1]);
                      }
                      return Number(item);
                    }).filter((v: number) => !isNaN(v));
                    
                    if (iterations.length === 0 || losses.length === 0 || iterations.length !== losses.length) {
                      console.warn('Loss history data format issue in Model Data Overview:', { iterations, losses, lossHistory });
                      return null;
                    }
                    
                    return (
                      <Card variant="outlined" sx={{ mb: 3, borderLeft: '3px solid', borderLeftColor: 'primary.main' }}>
                        <CardContent>
                          <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
                            Training Loss History
                          </Typography>
                          <Box sx={{ width: '100%', height: 300 }}>
                            <LineChart
                              xAxis={[{
                                data: iterations,
                                label: 'Iteration',
                                scaleType: 'linear',
                              }]}
                              yAxis={[{
                                label: 'Loss (MSE)',
                                scaleType: 'linear',
                              }]}
                              series={[{
                                data: losses,
                                label: 'Loss',
                                color: '#1976d2',
                                curve: 'monotone',
                                showMark: true,
                              }]}
                              width={undefined}
                              height={300}
                              margin={{ top: 20, right: 20, bottom: 40, left: 50 }}
                              grid={{ vertical: true, horizontal: true }}
                            />
                          </Box>
                        </CardContent>
                      </Card>
                    );
                  })()}

                  {/* Model Metrics from /model/metrics */}
                  {currentModelMetrics && currentModelMetrics.metrics && (
                    <Box sx={{ pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                      <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>
                        Model Metrics
                      </Typography>
                      
                      <Grid container spacing={2}>
                        {currentModelMetrics.metrics.accuracy != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">Accuracy</Typography>
                              <Typography variant="h6">{(Number(currentModelMetrics.metrics.accuracy) * 100).toFixed(1)}%</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelMetrics.metrics.precision != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">Precision</Typography>
                              <Typography variant="h6">{(Number(currentModelMetrics.metrics.precision) * 100).toFixed(1)}%</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelMetrics.metrics.recall != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">Recall</Typography>
                              <Typography variant="h6">{(Number(currentModelMetrics.metrics.recall) * 100).toFixed(1)}%</Typography>
                            </Box>
                          </Grid>
                        )}
                        {(currentModelMetrics.metrics.f1_score != null || currentModelMetrics.metrics.f1 != null) && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">F1 Score</Typography>
                              <Typography variant="h6">{((Number(currentModelMetrics.metrics.f1_score || currentModelMetrics.metrics.f1)) * 100).toFixed(1)}%</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelMetrics.metrics.specificity != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">Specificity</Typography>
                              <Typography variant="h6">{(Number(currentModelMetrics.metrics.specificity) * 100).toFixed(1)}%</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelMetrics.metrics.sensitivity != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">Sensitivity</Typography>
                              <Typography variant="h6">{(Number(currentModelMetrics.metrics.sensitivity) * 100).toFixed(1)}%</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelMetrics.metrics.npv != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">NPV</Typography>
                              <Typography variant="h6">{(Number(currentModelMetrics.metrics.npv) * 100).toFixed(1)}%</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelMetrics.metrics.mcc != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">MCC</Typography>
                              <Typography variant="h6">{Number(currentModelMetrics.metrics.mcc).toFixed(3)}</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelMetrics.metrics.roc_auc != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">ROC AUC</Typography>
                              <Typography variant="h6">{Number(currentModelMetrics.metrics.roc_auc).toFixed(3)}</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelMetrics.metrics.pr_auc != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">PR AUC</Typography>
                              <Typography variant="h6">{Number(currentModelMetrics.metrics.pr_auc).toFixed(3)}</Typography>
                            </Box>
                          </Grid>
                        )}
                        {currentModelMetrics.metrics.total_support != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">Total Samples</Typography>
                              <Typography variant="h6">{Number(currentModelMetrics.metrics.total_support).toLocaleString()}</Typography>
                              {currentModelMetrics.metrics.support_0 != null && currentModelMetrics.metrics.support_1 != null && (
                                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                                  Class 0: {Number(currentModelMetrics.metrics.support_0).toLocaleString()}, Class 1: {Number(currentModelMetrics.metrics.support_1).toLocaleString()}
                                </Typography>
                              )}
                            </Box>
                          </Grid>
                        )}
                      </Grid>
                    </Box>
                  )}
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                  No models available
                </Typography>
              )}
            </CardContent>
          </Card>
        )}

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
                      <Stack key={idx} direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" spacing={1}>
                        <Typography variant="body2">Sample #{res.index + 1}</Typography>
                        <Chip
                          label={res.label.toUpperCase()}
                          size="small"
                          color={res.label === 'anomaly' ? 'error' : 'success'}
                          sx={{ fontWeight: 700 }}
                        />
                        {res.attack_cat && res.attack_cat !== 'Normal' && res.attack_cat !== null && res.attack_cat !== 'Unknown' && (
                          <Chip
                            label={`Attack: ${res.attack_cat}`}
                            size="small"
                            color="error"
                            variant="outlined"
                          />
                        )}
                        {res.attack_cat === 'Unknown' && (
                          <Chip
                            label="Attack: Unknown"
                            size="small"
                            color="warning"
                            variant="outlined"
                          />
                        )}
                        {typeof res.prediction === 'number' && res.prediction >= 50 && !res.attack_cat && (
                          <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                            (No attack category model)
                          </Typography>
                        )}
                        <Typography variant="caption" color="text.secondary">
                          Risk: {typeof res.prediction === 'number' ? res.prediction.toFixed(1) : (res.prediction * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Confidence: {(res.confidence * 100).toFixed(1)}%
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

      <Dialog open={deleteConfirmOpen} onClose={handleDeleteCancel}>
        <DialogTitle>Delete Model?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the model "{modelToDelete}"? This will permanently delete the model file, metadata, and all associated files (scaler, attack category model, etc.). This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel} disabled={deleting}>Cancel</Button>
          <Button 
            onClick={handleDeleteConfirm} 
            color="error" 
            variant="contained" 
            disabled={deleting} 
            startIcon={deleting ? <CircularProgress size={16} color="inherit" /> : <DeleteSweepRoundedIcon />}
          >
            {deleting ? 'Deleting…' : 'Delete'}
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
