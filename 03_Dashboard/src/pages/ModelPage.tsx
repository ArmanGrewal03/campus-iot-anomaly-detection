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
import ChevronLeftRoundedIcon from '@mui/icons-material/ChevronLeftRounded';
import ChevronRightRoundedIcon from '@mui/icons-material/ChevronRightRounded';
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

const API_BASE = 'http://127.0.0.1:8000'; // Data Ingestion Service
const MODEL_API_BASE = 'http://127.0.0.1:8001'; // Model Service

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
  const [datasetStats, setDatasetStats] = React.useState<any>(null);
  const [typeStats, setTypeStats] = React.useState<any>(null);
  const [statsLoading, setStatsLoading] = React.useState(false);
  const [availableDatasets, setAvailableDatasets] = React.useState<string[]>([]);
  const [selectedViewDataset, setSelectedViewDataset] = React.useState<string>('');
  const [datasetsLoading, setDatasetsLoading] = React.useState(false);
  const [selectedValidateDataset, setSelectedValidateDataset] = React.useState<string>('');
  const [selectedStatsDataset, setSelectedStatsDataset] = React.useState<string>('');
  const [availableFields, setAvailableFields] = React.useState<string[]>([]);
  const [fieldsLoading, setFieldsLoading] = React.useState(false);
  const [availableModels, setAvailableModels] = React.useState<any[]>([]);
  const [modelsLoading, setModelsLoading] = React.useState(false);
  const [selectedTestDataset, setSelectedTestDataset] = React.useState<string>('');
  const [selectedTestModel, setSelectedTestModel] = React.useState<string>('');
  const [testing, setTesting] = React.useState(false);
  const [testResults, setTestResults] = React.useState<any>(null);
  const [modelStatuses, setModelStatuses] = React.useState<Record<string, any>>({});
  const [modelMetrics, setModelMetrics] = React.useState<Record<string, any>>({});
  const [currentModelIndex, setCurrentModelIndex] = React.useState(0);
  const [modelDetailsLoading, setModelDetailsLoading] = React.useState(false);


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
        headers['dataset_name'] = datasetName.trim();
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
        // Auto-select first dataset if none selected
        setSelectedViewDataset((prev) => {
          if (!prev && datasetNames.length > 0) {
            return datasetNames[0];
          }
          return prev;
        });
        // Also auto-select for validation if none selected
        setSelectedValidateDataset((prev) => {
          if (!prev && datasetNames.length > 0) {
            return datasetNames[0];
          }
          return prev;
        });
        // Also auto-select for stats if none selected
        setSelectedStatsDataset((prev) => {
          if (!prev && datasetNames.length > 0) {
            return datasetNames[0];
          }
          return prev;
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
    if (selectedViewDataset) {
      fetchFields(selectedViewDataset);
    } else if (datasetName.trim()) {
      fetchFields(datasetName);
    } else {
      setAvailableFields([]);
    }
  }, [selectedViewDataset, datasetName, fetchFields]);

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
    const datasetToUse = metrics?.dataset || selectedStatsDataset || selectedViewDataset || datasetName.trim();
    if (datasetToUse) {
      fetchDatasetStats(datasetToUse);
    }
  }, [selectedStatsDataset, selectedViewDataset, datasetName, metrics?.dataset, fetchDatasetStats]);

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

  const handleTest = async () => {
    if (!selectedTestModel.trim()) {
      setSnackbar({ open: true, message: 'Please select a model to test.', severity: 'warning' });
      return;
    }

    const testDataset = selectedTestDataset.trim() || selectedViewDataset.trim() || datasetName.trim();
    if (!testDataset) {
      setSnackbar({ open: true, message: 'Please select a dataset to test on.', severity: 'warning' });
      return;
    }

    setTesting(true);
    setTestResults(null);

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
      setSnackbar({ 
        open: true, 
        message: `Model "${selectedTestModel.trim()}" tested successfully on dataset "${testDataset}"`, 
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
      if (!selectedViewDataset.trim()) {
        setSnackbar({ open: true, message: 'Please select a dataset to view.', severity: 'warning' });
        return;
      }
      setViewLoading(true);
      setViewTotalRows(null);
      try {
        const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
        const headers: Record<string, string> = {};
        headers['dataset_name'] = selectedViewDataset.trim();
        
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
    [selectedViewDataset, filterMode]
  );

  React.useEffect(() => {
    if (selectedViewDataset) {
      fetchViewData(viewLimit, 0);
    }
  }, [viewLimit, selectedViewDataset, filterMode, fetchViewData]);

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
    const interval = setInterval(() => fetchApiHealth(true), 15000);
    return () => clearInterval(interval);
  }, [fetchApiHealth]);

  const handleViewLimitChange = (newLimit: number) => {
    setViewLimit(newLimit);
  };

  const [validationResult, setValidationResult] = React.useState<{ message: string; severity: 'success' | 'warning' } | null>(null);

  const handleRevalidate = async () => {
    // Use selectedValidateDataset if available, otherwise fall back to datasetName or selectedViewDataset
    const validateDataset = selectedValidateDataset.trim() || selectedViewDataset.trim() || datasetName.trim();
    if (!validateDataset) {
      setSnackbar({ open: true, message: 'Please select a dataset to validate.', severity: 'warning' });
      return;
    }

    setValidating(true);
    setValidationResult(null);
    try {
      const headers: Record<string, string> = {
        'dataset_name': validateDataset,
      };
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
    // /insert endpoint was removed from backend
    setSnackbar({ open: true, message: 'Insert endpoint is no longer available. Please upload data via CSV file upload.', severity: 'info' });
  };

  const handleClearConfirm = async () => {
    setClearLoading(true);
    try {
      const headers: Record<string, string> = {};
      if (datasetName.trim()) headers['dataset_name'] = datasetName.trim();
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
      fetchViewData(viewLimit, 0);
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
    
    // Use selectedViewDataset if available, otherwise fall back to datasetName
    const trainingDataset = selectedViewDataset.trim() || datasetName.trim();
    if (!trainingDataset) {
      setSnackbar({ open: true, message: 'Please select a dataset to train on.', severity: 'warning' });
      return;
    }

    setTraining(true);
    setMetrics(null);

    try {
      // TrainRequest body structure
      const payload: any = {
        n_estimators: 100,
        max_depth: null,
        random_state: 42,
      };
      
      // Only include include_fields if fields are selected
      if (selectedFeatures.length > 0) {
        payload.include_fields = selectedFeatures;
      }
      // exclude_fields is optional, don't include if not needed

      // Headers: dataset_name and model_name are required
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'dataset_name': trainingDataset,
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
      // Refresh models list after training
      fetchModels();
      setSnackbar({ 
        open: true, 
        message: `Model "${modelName.trim()}" trained successfully on dataset "${trainingDataset}"`, 
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
        if (datasetName.trim()) headers['dataset_name'] = datasetName.trim();
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

      // Transform backend response to match frontend expectations
      const transformedResults = (result.predictions || []).map((pred: any, idx: number) => ({
        index: idx,
        label: pred.label || (pred.prediction === 0 ? 'normal' : 'anomaly'),
        confidence: pred.confidence || pred.probability_safe || 0,
        prediction: pred.prediction,
        probability_safe: pred.probability_safe,
        probability_unsafe: pred.probability_unsafe,
      }));
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
                    value={selectedViewDataset}
                    label="Dataset"
                    onChange={(e) => setSelectedViewDataset(e.target.value)}
                    disabled={datasetsLoading || availableDatasets.length === 0}
                  >
                    {availableDatasets.map((ds) => (
                      <MenuItem key={ds} value={ds}>{ds}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
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
                  onClick={() => fetchViewData(viewLimit, 0)}
                  disabled={viewLoading || !selectedViewDataset}
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
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, display: 'block', mb: 0.75 }}>
                  Revalidate Dataset
                </Typography>
                <Stack direction="row" spacing={1} sx={{ mb: 1.5 }}>
                  <FormControl size="small" sx={{ minWidth: 200 }}>
                    <InputLabel id="validate-dataset-label">Dataset to Validate</InputLabel>
                    <Select
                      labelId="validate-dataset-label"
                      value={selectedValidateDataset}
                      label="Dataset to Validate"
                      onChange={(e) => setSelectedValidateDataset(e.target.value)}
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
                    disabled={validating || !selectedValidateDataset}
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
                    value={selectedViewDataset || datasetName.trim() || ''}
                    label="Dataset for Training"
                    onChange={(e) => {
                      // Update selectedViewDataset if it exists in availableDatasets, otherwise update datasetName
                      if (availableDatasets.includes(e.target.value)) {
                        setSelectedViewDataset(e.target.value);
                      } else {
                        setDatasetName(e.target.value);
                      }
                    }}
                    disabled={datasetsLoading || availableDatasets.length === 0}
                  >
                    {availableDatasets.map((ds) => (
                      <MenuItem key={ds} value={ds}>{ds}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <TextField
                  fullWidth
                  size="small"
                  label="Model Name"
                  placeholder="e.g., model_v1"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  error={!!modelNameError}
                  helperText={modelNameError || 'Required: Unique name for this model'}
                  sx={{ mb: 2 }}
                />
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
                  sx={{ mb: 2 }}
                >
                  <MenuItem value="Random Forest">Random Forest (rfV1)</MenuItem>
                  <MenuItem value="Isolation Forest">Isolation Forest</MenuItem>
                  <MenuItem value="Autoencoder">Autoencoder (MLP)</MenuItem>
                </TextField>
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
                    {selectedViewDataset || datasetName.trim() 
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
                disabled={training || !modelName.trim() || (!selectedViewDataset.trim() && !datasetName.trim())}
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
                      value={selectedStatsDataset}
                      label="Select Dataset"
                      onChange={(e) => setSelectedStatsDataset(e.target.value)}
                      disabled={datasetsLoading || availableDatasets.length === 0}
                    >
                      {availableDatasets.map((ds) => (
                        <MenuItem key={ds} value={ds}>{ds}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>
                {/* Show dataset stats even without model metrics */}
                {(datasetStats || typeStats) && selectedStatsDataset && (
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
                {!selectedStatsDataset && availableDatasets.length > 0 && (
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
                    value={selectedTestDataset}
                    label="Dataset for Testing"
                    onChange={(e) => setSelectedTestDataset(e.target.value)}
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
                  disabled={testing || !selectedTestModel.trim() || (!selectedTestDataset.trim() && !selectedViewDataset.trim() && !datasetName.trim())}
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
                        {currentModelMetrics.metrics.f1 != null && (
                          <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                              <Typography variant="caption" color="text.secondary" display="block">F1 Score</Typography>
                              <Typography variant="h6">{(Number(currentModelMetrics.metrics.f1) * 100).toFixed(1)}%</Typography>
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
