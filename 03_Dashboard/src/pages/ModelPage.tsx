import React, { Suspense, lazy, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CircularProgress,
  Stack,
  Tab,
  Tabs,
  Typography,
} from '@mui/material';
import PsychologyRoundedIcon from '@mui/icons-material/PsychologyRounded';

// Lazy load the model management pages
const UploadPage = lazy(() => import('./model/UploadPage'));
const ValidatePage = lazy(() => import('./model/ValidatePage'));
const TrainPage = lazy(() => import('./model/TrainPage'));
const TestPage = lazy(() => import('./model/TestPage'));
const ModelRegistryPage = lazy(() => import('./model/ModelRegistryPage'));

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  const fetchViewData = React.useCallback(
    async (limit: number, offset: number) => {
      if (!selectedDataset.trim()) {
        setSnackbar({ open: true, message: 'Please select a dataset to view.', severity: 'warning' });
        return;
      }
      setViewLoading(true);
      // Don't reset total_rows - keep it to maintain pagination state
      try {
        const headers: Record<string, string> = {};
        headers['dataset_name'] = selectedDataset.trim();
        // Send pagination via headers per API contract
        headers['X-Limit'] = String(limit);
        headers['X-Offset'] = String(offset);
        
        // Determine which endpoint to call based on filterMode
        // Note: These are GET endpoints from Data Ingestion Service, not POST /train from Model Service
        let endpoint = '/view';
        if (filterMode === 'training') {
          endpoint = '/training';  // GET endpoint from Data Ingestion Service
        } else if (filterMode === 'testing') {
          endpoint = '/testing';  // GET endpoint from Data Ingestion Service
        }
        
        const res = await fetch(`${API_BASE}${endpoint}`, { headers });
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
}

function a11yProps(index: number) {
  return {
    id: `model-tab-${index}`,
    'aria-controls': `model-tabpanel-${index}`,
  };
}

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
        const headers: Record<string, string> = {};
        if (selectedDataset.trim()) headers['dataset_name'] = selectedDataset.trim();
        headers['X-Limit'] = '100';
        headers['X-Offset'] = '0';
        const res = await fetch(`${API_BASE}/testing`, { headers });
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

export default function ModelPage() {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
        <Box
          sx={(theme) => ({
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 48,
            height: 48,
            borderRadius: 1.5,
            background: `linear-gradient(145deg, ${theme.palette.primary.main}28 0%, ${theme.palette.primary.dark}12 100%)`,
            border: '1px solid',
            borderColor: 'primary.main',
          })}
        >
          <PsychologyRoundedIcon sx={{ color: 'primary.main' }} />
        </Box>
        <Stack>
          <Typography
            component="h1"
            variant="h4"
            sx={{ fontWeight: 700, letterSpacing: '-0.02em' }}
          >
            Model Management
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Train, validate, test, and manage machine learning models
          </Typography>
        </Stack>
      </Stack>

      {/* Tabs */}
      <Card>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="model management tabs"
          sx={{
            borderBottom: 1,
            borderColor: 'divider',
            px: 2,
            bgcolor: 'background.paper',
          }}
        >
          <Tab label="Upload Data" {...a11yProps(0)} />
          <Tab label="Validate & Split" {...a11yProps(1)} />
          <Tab label="Train Model" {...a11yProps(2)} />
          <Tab label="Test & Predict" {...a11yProps(3)} />
          <Tab label="Model Registry" {...a11yProps(4)} />
        </Tabs>

        <CardContent>
          {/* Upload Tab */}
          <TabPanel value={tabValue} index={0}>
            <Suspense fallback={<PageLoader />}>
              <UploadPage />
            </Suspense>
          </TabPanel>

          {/* Validate & Split Tab */}
          <TabPanel value={tabValue} index={1}>
            <Suspense fallback={<PageLoader />}>
              <ValidatePage />
            </Suspense>
          </TabPanel>

          {/* Train Tab */}
          <TabPanel value={tabValue} index={2}>
            <Suspense fallback={<PageLoader />}>
              <TrainPage />
            </Suspense>
          </TabPanel>

          {/* Test & Predict Tab */}
          <TabPanel value={tabValue} index={3}>
            <Suspense fallback={<PageLoader />}>
              <TestPage />
            </Suspense>
          </TabPanel>

          {/* Model Registry Tab */}
          <TabPanel value={tabValue} index={4}>
            <Suspense fallback={<PageLoader />}>
              <ModelRegistryPage />
            </Suspense>
          </TabPanel>
        </CardContent>
      </Card>
    </Box>
  );
}
