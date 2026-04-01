import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  TextField,
  Typography,
  Alert,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  LinearProgress,
  Table,
  TableContainer,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
} from '@mui/material';
import BugReportIcon from '@mui/icons-material/BugReport';

interface TestParams {
  datasetName: string;
  modelName: string;
  testSize: string;
}

export default function TestPage() {
  const [params, setParams] = useState<TestParams>({
    datasetName: '',
    modelName: '',
    testSize: '0.2',
  });
  const [datasets, setDatasets] = useState<string[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingData, setLoadingData] = useState(true);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error'>('success');
  const [progress, setProgress] = useState(0);
  const [testResult, setTestResult] = useState<any>(null);

  useEffect(() => {
    fetchDatasets();
    fetchModels();
    const pollInterval = setInterval(() => {
      if (loading) {
        setProgress((prev) => Math.min(prev + 5, 95));
      }
    }, 500);
    return () => clearInterval(pollInterval);
  }, [loading]);

  const fetchDatasets = async () => {
    try {
      const response = await fetch('http://localhost:8004/tables');
      const data = await response.json();
      if (data.tables) {
        // Extract unique dataset names from table names
        const datasetSet = new Set<string>();
        data.tables.forEach((tableName: string) => {
          if (tableName.startsWith('csv_data_')) {
            datasetSet.add(tableName.replace('csv_data_', ''));
          } else if (tableName.startsWith('inserted_data_')) {
            datasetSet.add(tableName.replace('inserted_data_', ''));
          }
        });
        setDatasets(Array.from(datasetSet).sort());
      }
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:8004/models');
      const data = await response.json();
      if (data.models) {
        setModels(data.models);
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
    } finally {
      setLoadingData(false);
    }
  };

  const handleTest = async () => {
    if (!params.datasetName || !params.modelName) {
      setMessage('Please fill in all required fields');
      setMessageType('error');
      return;
    }

    setLoading(true);
    setProgress(10);
    setMessage('');

    try {
      const response = await fetch('http://localhost:8004/test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'dataset_name': params.datasetName,
          'model_name': params.modelName,
        },
        body: JSON.stringify({
          test_size: parseFloat(params.testSize),
        }),
      });

      setProgress(80);
      const data = await response.json();

      if (response.ok) {
        setProgress(100);
        // Extract metrics from nested structure
        const metrics = data.metrics || {};
        setTestResult({
          modelName: params.modelName,
          datasetName: params.datasetName,
          testingSamples: data.testing_samples,
          testingDuration: data.testing_duration_seconds,
          metrics,
        });
        setMessage(`✅ Model tested successfully!`);
        setMessageType('success');
        setTimeout(() => setProgress(0), 500);
      } else {
        setMessage(`❌ Testing failed: ${data.detail || 'Unknown error'}`);
        setMessageType('error');
        setProgress(0);
      }
    } catch (error) {
      setMessage(`❌ Error: ${error}`);
      setMessageType('error');
      setProgress(0);
    } finally {
      setLoading(false);
    }
  };

  if (loadingData) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 'bold' }}>
        <BugReportIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
        Test Model
      </Typography>

      <Card sx={{ mb: 3, maxWidth: 600 }}>
        <CardContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <FormControl fullWidth>
              <InputLabel>Dataset *</InputLabel>
              <Select
                value={params.datasetName}
                label="Dataset *"
                onChange={(e) => setParams({ ...params, datasetName: e.target.value })}
              >
                {datasets.map((ds) => (
                  <MenuItem key={ds} value={ds}>
                    {ds}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Model *</InputLabel>
              <Select
                value={params.modelName}
                label="Model *"
                onChange={(e) => setParams({ ...params, modelName: e.target.value })}
              >
                {models.map((model) => (
                  <MenuItem key={model.model_name} value={model.model_name}>
                    {model.model_name} (Accuracy: {model.accuracy !== undefined && model.accuracy !== null ? (model.accuracy === 0 ? 'Untested' : model.accuracy.toFixed(3)) : 'N/A'})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              label="Test Size"
              type="number"
              value={params.testSize}
              onChange={(e) => setParams({ ...params, testSize: e.target.value })}
              inputProps={{ min: 0.1, max: 0.5, step: 0.1 }}
              fullWidth
            />

            {progress > 0 && (
              <Box>
                <LinearProgress variant="determinate" value={progress} />
                <Typography variant="caption" sx={{ mt: 1 }}>
                  {progress}% Complete
                </Typography>
              </Box>
            )}

            <Button
              variant="contained"
              color="primary"
              onClick={handleTest}
              disabled={loading || !params.datasetName || !params.modelName}
              sx={{ py: 1.5 }}
            >
              {loading ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
              {loading ? 'Testing...' : 'Run Test'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {message && (
        <Alert severity={messageType} sx={{ mb: 2 }}>
          {message}
        </Alert>
      )}

      {testResult && (
        <Card>
          <CardContent>
            <Typography variant="h5" sx={{ mb: 2, fontWeight: 'bold' }}>
              Test Results
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
              Model: <strong>{testResult.modelName}</strong> | Dataset: <strong>{testResult.datasetName}</strong> | Samples: <strong>{testResult.testingSamples}</strong> | Duration: <strong>{testResult.testingDuration}s</strong>
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead sx={{ backgroundColor: '#f5f5f5' }}>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Metric</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                      Value
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(testResult.metrics).map(([key, value]) => (
                    <TableRow key={key}>
                      <TableCell>
                        {key
                          .replace(/_/g, ' ')
                          .split(' ')
                          .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                          .join(' ')}
                      </TableCell>
                      <TableCell align="right">
                        {typeof value === 'number'
                          ? value < 1
                            ? value.toFixed(4)
                            : Math.round(value)
                          : value?.toString() || 'N/A'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
