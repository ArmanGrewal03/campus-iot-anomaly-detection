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
} from '@mui/material';
import SchoolIcon from '@mui/icons-material/School';

interface TrainParams {
  datasetName: string;
  modelName: string;
  modelType: string;
  epochs: string;
  testSize: string;
}

export default function TrainPage() {
  const [params, setParams] = useState<TrainParams>({
    datasetName: '',
    modelName: '',
    modelType: 'RFv1',
    epochs: '10',
    testSize: '0.2',
  });
  const [datasets, setDatasets] = useState<string[]>([]);
  const [modelTypes, setModelTypes] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingData, setLoadingData] = useState(true);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error'>('success');
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    fetchDatasets();
    fetchModelTypes();
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

  const fetchModelTypes = async () => {
    try {
      const response = await fetch('http://localhost:8004/model-types');
      const data = await response.json();
      if (data.model_types) {
        setModelTypes(data.model_types.map((mt: any) => mt.model_type));
      }
    } catch (error) {
      console.error('Failed to fetch model types:', error);
    } finally {
      setLoadingData(false);
    }
  };

  const handleTrain = async () => {
    if (!params.datasetName || !params.modelName) {
      setMessage('Please fill in all required fields');
      setMessageType('error');
      return;
    }

    setLoading(true);
    setProgress(10);
    setMessage('');

    try {
      const response = await fetch('http://localhost:8004/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'dataset_name': params.datasetName,
          'model_name': params.modelName,
        },
        body: JSON.stringify({
          model_type: params.modelType,
          epochs: parseInt(params.epochs),
          test_size: parseFloat(params.testSize),
        }),
      });

      setProgress(80);
      const data = await response.json();

      if (response.ok) {
        setProgress(100);
        setMessage(
          `Model "${params.modelName}" trained successfully!\n` +
          `Dataset: ${params.datasetName} | Model type: ${params.modelType} | Features: ${data.n_features}\n` +
          `Training duration: ${data.training_duration_seconds}s`
        );
        setMessageType('success');
        setParams({ ...params, modelName: '', datasetName: '' });
        setTimeout(() => setProgress(0), 500);
      } else {
        setMessage(`Training failed: ${data.detail || 'Unknown error'}`);
        setMessageType('error');
        setProgress(0);
      }
    } catch (error) {
      setMessage(`Error: ${error}`);
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
        <SchoolIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
        Train Model
      </Typography>

      <Card
        sx={{
          mb: 3,
          maxWidth: 600,
          borderLeft: '4px solid',
          borderLeftColor: '#F59E0B',
          borderTopLeftRadius: 0,
          borderBottomLeftRadius: 0,
        }}
      >
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

            <TextField
              label="Model Name"
              value={params.modelName}
              onChange={(e) => setParams({ ...params, modelName: e.target.value })}
              placeholder="e.g., model_v1"
              fullWidth
            />

            <FormControl fullWidth>
              <InputLabel>Model Type</InputLabel>
              <Select
                value={params.modelType}
                label="Model Type"
                onChange={(e) => setParams({ ...params, modelType: e.target.value })}
              >
                {modelTypes.map((mt) => (
                  <MenuItem key={mt} value={mt}>
                    {mt}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              label="Epochs"
              type="number"
              value={params.epochs}
              onChange={(e) => setParams({ ...params, epochs: e.target.value })}
              inputProps={{ min: 1, max: 100 }}
              fullWidth
            />

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
              onClick={handleTrain}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <SchoolIcon />}
              sx={(theme) => ({
                py: 1.5,
                fontWeight: 600,
                boxShadow: 'none !important',
                backgroundColor: '#F59E0B !important',
                backgroundImage: 'none',
                color: '#F8FAFC !important',
                border: 'none !important',
                outline: 'none !important',
                transition: 'background-color 0.2s ease, transform 0.15s ease',
                '&:hover:not(.Mui-disabled)': {
                  backgroundColor: '#D97706 !important',
                  border: 'none !important',
                  outline: 'none !important',
                  boxShadow: 'none !important',
                  transform: 'translateY(-1px)',
                },
                '&:active:not(.Mui-disabled)': {
                  backgroundColor: '#B45309 !important',
                  border: 'none !important',
                  outline: 'none !important',
                  boxShadow: 'none !important',
                  transform: 'translateY(0)',
                },
                '&:focus, &:focus-visible': {
                  border: 'none !important',
                  outline: 'none !important',
                  boxShadow: 'none !important',
                },
                '&.Mui-disabled': {
                  color: `${theme.palette.mode === 'dark' ? '#F9FAFB' : '#111827'} !important`,
                  backgroundColor: `${theme.palette.mode === 'dark' ? '#374151' : '#D1D5DB'} !important`,
                  backgroundImage: 'none !important',
                  border: '1px solid',
                  borderColor: theme.palette.mode === 'dark' ? '#4B5563' : '#9CA3AF',
                  opacity: '1 !important',
                },
              })}
            >
              {loading ? 'Training...' : 'Start Training'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {message && (
        <Alert severity={messageType} sx={{ whiteSpace: 'pre-wrap' }}>
          {message}
        </Alert>
      )}
    </Box>
  );
}
