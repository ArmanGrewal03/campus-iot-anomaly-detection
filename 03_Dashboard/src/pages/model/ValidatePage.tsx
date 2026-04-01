import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Typography,
  Alert,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';

interface ValidationResult {
  status: string;
  message: string;
  total_rows: number;
  training_rows: number;
  testing_rows: number;
  training_percentage: number;
  testing_percentage: number;
  label_0_rows?: number;
  label_1_rows?: number;
  label_0_percentage?: number;
  label_1_percentage?: number;
  errors?: string[];
}

export default function ValidatePage() {
  const [selectedDataset, setSelectedDataset] = useState('');
  const [datasets, setDatasets] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingDatasets, setLoadingDatasets] = useState(true);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    const pollInterval = setInterval(() => {
      if (loading && progress < 90) {
        setProgress((prev) => Math.min(prev + 5, 90));
      }
    }, 300);
    return () => clearInterval(pollInterval);
  }, [loading, progress]);

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
    } finally {
      setLoadingDatasets(false);
    }
  };

  const handleValidate = async () => {
    if (!selectedDataset) {
      return;
    }

    setLoading(true);
    setProgress(10);
    setValidationResult(null);

    try {
      const response = await fetch('http://localhost:8004/validate', {
        method: 'PUT',
        headers: {
          'dataset_name': selectedDataset,
        },
      });

      setProgress(80);
      const data = await response.json();

      if (response.ok) {
        setProgress(100);
        setValidationResult({
          status: 'success',
          message: data.message || 'Validation passed',
          total_rows: data.total_rows || 0,
          training_rows: data.training_rows || 0,
          testing_rows: data.testing_rows || 0,
          training_percentage: data.training_percentage || 0,
          testing_percentage: data.testing_percentage || 0,
          label_0_rows: data.label_0_rows,
          label_1_rows: data.label_1_rows,
          label_0_percentage: data.label_0_percentage,
          label_1_percentage: data.label_1_percentage,
        });
      } else {
        setValidationResult({
          status: 'error',
          message: data.detail || 'Validation failed',
          total_rows: 0,
          training_rows: 0,
          testing_rows: 0,
          training_percentage: 0,
          testing_percentage: 0,
        });
        setProgress(0);
      }
    } catch (error) {
      setValidationResult({
        status: 'error',
        message: `Error: ${error}`,
        total_rows: 0,
        training_rows: 0,
        testing_rows: 0,
        training_percentage: 0,
        testing_percentage: 0,
      });
      setProgress(0);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 'bold' }}>
        Validate Dataset
      </Typography>

      <Card sx={{ mb: 3, maxWidth: 600 }}>
        <CardContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Select a dataset to validate its structure and content before training.
            </Typography>

            <FormControl fullWidth>
              <InputLabel>Dataset *</InputLabel>
              <Select
                value={selectedDataset}
                label="Dataset *"
                onChange={(e) => {
                  setSelectedDataset(e.target.value);
                  setValidationResult(null);
                  setProgress(0);
                }}
              >
                {datasets.length === 0 && !loadingDatasets ? (
                  <MenuItem disabled>No datasets available</MenuItem>
                ) : (
                  datasets.map((ds) => (
                    <MenuItem key={ds} value={ds}>
                      {ds}
                    </MenuItem>
                  ))
                )}
              </Select>
            </FormControl>

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
              onClick={handleValidate}
              disabled={loading || !selectedDataset}
              sx={{ py: 1.5 }}
            >
              {loading ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
              {loading ? 'Validating...' : 'Validate Dataset'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {validationResult && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
              {validationResult.status === 'success' ? (
                <>
                  <CheckCircleIcon sx={{ color: 'green', fontSize: 32 }} />
                  <Alert severity="success" sx={{ flex: 1 }}>
                    {validationResult.message}
                  </Alert>
                </>
              ) : (
                <>
                  <ErrorIcon sx={{ color: 'error.main', fontSize: 32 }} />
                  <Alert severity="error" sx={{ flex: 1 }}>
                    {validationResult.message}
                  </Alert>
                </>
              )}
            </Box>

            {validationResult.status === 'success' && (
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
                    <TableRow>
                      <TableCell>Total Rows</TableCell>
                      <TableCell align="right">{validationResult.total_rows}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Training Rows</TableCell>
                      <TableCell align="right">
                        {validationResult.training_rows} ({validationResult.training_percentage}%)
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Testing Rows</TableCell>
                      <TableCell align="right">
                        {validationResult.testing_rows} ({validationResult.testing_percentage}%)
                      </TableCell>
                    </TableRow>
                    {validationResult.label_0_rows !== undefined && (
                      <>
                        <TableRow>
                          <TableCell>Label 0 Rows</TableCell>
                          <TableCell align="right">
                            {validationResult.label_0_rows} ({validationResult.label_0_percentage}%)
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Label 1 Rows</TableCell>
                          <TableCell align="right">
                            {validationResult.label_1_rows} ({validationResult.label_1_percentage}%)
                          </TableCell>
                        </TableRow>
                      </>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            )}

            {validationResult.errors && validationResult.errors.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: 'error' }}>
                  Errors:
                </Typography>
                {validationResult.errors.map((error: string, idx: number) => (
                  <Typography key={idx} variant="body2" color="error" sx={{ mt: 1 }}>
                    • {error}
                  </Typography>
                ))}
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      <Typography variant="h6" sx={{ mt: 4, mb: 2, fontWeight: 'bold' }}>
        Available Datasets
      </Typography>
      <Card>
        {loadingDatasets ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : datasets.length === 0 ? (
          <CardContent>
            <Typography color="text.secondary">No datasets available</Typography>
          </CardContent>
        ) : (
          <Box sx={{ p: 2 }}>
            {datasets.map((ds) => (
              <Typography key={ds} variant="body2" sx={{ py: 0.5 }}>
                📊 {ds}
              </Typography>
            ))}
          </Box>
        )}
      </Card>
    </Box>
  );
}
