import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Typography,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import StorageIcon from '@mui/icons-material/Storage';
import VisibilityIcon from '@mui/icons-material/Visibility';

// Direct connection to Model Service (align with Analytics page)
const MODEL_API_BASE = 'http://127.0.0.1:8001';
interface Model {
  model_name: string;
  model_file: string;
  has_metadata: boolean;
  training_date?: string;
  n_features?: number;
  accuracy?: number;
}

interface ModelMetricsResponse {
  status?: string;
  metrics?: Record<string, number>;
  training_params?: {
    loss_history?: { epoch?: number; step?: number; timestamp?: string; loss: number }[];
  };
}

export default function ModelRegistryPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error'>('success');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [viewOpen, setViewOpen] = useState(false);
  const [viewLoading, setViewLoading] = useState(false);
  const [viewModelName, setViewModelName] = useState<string | null>(null);
  const [viewData, setViewData] = useState<ModelMetricsResponse | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${MODEL_API_BASE}/models`);
      const data = await response.json();
      if (data.models) {
        setModels(data.models);
      }
    } catch (error) {
      setMessage(`Failed to fetch models: ${error}`);
      setMessageType('error');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteClick = (modelName: string) => {
    setSelectedModel(modelName);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!selectedModel) return;

    setDeleting(true);
    try {
      const response = await fetch(`http://localhost:8004/models/${selectedModel}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setMessage(`Model "${selectedModel}" deleted successfully!`);
        setMessageType('success');
        fetchModels();
      } else {
        const data = await response.json();
        setMessage(`Delete failed: ${data.detail || 'Unknown error'}`);
        setMessageType('error');
      }
    } catch (error) {
      setMessage(`Error: ${error}`);
      setMessageType('error');
    } finally {
      setDeleting(false);
      setDeleteDialogOpen(false);
      setSelectedModel(null);
    }
  };

  const handleViewClick = async (modelName: string) => {
    setViewModelName(modelName);
    setViewOpen(true);
    setViewLoading(true);
    setViewData(null);
    try {
      const res = await fetch(`${MODEL_API_BASE}/model/metrics`, {
        headers: { 'model_name': modelName }
      });
      const json = (await res.json()) as ModelMetricsResponse;
      if (!res.ok) throw new Error((json as any)?.detail || 'Failed to fetch metrics');
      setViewData(json);
    } catch (e: any) {
      setMessage(`Failed to load model details: ${e?.message || e}`);
      setMessageType('error');
    } finally {
      setViewLoading(false);
    }
  };

  const renderLossChart = (history?: { loss: number }[]) => {
    if (!history || history.length === 0) {
      return <Typography color="text.secondary">No loss history recorded for this model.</Typography>;
    }
    const width = 520;
    const height = 160;
    const padding = 24;
    const xs = history.map((_, i) => i);
    const ys = history.map((h) => h.loss);
    const xMin = 0;
    const xMax = Math.max(1, xs[xs.length - 1]);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const xScale = (x: number) => padding + (x - xMin) / (xMax - xMin || 1) * (width - 2 * padding);
    const yScale = (y: number) => height - padding - (y - yMin) / (yMax - yMin || 1) * (height - 2 * padding);
    const d = xs.map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(ys[i])}`).join(' ');
    return (
      <Box sx={{ mt: 2 }}>
        <svg width={width} height={height}>
          <rect x="0" y="0" width={width} height={height} fill="#fafafa" stroke="#eee" />
          <path d={d} fill="none" stroke="#1976d2" strokeWidth="2" />
          {/* Axis labels */}
          <text x={padding} y={height - 6} fontSize="10" fill="#666">0</text>
          <text x={width - padding - 10} y={height - 6} fontSize="10" fill="#666">{xMax}</text>
          <text x={6} y={yScale(yMax)} fontSize="10" fill="#666">{yMax.toFixed(3)}</text>
          <text x={6} y={yScale(yMin)} fontSize="10" fill="#666">{yMin.toFixed(3)}</text>
        </svg>
        <Typography variant="caption" color="text.secondary">Loss vs. Step</Typography>
      </Box>
    );
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
          <StorageIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Model Registry
        </Typography>
        <Button variant="outlined" onClick={fetchModels} disabled={loading}>
          Refresh
        </Button>
      </Box>

      {message && (
        <Alert severity={messageType} sx={{ mb: 3 }}>
          {message}
        </Alert>
      )}

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
          <CircularProgress />
        </Box>
      ) : (
        <TableContainer component={Card}>
          <Table>
            <TableHead
              sx={{
                backgroundColor: 'transparent', // transparent header background
                '& .MuiTableCell-root': {
                  color: 'text.primary',
                  fontWeight: 700,
                },
              }}
            >
              <TableRow>
                <TableCell sx={{ fontWeight: 'bold' }}>Model Name</TableCell>
                <TableCell align="right">Features</TableCell>
                <TableCell align="right">Accuracy</TableCell>
                <TableCell>Training Date</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {models.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} align="center" sx={{ py: 5 }}>
                    <Typography color="text.secondary">No models available</Typography>
                  </TableCell>
                </TableRow>
              ) : (
                models
                  .filter((m) => m.has_metadata)
                  .map((model) => (
                    <TableRow key={model.model_name} hover>
                      <TableCell>
                        <Typography sx={{ fontWeight: 500, color: 'primary.main' }}>{model.model_name}</Typography>
                      </TableCell>
                      <TableCell align="right">{model.n_features || 'N/A'}</TableCell>
                      <TableCell align="right">
                        <Typography sx={{ fontWeight: 500, color: model.accuracy && model.accuracy > 0.8 ? 'green' : 'inherit' }}>
                          {model.accuracy ? `${(model.accuracy * 100).toFixed(2)}%` : 'N/A'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {model.training_date
                          ? new Date(model.training_date).toLocaleDateString()
                          : 'Unknown'}
                      </TableCell>
                      <TableCell align="center">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleViewClick(model.model_name)}
                          title="View details"
                          sx={{ mr: 1 }}
                        >
                          <VisibilityIcon />
                        </IconButton>
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleDeleteClick(model.model_name)}
                          disabled={deleting}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* View Details Dialog */}
      <Dialog open={viewOpen} onClose={() => setViewOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>{viewModelName || 'Model'} Details</DialogTitle>
        <DialogContent dividers>
          {viewLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          ) : (
            <>
              {/* Metrics */}
              {viewData?.metrics ? (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>Metrics</Typography>
                  <Table size="small">
                    <TableBody>
                      {Object.entries(viewData.metrics).map(([k, v]) => (
                        <TableRow key={k}>
                          <TableCell sx={{ width: 260 }}>{k}</TableCell>
                          <TableCell>{typeof v === 'number' ? v.toFixed(4) : String(v)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Box>
              ) : (
                <Typography color="text.secondary">No metrics available.</Typography>
              )}

              {/* Loss history chart */}
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Training Loss</Typography>
                {renderLossChart(viewData?.training_params?.loss_history?.map(h => ({ loss: h.loss })) )}
              </Box>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Model?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the model <strong>{selectedModel}</strong>? This action
            cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleDeleteConfirm}
            color="error"
            variant="contained"
            disabled={deleting}
          >
            {deleting ? <CircularProgress size={24} /> : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
