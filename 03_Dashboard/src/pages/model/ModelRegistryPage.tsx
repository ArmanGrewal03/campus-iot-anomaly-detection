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

interface Model {
  model_name: string;
  model_file: string;
  has_metadata: boolean;
  training_date?: string;
  n_features?: number;
  accuracy?: number;
}

export default function ModelRegistryPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error'>('success');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8004/models');
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
            <TableHead sx={{ backgroundColor: '#f5f5f5' }}>
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
                        <Typography sx={{ fontWeight: 500 }}>{model.model_name}</Typography>
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
