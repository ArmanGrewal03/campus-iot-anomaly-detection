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
  List,
  ListItem,
  ListItemText,
  Divider,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';

interface Dataset {
  name: string;
  rows: number;
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error'>('success');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(true);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedForDelete, setSelectedForDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      setLoadingDatasets(true);
      const response = await fetch('http://localhost:8004/tables');
      const data = await response.json();
      if (data.tables) {
        // Extract unique dataset names from table names
        // Tables are named: csv_data_<dataset_name> and inserted_data_<dataset_name>
        const datasetSet = new Set<string>();
        
        data.tables.forEach((tableName: string) => {
          if (tableName.startsWith('csv_data_')) {
            const datasetName = tableName.replace('csv_data_', '');
            datasetSet.add(datasetName);
          } else if (tableName.startsWith('inserted_data_')) {
            const datasetName = tableName.replace('inserted_data_', '');
            datasetSet.add(datasetName);
          }
        });
        
        // Convert to sorted array for consistent ordering
        const uniqueDatasets = Array.from(datasetSet).sort();
        setDatasets(
          uniqueDatasets.map((name: string) => ({
            name,
            rows: 0,
          }))
        );
      }
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
      setMessage(`Error loading datasets: ${error}`);
      setMessageType('error');
    } finally {
      setLoadingDatasets(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      setFile(files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file || !datasetName) {
      setMessage('Please select a file and enter a dataset name');
      setMessageType('error');
      return;
    }

    setLoading(true);
    setMessage('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8004/upload', {
        method: 'POST',
        headers: {
          'dataset_name': datasetName,
        },
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMessage(`Dataset "${datasetName}" uploaded successfully!`);
        setMessageType('success');
        setFile(null);
        setDatasetName('');
        (document.querySelector('input[type="file"]') as HTMLInputElement).value = '';
        
        // Refresh datasets after short delay
        setTimeout(() => fetchDatasets(), 500);
      } else {
        setMessage(`Upload failed: ${data.detail || 'Unknown error'}`);
        setMessageType('error');
      }
    } catch (error) {
      setMessage(`Error uploading file: ${error}`);
      setMessageType('error');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteClick = (datasetName: string) => {
    setSelectedForDelete(datasetName);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!selectedForDelete) return;

    setDeleting(true);
    try {
      const response = await fetch('http://localhost:8004/clear', {
        method: 'DELETE',
        headers: {
          'dataset_name': selectedForDelete,
        },
      });

      if (response.ok) {
        setMessage(`Dataset "${selectedForDelete}" deleted successfully!`);
        setMessageType('success');
        // Add delay to ensure deletion is fully committed before fetching
        setTimeout(() => fetchDatasets(), 800);
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
      setSelectedForDelete(null);
    }
  };

  return (
    <Box sx={{ p: { xs: 2, md: 3 } }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          Upload Dataset
        </Typography>
        <IconButton
          onClick={fetchDatasets}
          disabled={loadingDatasets}
          title="Refresh datasets"
          sx={(theme) => ({
            transition: 'background-color 0.2s ease, transform 0.15s ease',
            '&:hover': {
              backgroundColor: alpha(theme.palette.info.main, 0.14),
              transform: 'translateY(-1px)',
            },
          })}
        >
          <RefreshIcon />
        </IconButton>
      </Box>

      <Card
        variant="outlined"
        sx={{
          mb: 3,
          maxWidth: 680,
          borderRadius: 2,
          borderColor: 'divider',
          borderLeft: '4px solid',
          borderLeftColor: 'info.main',
          borderTopLeftRadius: 0,
          borderBottomLeftRadius: 0,
        }}
      >
        <CardContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Dataset Name"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="e.g., network_traffic_april"
              fullWidth
            />

            <Box
              sx={(t) => ({
                border: '2px dashed',
                borderColor: file ? 'primary.main' : 'divider',
                borderRadius: 1,
                p: 3,
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                backgroundColor: file ? alpha(t.palette.primary.main, 0.08) : 'transparent',
                '&:hover': {
                  backgroundColor: alpha(t.palette.primary.main, 0.08),
                  borderColor: 'primary.main',
                },
              })}
            >
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                style={{ display: 'none' }}
                id="file-input"
              />
              <label htmlFor="file-input" style={{ cursor: 'pointer', width: '100%', display: 'block' }}>
                <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
                <Typography variant="body1" sx={{ mb: 1 }}>
                  {file ? file.name : 'Click to select CSV file'}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  or drag and drop
                </Typography>
              </label>
            </Box>

            <Button
              variant="contained"
              color="primary"
              onClick={handleUpload}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <CloudUploadIcon />}
              sx={(theme) => ({
                py: 1.5,
                fontWeight: 600,
                boxShadow: 'none !important',
                backgroundColor: '#0284C7 !important',
                backgroundImage: 'none !important',
                color: '#F8FAFC !important',
                border: 'none !important',
                outline: 'none !important',
                transition: 'background-color 0.2s ease, transform 0.15s ease',
                '&:hover:not(.Mui-disabled)': {
                  backgroundColor: '#0369A1 !important',
                  border: 'none !important',
                  outline: 'none !important',
                  boxShadow: 'none !important',
                  transform: 'translateY(-1px)',
                },
                '&:active:not(.Mui-disabled)': {
                  backgroundColor: '#075985 !important',
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
              {loading ? 'Uploading...' : 'Upload Dataset'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {message && <Alert severity={messageType}>{message}</Alert>}

      <Typography variant="h6" sx={{ mt: 4, mb: 2, fontWeight: 700 }}>
        Available Datasets
      </Typography>

      {loadingDatasets ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Card>
          <List>
            {datasets.length === 0 ? (
              <ListItem>
                <ListItemText 
                  primary="No datasets available" 
                  secondary="Upload one to get started"
                />
              </ListItem>
            ) : (
              datasets.map((dataset, idx) => (
                <Box key={dataset.name}>
                  <ListItem
                    secondaryAction={
                      <IconButton
                        edge="end"
                        onClick={() => handleDeleteClick(dataset.name)}
                        disabled={deleting}
                        title="Delete dataset"
                        sx={(t) => ({
                          backgroundColor: '#DC2626',
                          color: '#F8FAFC',
                          border: 'none !important',
                          outline: 'none !important',
                          boxShadow: 'none !important',
                          '&:hover': {
                            backgroundColor: '#B91C1C',
                            boxShadow: 'none !important',
                          },
                          '&:focus, &:focus-visible': {
                            border: 'none !important',
                            outline: 'none !important',
                            boxShadow: 'none !important',
                          },
                          '&.Mui-disabled': {
                            backgroundColor: `${t.palette.mode === 'dark' ? '#374151' : '#D1D5DB'} !important`,
                            color: `${t.palette.mode === 'dark' ? '#F9FAFB' : '#111827'} !important`,
                          },
                        })}
                      >
                        <DeleteIcon />
                      </IconButton>
                    }
                  >
                    <ListItemText
                      primary={dataset.name}
                      secondary={<Chip label="Dataset" size="small" variant="outlined" />}
                    />
                  </ListItem>
                  {idx < datasets.length - 1 && <Divider />}
                </Box>
              ))
            )}
          </List>
        </Card>
      )}

      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Dataset?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete <strong>{selectedForDelete}</strong>? This action cannot be
            undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleDeleteConfirm}
            variant="contained"
            disabled={deleting}
            startIcon={deleting ? <CircularProgress size={16} color="inherit" /> : <DeleteIcon />}
            sx={(theme) => ({
              backgroundColor: '#DC2626 !important',
              backgroundImage: 'none',
              color: '#F8FAFC !important',
              border: 'none !important',
              outline: 'none !important',
              boxShadow: 'none !important',
              transition: 'background-color 0.2s ease, transform 0.15s ease',
              '&:hover:not(.Mui-disabled)': {
                backgroundColor: '#B91C1C !important',
                border: 'none !important',
                outline: 'none !important',
                boxShadow: 'none !important',
                transform: 'translateY(-1px)',
              },
              '&:active:not(.Mui-disabled)': {
                backgroundColor: '#991B1B !important',
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
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
