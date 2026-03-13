import * as React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import { DataGrid, GridColDef, GridActionsCellItem } from '@mui/x-data-grid';
import BlockIcon from '@mui/icons-material/Block';
import LockOpenIcon from '@mui/icons-material/LockOpen';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';

const GATEWAY_BASE = 'http://127.0.0.1:8003'; // API Gateway
const USER_SERVICE_BASE = `${GATEWAY_BASE}`; // User Service via Gateway

export default function ClientsPage() {
  const [users, setUsers] = React.useState<any[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [totalUsers, setTotalUsers] = React.useState(0);
  const [paginationModel, setPaginationModel] = React.useState({ page: 0, pageSize: 25 });
  const [blockDialogOpen, setBlockDialogOpen] = React.useState(false);
  const [selectedUserId, setSelectedUserId] = React.useState<number | null>(null);
  const [blockType, setBlockType] = React.useState<string>('temporarily_blocked');
  const [blockReason, setBlockReason] = React.useState<string>('');
  const [blockDurationHours, setBlockDurationHours] = React.useState<number>(24);
  const [blocking, setBlocking] = React.useState(false);
  const [snackbar, setSnackbar] = React.useState<{ open: boolean; message: string; severity: 'success' | 'error' | 'info' | 'warning' }>({
    open: false,
    message: '',
    severity: 'success',
  });

  const fetchUsers = React.useCallback(async (limit: number = 25, offset: number = 0, skipCache: boolean = false) => {
    setLoading(true);
    try {
      const url = `${USER_SERVICE_BASE}/users?limit=${limit}&offset=${offset}${skipCache ? `&_=${Date.now()}` : ''}`;
      const res = await fetch(url, { cache: skipCache ? 'no-store' : 'default' });
      const json = await res.json() as { 
        status?: string; 
        users?: any[]; 
        total_users?: number;
        returned_users?: number;
        limit?: number;
        offset?: number;
        has_more?: boolean;
        detail?: string 
      };
      
      if (res.ok && json.status === 'success') {
        if (json.users && Array.isArray(json.users)) {
          setUsers(json.users);
          setTotalUsers(json.total_users || json.users.length);
        } else {
          setUsers([]);
          setTotalUsers(0);
          setSnackbar({ open: true, message: 'No users found in response', severity: 'info' });
        }
      } else {
        setUsers([]);
        setTotalUsers(0);
        if (json.detail) {
          setSnackbar({ open: true, message: `Failed to fetch users: ${json.detail}`, severity: 'error' });
        } else {
          setSnackbar({ open: true, message: 'Failed to fetch users: Invalid response format', severity: 'error' });
        }
      }
    } catch (err) {
      console.error('Failed to fetch users:', err);
      setUsers([]);
      setTotalUsers(0);
      setSnackbar({ open: true, message: 'Failed to fetch users. Is the User Service running?', severity: 'error' });
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch users when pagination changes
  React.useEffect(() => {
    const offset = paginationModel.page * paginationModel.pageSize;
    fetchUsers(paginationModel.pageSize, offset);
  }, [paginationModel, fetchUsers]);

  React.useEffect(() => {
    // Initial fetch is handled by pagination effect
  }, [fetchUsers]);

  const handleBlock = (userId: number) => {
    setSelectedUserId(userId);
    setBlockType('temporarily_blocked');
    setBlockReason('');
    setBlockDurationHours(24);
    setBlockDialogOpen(true);
  };

  const handleUnblock = async (userId: number) => {
    setBlocking(true);
    try {
      const res = await fetch(`${USER_SERVICE_BASE}/users/${userId}/unblock`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const json = await res.json();
      if (res.ok) {
        setSnackbar({ open: true, message: `User ${userId} unblocked successfully`, severity: 'success' });
        const offset = paginationModel.page * paginationModel.pageSize;
        fetchUsers(paginationModel.pageSize, offset, true);
      } else {
        setSnackbar({ open: true, message: json.detail || 'Failed to unblock user', severity: 'error' });
      }
    } catch (err) {
      setSnackbar({ open: true, message: 'Failed to unblock user', severity: 'error' });
    } finally {
      setBlocking(false);
    }
  };

  const handleBlockConfirm = async () => {
    if (!selectedUserId) return;
    
    setBlocking(true);
    try {
      const payload: any = {
        block_type: blockType,
      };
      if (blockReason.trim()) {
        payload.block_reason = blockReason.trim();
      }
      if (blockType === 'temporarily_blocked' && blockDurationHours > 0) {
        payload.block_duration_hours = blockDurationHours;
      }

      const res = await fetch(`${USER_SERVICE_BASE}/users/${selectedUserId}/block`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const json = await res.json();
      if (res.ok) {
        setSnackbar({ open: true, message: `User ${selectedUserId} blocked successfully`, severity: 'success' });
        setBlockDialogOpen(false);
        const offset = paginationModel.page * paginationModel.pageSize;
        fetchUsers(paginationModel.pageSize, offset, true);
      } else {
        setSnackbar({ open: true, message: json.detail || 'Failed to block user', severity: 'error' });
      }
    } catch (err) {
      setSnackbar({ open: true, message: 'Failed to block user', severity: 'error' });
    } finally {
      setBlocking(false);
    }
  };

  const columns: GridColDef[] = [
    { field: 'id', headerName: 'ID', width: 80 },
    { field: 'first_name', headerName: 'First Name', flex: 1, minWidth: 120 },
    { field: 'last_name', headerName: 'Last Name', flex: 1, minWidth: 120 },
    {
      field: 'block_status',
      headerName: 'Status',
      width: 150,
      renderCell: (params) => {
        const status = params.value || 'active';
        const color = status === 'active' ? 'success' : status === 'permanently_blocked' ? 'error' : 'warning';
        return <Chip label={status.replace('_', ' ')} color={color} size="small" />;
      },
    },
    {
      field: 'block_type',
      headerName: 'Block Type',
      width: 150,
      renderCell: (params) => params.value || '-',
    },
    {
      field: 'block_until',
      headerName: 'Block Until',
      width: 180,
      renderCell: (params) => {
        if (!params.value) return '-';
        try {
          return new Date(params.value).toLocaleString();
        } catch {
          return params.value;
        }
      },
    },
    {
      field: 'block_reason',
      headerName: 'Block Reason',
      flex: 1,
      minWidth: 150,
      renderCell: (params) => params.value || '-',
    },
    {
      field: 'created_at',
      headerName: 'Created At',
      width: 180,
      renderCell: (params) => {
        if (!params.value) return '-';
        try {
          return new Date(params.value).toLocaleString();
        } catch {
          return params.value;
        }
      },
    },
    {
      field: 'actions',
      type: 'actions',
      headerName: 'Actions',
      width: 150,
      getActions: (params) => {
        const isBlocked = params.row.block_status && params.row.block_status !== 'active';
        return [
          <GridActionsCellItem
            icon={isBlocked ? <LockOpenIcon /> : <BlockIcon />}
            label={isBlocked ? 'Unblock' : 'Block'}
            onClick={() => {
              if (isBlocked) {
                handleUnblock(params.row.id);
              } else {
                handleBlock(params.row.id);
              }
            }}
            disabled={blocking}
          />,
        ];
      },
    },
  ];

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2.5 }}>
        <Stack>
          <Typography component="h1" variant="h5" sx={{ fontWeight: 600 }}>
            Clients
          </Typography>
          <Typography color="text.secondary">
            Manage users and their block status.
          </Typography>
        </Stack>
        <Button
          variant="outlined"
          startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <RefreshRoundedIcon />}
          onClick={() => {
            const offset = paginationModel.page * paginationModel.pageSize;
            fetchUsers(paginationModel.pageSize, offset, true);
          }}
          disabled={loading}
        >
          Refresh
        </Button>
      </Stack>

      <Card variant="outlined">
        <CardContent>
          <Box sx={{ height: 600, width: '100%' }}>
            {loading && users.length === 0 ? (
              <Stack alignItems="center" justifyContent="center" sx={{ height: '100%' }}>
                <CircularProgress />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Loading users...
                </Typography>
              </Stack>
            ) : users.length === 0 ? (
              <Stack alignItems="center" justifyContent="center" sx={{ height: '100%' }}>
                <Typography variant="body2" color="text.secondary">
                  No users found. Check console for API response details.
                </Typography>
              </Stack>
            ) : (
              <DataGrid
                rows={users}
                columns={columns}
                getRowId={(row) => row.id}
                paginationModel={paginationModel}
                onPaginationModelChange={setPaginationModel}
                pageSizeOptions={[10, 25, 50, 100]}
                disableRowSelectionOnClick
                loading={loading}
                rowCount={totalUsers}
                paginationMode="server"
              />
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Block Dialog */}
      <Dialog open={blockDialogOpen} onClose={() => !blocking && setBlockDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Block User {selectedUserId}</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <FormControl fullWidth>
              <InputLabel>Block Type</InputLabel>
              <Select
                value={blockType}
                label="Block Type"
                onChange={(e) => setBlockType(e.target.value)}
                disabled={blocking}
              >
                <MenuItem value="permanently_blocked">Permanently Blocked</MenuItem>
                <MenuItem value="temporarily_blocked">Temporarily Blocked</MenuItem>
                <MenuItem value="rate_limited">Rate Limited</MenuItem>
                <MenuItem value="suspended">Suspended</MenuItem>
                <MenuItem value="quarantined">Quarantined</MenuItem>
                <MenuItem value="other">Other</MenuItem>
              </Select>
            </FormControl>
            {blockType === 'temporarily_blocked' && (
              <TextField
                fullWidth
                type="number"
                label="Duration (hours)"
                value={blockDurationHours}
                onChange={(e) => setBlockDurationHours(Number(e.target.value))}
                disabled={blocking}
                inputProps={{ min: 1 }}
              />
            )}
            <TextField
              fullWidth
              multiline
              rows={3}
              label="Block Reason (optional)"
              value={blockReason}
              onChange={(e) => setBlockReason(e.target.value)}
              disabled={blocking}
              placeholder="Enter reason for blocking this user..."
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBlockDialogOpen(false)} disabled={blocking}>
            Cancel
          </Button>
          <Button
            onClick={handleBlockConfirm}
            variant="contained"
            color="error"
            disabled={blocking}
            startIcon={blocking ? <CircularProgress size={16} color="inherit" /> : <BlockIcon />}
          >
            {blocking ? 'Blocking...' : 'Block User'}
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
