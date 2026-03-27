import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import Divider from '@mui/material/Divider';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import TextField from '@mui/material/TextField';

export default function SettingsPage() {
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Typography component="h1" variant="h5" sx={{ mb: 3 }}>
        Settings
      </Typography>

      {/* Display Preferences */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Display Preferences
        </Typography>
        <Stack spacing={2}>
          <FormControlLabel
            control={<Switch defaultChecked={true} disabled />}
            label="Enable Auto Refresh"
          />
          <FormControl sx={{ minWidth: 200 }} disabled>
            <InputLabel>Refresh Interval (seconds)</InputLabel>
            <Select
              value={30}
              label="Refresh Interval (seconds)"
            >
              <MenuItem value={10}>10 seconds</MenuItem>
              <MenuItem value={30}>30 seconds</MenuItem>
              <MenuItem value={60}>1 minute</MenuItem>
              <MenuItem value={300}>5 minutes</MenuItem>
            </Select>
          </FormControl>
        </Stack>
      </Paper>

      {/* Notification Settings */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Notifications
        </Typography>
        <Stack spacing={2}>
          <FormControlLabel
            control={<Switch defaultChecked={true} disabled />}
            label="Enable Anomaly Alerts"
          />
          <Typography variant="body2" color="text.secondary">
            Configured to notify on suspicious network activities.
          </Typography>
        </Stack>
      </Paper>

      {/* Model Settings */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Model Configuration
        </Typography>
        <Stack spacing={2}>
          <Box>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Active Model: 2026-03-13_RFv1
            </Typography>
            <FormControl sx={{ minWidth: 200 }} disabled>
              <InputLabel>Model Selection</InputLabel>
              <Select
                value="rfv1_2026"
                label="Model Selection"
              >
                <MenuItem value="rfv1_2026">2026-03-13 RFv1</MenuItem>
                <MenuItem value="ifv1_2025">2025-12-15 IFv1</MenuItem>
                <MenuItem value="aev1_2025">2025-11-20 AEv1</MenuItem>
              </Select>
            </FormControl>
          </Box>
          <Box>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Detection Threshold: 0.60
            </Typography>
            <TextField
              type="number"
              size="small"
              defaultValue={0.6}
              inputProps={{ min: 0.5, max: 0.9, step: 0.1, disabled: true }}
              disabled
              sx={{ width: 200 }}
            />
          </Box>
        </Stack>
      </Paper>

      {/* Network Configuration */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Network Configuration
        </Typography>
        <Stack spacing={2}>
          <TextField
            label="Data Ingestion Service"
            defaultValue="grpc://localhost:50051"
            fullWidth
            size="small"
            disabled
          />
          <TextField
            label="Backend API Server"
            defaultValue="http://localhost:8002"
            fullWidth
            size="small"
            disabled
          />
          <TextField
            label="Message Queue"
            defaultValue="amqp://localhost:5672"
            fullWidth
            size="small"
            disabled
          />
        </Stack>
      </Paper>

      {/* System Information */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          System Information
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Stack spacing={1}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">Dashboard Version:</Typography>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>1.0.0</Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">Last Configuration Update:</Typography>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>2026-03-27</Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">System Status:</Typography>
            <Typography variant="body2" sx={{ fontWeight: 500, color: 'success.main' }}>Operational</Typography>
          </Box>
        </Stack>
      </Paper>
    </Box>
  );
}
