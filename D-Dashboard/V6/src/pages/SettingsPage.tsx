import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

export default function SettingsPage() {
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Typography component="h1" variant="h5" sx={{ mb: 2 }}>
        Settings
      </Typography>
      <Typography color="text.secondary">
        Configure dashboard preferences and system settings.
      </Typography>
    </Box>
  );
}
