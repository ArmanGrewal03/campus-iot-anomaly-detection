import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

export default function ClientsPage() {
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Typography component="h1" variant="h5" sx={{ mb: 2 }}>
        Clients
      </Typography>
      <Typography color="text.secondary">
        Manage connected devices and client configurations.
      </Typography>
    </Box>
  );
}
