import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

export default function AnalyticsPage() {
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Typography component="h1" variant="h5" sx={{ mb: 2 }}>
        Analytics
      </Typography>
      <Typography color="text.secondary">
        View and analyze your IoT sensor data, trends, and metrics.
      </Typography>
    </Box>
  );
}
