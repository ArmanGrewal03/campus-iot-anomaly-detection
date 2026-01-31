import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link';

export default function AboutPage() {
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Typography component="h1" variant="h5" sx={{ mb: 2 }}>
        About
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 2 }}>
        Campus IoT Anomaly Detection Dashboard
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Contact: <Link href="mailto:CampusIOT@gmail.com">CampusIOT@gmail.com</Link>
      </Typography>
    </Box>
  );
}
