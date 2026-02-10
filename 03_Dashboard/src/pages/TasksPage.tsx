import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

export default function TasksPage() {
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Typography component="h1" variant="h5" sx={{ mb: 2 }}>
        Tasks
      </Typography>
      <Typography color="text.secondary">
        View and manage system tasks and scheduled jobs.
      </Typography>
    </Box>
  );
}
