import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import MainGrid from '../dashboard/components/MainGrid';
import IoTSecurityKPISection from '../dashboard/components/IoTSecurityKPISection';

export default function TestPage() {
  return (
    <Box sx={{ width: '100%' }}>
      <Stack spacing={0} sx={{ width: '100%' }}>
        <MainGrid />
        <IoTSecurityKPISection />
      </Stack>
    </Box>
  );
}
