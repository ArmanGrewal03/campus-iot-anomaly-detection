import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import MainGrid from '../dashboard/components/MainGrid';
import IoTSecurityKPISection from '../dashboard/components/IoTSecurityKPISection';

/**
 * Backup copy of the home page (as of 2026-02-05).
 * Keep this unchanged so you can restore from it if needed.
 */
export default function HomeBackupFeb5() {
  return (
    <Box sx={{ width: '100%' }}>
      <Stack spacing={0} sx={{ width: '100%' }}>
        <MainGrid />
        <IoTSecurityKPISection />
      </Stack>
    </Box>
  );
}
