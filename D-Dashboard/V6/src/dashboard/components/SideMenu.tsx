import { styled } from '@mui/material/styles';
import Avatar from '@mui/material/Avatar';
import MuiDrawer, { drawerClasses } from '@mui/material/Drawer';
import Box from '@mui/material/Box';
import Divider from '@mui/material/Divider';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import SensorsIcon from '@mui/icons-material/Sensors';
import MenuContent from './MenuContent';
import OptionsMenu from './OptionsMenu';

const drawerWidth = 240;

const Drawer = styled(MuiDrawer)({
  width: drawerWidth,
  flexShrink: 0,
  boxSizing: 'border-box',
  mt: 10,
  [`& .${drawerClasses.paper}`]: {
    width: drawerWidth,
    boxSizing: 'border-box',
  },
});

export default function SideMenu() {
  return (
    <Drawer
      variant="permanent"
      sx={{
        display: { xs: 'none', md: 'block' },
        [`& .${drawerClasses.paper}`]: {
          backgroundColor: 'background.paper',
          overflowX: 'hidden',
        },
      }}
    >
      <Box sx={{ px: 2, pt: 2, pb: 1 }}>
        <Stack direction="row" spacing={1} alignItems="center">
          <SensorsIcon color="primary" sx={{ fontSize: 28 }} />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Campus IoT
          </Typography>
        </Stack>
      </Box>
      <Divider sx={{ mx: 2 }} />
      <Box
        sx={{
          overflow: 'auto',
          overflowX: 'hidden',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          pt: 1,
        }}
      >
        <MenuContent />
      </Box>
      <Stack
        direction="row"
        sx={{
          p: 2,
          gap: 1,
          alignItems: 'center',
          borderTop: '1px solid',
          borderColor: 'divider',
          minWidth: 0,
          overflow: 'hidden',
        }}
      >
        <Avatar
          alt="Campus IoT"
          sx={{ width: 36, height: 36, flexShrink: 0, bgcolor: 'grey.300', color: 'grey.600' }}
        />
        <Box sx={{ mr: 'auto', minWidth: 0, overflow: 'hidden' }}>
          <Typography
            variant="body2"
            sx={{ fontWeight: 500, lineHeight: '16px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
          >
            Campus IOT
          </Typography>
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              display: 'block',
            }}
          >
            CampusIOT@gmail.com
          </Typography>
        </Box>
        <OptionsMenu />
      </Stack>
    </Drawer>
  );
}
