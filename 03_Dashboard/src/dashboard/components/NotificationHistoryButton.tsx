import * as React from 'react';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import ErrorRoundedIcon from '@mui/icons-material/ErrorRounded';
import InfoRoundedIcon from '@mui/icons-material/InfoRounded';
import NotificationsRoundedIcon from '@mui/icons-material/NotificationsRounded';
import MenuButton from './MenuButton';
import {
  clearActivityLog,
  getActivityLog,
  onActivityLogUpdated,
  type ActivityStatus,
  type ActivityEntry,
} from './activityLog';

const LAST_READ_KEY = 'dashboard_activity_last_read';

function formatTime(isoTs: string): string {
  const d = new Date(isoTs);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleString();
}

function getSeverity(entry: ActivityEntry): ActivityStatus {
  if (entry.status) return entry.status;
  const text = entry.message.toLowerCase();
  if (text.includes('failed')) return 'error';
  if (text.includes('completed') || text.includes('trained') || text.includes('uploaded') || text.includes('deleted') || text.includes('changed')) {
    return 'success';
  }
  return 'info';
}

function severityIcon(severity: ActivityStatus) {
  if (severity === 'pending') {
    return <CircularProgress size={16} sx={{ mt: 0.2 }} />;
  }
  if (severity === 'success') return <CheckCircleRoundedIcon color="success" sx={{ fontSize: 18 }} />;
  if (severity === 'error') return <ErrorRoundedIcon color="error" sx={{ fontSize: 18 }} />;
  return <InfoRoundedIcon color="info" sx={{ fontSize: 18 }} />;
}

function statusLabel(status: ActivityStatus): string {
  if (status === 'pending') return 'PENDING';
  if (status === 'success') return 'SUCCESS';
  if (status === 'error') return 'ERROR';
  return 'INFO';
}

export default function NotificationHistoryButton() {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [entries, setEntries] = React.useState<ActivityEntry[]>([]);

  const open = Boolean(anchorEl);

  const loadEntries = React.useCallback(() => {
    setEntries(getActivityLog().filter((entry) => entry.type === 'action'));
  }, []);

  React.useEffect(() => {
    loadEntries();
    return onActivityLogUpdated(loadEntries);
  }, [loadEntries]);

  const unreadCount = React.useMemo(() => {
    const lastRead = Number(localStorage.getItem(LAST_READ_KEY) || '0');
    return entries.filter((entry) => {
      const ts = Date.parse(entry.ts);
      return Number.isFinite(ts) && ts > lastRead;
    }).length;
  }, [entries]);

  const handleOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
    localStorage.setItem(LAST_READ_KEY, String(Date.now()));
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleClear = () => {
    clearActivityLog();
  };

  return (
    <>
      <MenuButton
        showBadge={unreadCount > 0}
        aria-label="Open notification history"
        onClick={handleOpen}
      >
        <NotificationsRoundedIcon />
      </MenuButton>
      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        PaperProps={{ sx: { width: 420, maxWidth: '95vw' } }}
      >
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ px: 2, py: 1 }}>
          <Typography variant="subtitle2" fontWeight={700}>
            Notification History
          </Typography>
          <Button size="small" onClick={handleClear}>Clear</Button>
        </Stack>
        <Divider />
        <Stack sx={{ maxHeight: 360, overflowY: 'auto' }}>
          {entries.length === 0 ? (
            <MenuItem disabled>
              <Typography variant="body2" color="text.secondary">
                No activity yet.
              </Typography>
            </MenuItem>
          ) : (
            entries.map((entry) => {
              const severity = getSeverity(entry);
              return (
              <MenuItem key={entry.id} sx={{ alignItems: 'flex-start' }}>
                <Stack direction="row" spacing={1} sx={{ width: '100%' }}>
                  {severityIcon(severity)}
                  <Stack spacing={0.25} sx={{ minWidth: 0 }}>
                    <Typography variant="body2" fontWeight={600}>
                      {entry.message}
                    </Typography>
                    <Stack direction="row" spacing={1} alignItems="center">
                      {severity === 'pending' && (
                        <Typography variant="caption" color="warning.main" fontWeight={700}>
                          {statusLabel(severity)}
                        </Typography>
                      )}
                      <Typography variant="caption" color="text.secondary">
                        {formatTime(entry.ts)}
                      </Typography>
                    </Stack>
                  </Stack>
                </Stack>
              </MenuItem>
              );
            })
          )}
        </Stack>
      </Menu>
    </>
  );
}
