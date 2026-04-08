import * as React from 'react';
import { styled } from '@mui/material/styles';
import Divider, { dividerClasses } from '@mui/material/Divider';
import Menu from '@mui/material/Menu';
import MuiMenuItem from '@mui/material/MenuItem';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import TextField from '@mui/material/TextField';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import { paperClasses } from '@mui/material/Paper';
import { listClasses } from '@mui/material/List';
import Typography from '@mui/material/Typography';
import ListItemText from '@mui/material/ListItemText';
import ListItemIcon, { listItemIconClasses } from '@mui/material/ListItemIcon';
import LogoutRoundedIcon from '@mui/icons-material/LogoutRounded';
import MoreVertRoundedIcon from '@mui/icons-material/MoreVertRounded';
import MenuButton from './MenuButton';
import { useAuth } from '../../auth/AuthContext';

const MenuItem = styled(MuiMenuItem)({
  margin: '2px 0',
});

export default function OptionsMenu() {
  const { user, isAdmin, login, logoutToGuest } = useAuth();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [loginOpen, setLoginOpen] = React.useState(false);
  const [username, setUsername] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [errorText, setErrorText] = React.useState('');
  const usernameInputRef = React.useRef<HTMLInputElement | null>(null);

  const open = Boolean(anchorEl);
  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };

  const openLoginDialog = () => {
    setErrorText('');
    setUsername('');
    setPassword('');
    setLoginOpen(true);
    handleClose();
  };

  const closeLoginDialog = () => {
    setLoginOpen(false);
    setErrorText('');
  };

  React.useEffect(() => {
    if (!loginOpen) return;
    const timer = setTimeout(() => {
      usernameInputRef.current?.focus();
      usernameInputRef.current?.select();
    }, 60);
    return () => clearTimeout(timer);
  }, [loginOpen]);

  const handleLoginSubmit = () => {
    const result = login(username, password);
    if (result.success) {
      closeLoginDialog();
      return;
    }
    setErrorText(result.message || 'Invalid username or password.');
  };

  const handleLogout = () => {
    logoutToGuest();
    handleClose();
  };

  return (
    <React.Fragment>
      <MenuButton
        aria-label="Open menu"
        onClick={handleClick}
        sx={{ borderColor: 'transparent' }}
      >
        <MoreVertRoundedIcon />
      </MenuButton>
      <Menu
        anchorEl={anchorEl}
        id="menu"
        open={open}
        onClose={handleClose}
        onClick={handleClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        sx={{
          [`& .${listClasses.root}`]: {
            padding: '4px',
          },
          [`& .${paperClasses.root}`]: {
            padding: 0,
          },
          [`& .${dividerClasses.root}`]: {
            margin: '4px -4px',
          },
        }}
      >
        <MenuItem onClick={openLoginDialog}>Login</MenuItem>
        <Divider />
        <MenuItem
          onClick={handleLogout}
          sx={{
            [`& .${listItemIconClasses.root}`]: {
              ml: 'auto',
              minWidth: 0,
            },
          }}
        >
          <ListItemText>Logout</ListItemText>
          <ListItemIcon>
            <LogoutRoundedIcon fontSize="small" />
          </ListItemIcon>
        </MenuItem>
      </Menu>

      <Dialog
        open={loginOpen}
        onClose={closeLoginDialog}
        fullWidth
        maxWidth="xs"
        PaperProps={{
          sx: {
            borderRadius: 2,
            border: '1px solid',
            borderColor: 'divider',
            backgroundImage: 'linear-gradient(180deg, rgba(25,118,210,0.08) 0%, rgba(25,118,210,0.00) 28%)',
          },
        }}
      >
        <DialogTitle sx={{ pb: 1 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6" fontWeight={700}>Secure Login</Typography>
            <Chip size="small" label={isAdmin ? 'ADMIN' : 'GUEST'} color={isAdmin ? 'warning' : 'default'} />
          </Stack>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2 }}>
            Current user: {user.displayName} ({isAdmin ? 'admin' : 'guest'})
          </Typography>
          <TextField
            autoFocus
            inputRef={usernameInputRef}
            margin="dense"
            label="Username"
            fullWidth
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="guest or admin"
          />
          <TextField
            margin="dense"
            label="Password"
            type="password"
            fullWidth
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="guest or admin"
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleLoginSubmit();
              }
            }}
          />
          {!!errorText && (
            <Typography variant="caption" color="error" sx={{ mt: 1, display: 'block' }}>
              {errorText}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={closeLoginDialog}>Cancel</Button>
          <Button variant="contained" onClick={handleLoginSubmit}>Sign in</Button>
        </DialogActions>
      </Dialog>
    </React.Fragment>
  );
}
