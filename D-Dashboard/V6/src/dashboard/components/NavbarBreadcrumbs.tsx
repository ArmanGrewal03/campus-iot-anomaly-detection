import { useLocation, Link } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import Typography from '@mui/material/Typography';
import MuiLink from '@mui/material/Link';
import Breadcrumbs, { breadcrumbsClasses } from '@mui/material/Breadcrumbs';
import NavigateNextRoundedIcon from '@mui/icons-material/NavigateNextRounded';

const StyledBreadcrumbs = styled(Breadcrumbs)(({ theme }) => ({
  margin: theme.spacing(1, 0),
  [`& .${breadcrumbsClasses.separator}`]: {
    color: (theme.vars || theme).palette.action.disabled,
    margin: 1,
  },
  [`& .${breadcrumbsClasses.ol}`]: {
    alignItems: 'center',
  },
}));

const routeNames: Record<string, string> = {
  '/home': 'Home',
  '/model': 'Model',
  '/analytics': 'Analytics',
  '/clients': 'Clients',
  '/tasks': 'Tasks',
  '/settings': 'Settings',
  '/about': 'About',
};

export default function NavbarBreadcrumbs() {
  const location = useLocation();
  const pathname = location.pathname;
  const currentPage = routeNames[pathname] || 'Home';

  return (
    <StyledBreadcrumbs
      aria-label="breadcrumb"
      separator={<NavigateNextRoundedIcon fontSize="small" />}
    >
      <MuiLink
        component={Link}
        to="/home"
        color="inherit"
        sx={{ textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
      >
        <Typography variant="body1">Dashboard</Typography>
      </MuiLink>
      <Typography variant="body1" sx={{ color: 'text.primary', fontWeight: 600 }}>
        {currentPage}
      </Typography>
    </StyledBreadcrumbs>
  );
}
