import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Dashboard from './dashboard/Dashboard';

// Lazy load pages to avoid Three.js/react-spring initialization issues on direct load
const TestPage = lazy(() => import('./pages/TestPage'));
const ModelPage = lazy(() => import('./pages/ModelPage'));
const AnalyticsPage = lazy(() => import('./pages/AnalyticsPage'));
const ClientsPage = lazy(() => import('./pages/ClientsPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const AboutPage = lazy(() => import('./pages/AboutPage'));

function PageLoader() {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
      <CircularProgress />
    </Box>
  );
}

// Error boundary to catch render crashes and prevent white screens
class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallbackMessage?: string },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode; fallbackMessage?: string }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 200, gap: 2, p: 3 }}>
          <Typography variant="h6" color="error">
            Something went wrong
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', maxWidth: 500 }}>
            {this.props.fallbackMessage || 'An unexpected error occurred while loading this page. Please try again.'}
          </Typography>
          <Button
            variant="outlined"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Try Again
          </Button>
        </Box>
      );
    }
    return this.props.children;
  }
}

export { ErrorBoundary };

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />}>
        <Route index element={<Navigate to="/home" replace />} />
        <Route path="home" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><TestPage /></Suspense></ErrorBoundary>} />
        <Route path="model" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><ModelPage /></Suspense></ErrorBoundary>} />
        <Route path="analytics" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><AnalyticsPage /></Suspense></ErrorBoundary>} />
        <Route path="clients" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><ClientsPage /></Suspense></ErrorBoundary>} />
        <Route path="settings" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><SettingsPage /></Suspense></ErrorBoundary>} />
        <Route path="about" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><AboutPage /></Suspense></ErrorBoundary>} />
      </Route>
      <Route path="*" element={<Navigate to="/home" replace />} />
    </Routes>
  );
}
