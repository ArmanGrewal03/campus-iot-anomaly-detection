import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Dashboard from './dashboard/Dashboard';

// Lazy load pages
const UploadPage = lazy(() => import('./pages/UploadPage'));
const ValidatePage = lazy(() => import('./pages/ValidatePage'));
const TrainPage = lazy(() => import('./pages/TrainPage'));
const TestPage = lazy(() => import('./pages/TestPage'));
const ModelRegistryPage = lazy(() => import('./pages/ModelRegistryPage'));

function PageLoader() {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
      <CircularProgress />
    </Box>
  );
}

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
        <Route index element={<Navigate to="/upload" replace />} />
        <Route path="upload" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><UploadPage /></Suspense></ErrorBoundary>} />
        <Route path="validate" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><ValidatePage /></Suspense></ErrorBoundary>} />
        <Route path="train" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><TrainPage /></Suspense></ErrorBoundary>} />
        <Route path="test" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><TestPage /></Suspense></ErrorBoundary>} />
        <Route path="registry" element={<ErrorBoundary><Suspense fallback={<PageLoader />}><ModelRegistryPage /></Suspense></ErrorBoundary>} />
      </Route>
      <Route path="*" element={<Navigate to="/upload" replace />} />
    </Routes>
  );
}
