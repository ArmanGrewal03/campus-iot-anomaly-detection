import { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Dashboard from './dashboard/Dashboard';

// Lazy load pages to avoid Three.js/react-spring initialization issues on direct load
const HomePage = lazy(() => import('./pages/HomePage'));
const TestPage = lazy(() => import('./pages/TestPage'));
const HomeBackupFeb5 = lazy(() => import('./pages/HomeBackupFeb5'));
const ModelPage = lazy(() => import('./pages/ModelPage'));
const AnalyticsPage = lazy(() => import('./pages/AnalyticsPage'));
const ClientsPage = lazy(() => import('./pages/ClientsPage'));
const TasksPage = lazy(() => import('./pages/TasksPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const AboutPage = lazy(() => import('./pages/AboutPage'));

function PageLoader() {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
      <CircularProgress />
    </Box>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />}>
        <Route index element={<Navigate to="/home" replace />} />
        <Route path="home" element={<Suspense fallback={<PageLoader />}><TestPage /></Suspense>} />
        <Route path="test_2026_feb5" element={<Suspense fallback={<PageLoader />}><HomeBackupFeb5 /></Suspense>} />
        <Route path="model" element={<Suspense fallback={<PageLoader />}><ModelPage /></Suspense>} />
        <Route path="analytics" element={<Suspense fallback={<PageLoader />}><AnalyticsPage /></Suspense>} />
        <Route path="clients" element={<Suspense fallback={<PageLoader />}><ClientsPage /></Suspense>} />
        <Route path="tasks" element={<Suspense fallback={<PageLoader />}><TasksPage /></Suspense>} />
        <Route path="settings" element={<Suspense fallback={<PageLoader />}><SettingsPage /></Suspense>} />
        <Route path="about" element={<Suspense fallback={<PageLoader />}><AboutPage /></Suspense>} />
      </Route>
      <Route path="*" element={<Navigate to="/home" replace />} />
    </Routes>
  );
}
