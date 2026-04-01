import React, { Suspense, lazy, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CircularProgress,
  Stack,
  Tab,
  Tabs,
  Typography,
} from '@mui/material';
import PsychologyRoundedIcon from '@mui/icons-material/PsychologyRounded';

// Lazy load the model management pages
const UploadPage = lazy(() => import('./model/UploadPage'));
const ValidatePage = lazy(() => import('./model/ValidatePage'));
const TrainPage = lazy(() => import('./model/TrainPage'));
const TestPage = lazy(() => import('./model/TestPage'));
const ModelRegistryPage = lazy(() => import('./model/ModelRegistryPage'));

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`model-tabpanel-${index}`}
      aria-labelledby={`model-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `model-tab-${index}`,
    'aria-controls': `model-tabpanel-${index}`,
  };
}

function PageLoader() {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
      <CircularProgress />
    </Box>
  );
}

export default function ModelPage() {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
        <Box
          sx={(theme) => ({
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 48,
            height: 48,
            borderRadius: 1.5,
            background: `linear-gradient(145deg, ${theme.palette.primary.main}28 0%, ${theme.palette.primary.dark}12 100%)`,
            border: '1px solid',
            borderColor: 'primary.main',
          })}
        >
          <PsychologyRoundedIcon sx={{ color: 'primary.main' }} />
        </Box>
        <Stack>
          <Typography
            component="h1"
            variant="h4"
            sx={{ fontWeight: 700, letterSpacing: '-0.02em' }}
          >
            Model Management
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Train, validate, test, and manage machine learning models
          </Typography>
        </Stack>
      </Stack>

      {/* Tabs */}
      <Card>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="model management tabs"
          sx={{
            borderBottom: 1,
            borderColor: 'divider',
            px: 2,
            bgcolor: 'background.paper',
          }}
        >
          <Tab label="Upload Data" {...a11yProps(0)} />
          <Tab label="Validate & Split" {...a11yProps(1)} />
          <Tab label="Train Model" {...a11yProps(2)} />
          <Tab label="Test & Predict" {...a11yProps(3)} />
          <Tab label="Model Registry" {...a11yProps(4)} />
        </Tabs>

        <CardContent>
          {/* Upload Tab */}
          <TabPanel value={tabValue} index={0}>
            <Suspense fallback={<PageLoader />}>
              <UploadPage />
            </Suspense>
          </TabPanel>

          {/* Validate & Split Tab */}
          <TabPanel value={tabValue} index={1}>
            <Suspense fallback={<PageLoader />}>
              <ValidatePage />
            </Suspense>
          </TabPanel>

          {/* Train Tab */}
          <TabPanel value={tabValue} index={2}>
            <Suspense fallback={<PageLoader />}>
              <TrainPage />
            </Suspense>
          </TabPanel>

          {/* Test & Predict Tab */}
          <TabPanel value={tabValue} index={3}>
            <Suspense fallback={<PageLoader />}>
              <TestPage />
            </Suspense>
          </TabPanel>

          {/* Model Registry Tab */}
          <TabPanel value={tabValue} index={4}>
            <Suspense fallback={<PageLoader />}>
              <ModelRegistryPage />
            </Suspense>
          </TabPanel>
        </CardContent>
      </Card>
    </Box>
  );
}
