import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ListItemIcon from '@mui/material/ListItemIcon';

export default function AboutPage() {
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Typography component="h1" variant="h5" sx={{ mb: 3 }}>
        About Campus IoT Anomaly Detection
      </Typography>

      {/* Project Overview */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Project Overview
        </Typography>
        <Stack spacing={2}>
          <Typography variant="body2" color="text.secondary">
            The Campus IoT Security and Anomaly Detection System is an intelligent security monitoring platform designed to protect university campus IoT devices and network infrastructure from cyber threats.
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This capstone project combines machine learning, real-time data processing, and interactive visualization to detect and alert administrators about suspicious network activity and potential security breaches.
          </Typography>
        </Stack>
      </Paper>

      {/* Key Features */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Key Features
        </Typography>
        <List dense>
          <ListItem>
            <ListItemIcon>
              <CheckCircleIcon sx={{ color: 'success.main' }} />
            </ListItemIcon>
            <ListItemText
              primary="Real-time Anomaly Detection"
              secondary="ML-powered detection using Random Forest classification"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <CheckCircleIcon sx={{ color: 'success.main' }} />
            </ListItemIcon>
            <ListItemText
              primary="Interactive Dashboard"
              secondary="Comprehensive visualization of network metrics and anomalies"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <CheckCircleIcon sx={{ color: 'success.main' }} />
            </ListItemIcon>
            <ListItemText
              primary="Scalable Architecture"
              secondary="Containerized microservices for flexible deployment"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <CheckCircleIcon sx={{ color: 'success.main' }} />
            </ListItemIcon>
            <ListItemText
              primary="Multi-Model Support"
              secondary="Random Forest, Isolation Forest, and Autoencoder models"
            />
          </ListItem>
        </List>
      </Paper>

      {/* Technology Stack */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Technology Stack
        </Typography>
        <Stack spacing={2}>
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>Frontend</Typography>
            <Typography variant="body2" color="text.secondary">React, TypeScript, Material-UI, Vite</Typography>
          </Box>
          <Divider />
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>Backend</Typography>
            <Typography variant="body2" color="text.secondary">Flask, Python, PostgreSQL, gRPC</Typography>
          </Box>
          <Divider />
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>Machine Learning</Typography>
            <Typography variant="body2" color="text.secondary">scikit-learn, TensorFlow, joblib</Typography>
          </Box>
          <Divider />
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>DevOps</Typography>
            <Typography variant="body2" color="text.secondary">Docker, Docker Compose, RabbitMQ, Kafka</Typography>
          </Box>
        </Stack>
      </Paper>

      {/* Team & Contact */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Team & Contact
        </Typography>
        <Stack spacing={2}>
          <Typography variant="body2" color="text.secondary">
            <strong>Developed by:</strong> Arman Grewal, Gurpreet Bhatti, Matthew Ing, Jasdeep Singh
          </Typography>
          <Typography variant="body2" color="text.secondary">
            <strong>Institution:</strong> Toronto Metropolitan University
          </Typography>
          <Typography variant="body2" color="text.secondary">
            <strong>Department:</strong> Electrical & Computer Engineering
          </Typography>
          <Typography variant="body2" color="text.secondary">
            <strong>Contact:</strong> <Link href="mailto:CampusIOT@gmail.com">CampusIOT@gmail.com</Link>
          </Typography>
        </Stack>
      </Paper>

      {/* Version & License */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Version & License
        </Typography>
        <Stack spacing={1}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">Dashboard Version:</Typography>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>1.0.0</Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">Release Date:</Typography>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>Spring 2025</Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="text.secondary">Status:</Typography>
            <Typography variant="body2" sx={{ fontWeight: 500, color: 'success.main' }}>Active Development</Typography>
          </Box>
        </Stack>
      </Paper>
    </Box>
  );
}
