import React, { useState } from 'react';
import { 
  Container, 
  CssBaseline, 
  ThemeProvider, 
  createTheme, 
  Paper, 
  Tabs, 
  Tab, 
  Box,
  Typography,
  AppBar,
  Toolbar
} from '@mui/material';
import DynamicPricingForm from './components/DynamicPricingForm';
import ModelMetrics from './components/ModelMetrics';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

// TabPanel component for tab content
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Dynamic Pricing ML Dashboard
          </Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Paper sx={{ mb: 3 }} elevation={2}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            variant="fullWidth"
            textColor="primary"
            indicatorColor="primary"
          >
            <Tab label="Price Predictions" />
            <Tab label="Model Performance" />
          </Tabs>
        </Paper>

        <TabPanel value={tabValue} index={0}>
          <DynamicPricingForm />
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <ModelMetrics />
        </TabPanel>
      </Container>
    </ThemeProvider>
  );
}

export default App; 