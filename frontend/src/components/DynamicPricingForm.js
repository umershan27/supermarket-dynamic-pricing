import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
} from '@mui/material';
import { styled } from '@mui/material/styles';

// Always use localhost for the browser since browser can't access Docker service names
const API_URL = 'http://localhost:4000';

const Container = styled(Box)({
  maxWidth: 1200,
  margin: '0 auto',
  padding: '24px',
});

const FormPaper = styled(Paper)({
  padding: '24px',
  marginBottom: '24px',
  borderRadius: '8px',
  backgroundColor: 'white',
  boxShadow: 'none',
  border: '1px solid #e0e0e0',
});

const StyledTextField = styled(TextField)({
  '& .MuiOutlinedInput-root': {
    borderRadius: '4px',
    backgroundColor: 'white',
  },
  '& .MuiOutlinedInput-input': {
    padding: '14px',
  },
});

const StyledFormControl = styled(FormControl)({
  '& .MuiOutlinedInput-root': {
    borderRadius: '4px',
    backgroundColor: 'white',
  },
  '& .MuiOutlinedInput-input': {
    padding: '14px',
  },
});

const PredictButton = styled(Button)({
  backgroundColor: '#0074d9',
  color: 'white',
  padding: '12px',
  textTransform: 'uppercase',
  fontWeight: 'bold',
  '&:hover': {
    backgroundColor: '#0063b1',
  },
  borderRadius: '4px',
});

const StyledTableHead = styled(TableHead)({
  '& th': {
    backgroundColor: '#0074d9',
    color: 'white',
    fontWeight: 'bold',
    padding: '12px 16px',
  },
});

// Add a function to check if the backend is available
const checkBackendHealth = async () => {
  try {
    const response = await fetch(`${API_URL}/health`, {
      mode: 'cors',
      credentials: 'omit',
      headers: {
        'Accept': 'application/json'
      }
    });
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    const data = await response.json();
    console.log('Backend health check:', data);
    return true;
  } catch (error) {
    console.error('Backend health check failed:', error);
    return false;
  }
};

const DynamicPricingForm = () => {
  const [formData, setFormData] = useState({
    godown_code: '',
    date: '2025-03-20',
  });

  const [godownCodes, setGodownCodes] = useState([]);
  const [godownsLoading, setGodownsLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [productNames, setProductNames] = useState({});

  // Fetch godown codes on component mount
  useEffect(() => {
    fetchGodownCodes();
  }, []);

  // Fetch product names when predictions change
  useEffect(() => {
    if (predictions && predictions.length > 0) {
      console.log('Predictions updated, fetching product names for:', predictions.length, 'items');
      const itemNos = predictions.map(p => p.item_no).filter(Boolean);
      if (itemNos.length > 0) {
        fetchProductNames(itemNos);
      }
    }
  }, [predictions]);

  // Log product names state changes for debugging
  useEffect(() => {
    if (Object.keys(productNames).length > 0) {
      console.log('Product names updated:', {
        count: Object.keys(productNames).length,
        sample: Object.entries(productNames).slice(0, 3)
      });
    }
  }, [productNames]);

  const fetchGodownCodes = async () => {
    setGodownsLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/godowns`);
      if (!response.ok) {
        throw new Error(`Failed to fetch godown codes: ${response.statusText}`);
      }
      const data = await response.json();
      setGodownCodes(data || []);
      if (data && data.length > 0) {
        setFormData(prev => ({
          ...prev,
          godown_code: data[0].code,
        }));
      }
    } catch (err) {
      console.error('Error fetching godown codes:', err);
      setError(`Failed to fetch godown codes: ${err.message}`);
    } finally {
      setGodownsLoading(false);
    }
  };

  const fetchProductNames = async (itemNos) => {
    if (!itemNos || itemNos.length === 0) {
      console.log('No item numbers provided');
      return;
    }

    try {
      console.log('Fetching product names for:', itemNos);

      const response = await fetch(`${API_URL}/api/product-names`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ item_numbers: itemNos })
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Product names error:', {
          status: response.status,
          statusText: response.statusText,
          error: errorData.error
        });
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log('Product names data received:', {
        count: Object.keys(data).length,
        sample: Object.entries(data).slice(0, 3)
      });

      // Update state with new product names
      setProductNames(prevNames => {
        const newNames = { ...prevNames };
        let foundCount = 0;

        itemNos.forEach(itemNo => {
          if (data[itemNo]) {
            newNames[itemNo] = data[itemNo];
            foundCount++;
          } else {
            newNames[itemNo] = 'Product Not Found';
          }
        });

        console.log(`Updated ${foundCount} out of ${itemNos.length} product names`);
        return newNames;
      });
    } catch (error) {
      console.error('Error fetching product names:', error);
      setError(`Failed to fetch product names: ${error.message}`);

      // Set error state for product names
      setProductNames(prevNames => {
        const newNames = { ...prevNames };
        itemNos.forEach(itemNo => {
          if (!newNames[itemNo]) {
            newNames[itemNo] = 'Error loading name';
          }
        });
        return newNames;
      });
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setProductNames({}); // Clear previous product names

    const requestData = {
      godown_code: formData.godown_code.trim(),
      date: formData.date.split('-').reverse().join('-'), // Convert to DD-MM-YYYY
    };

    console.log('Making prediction request:', requestData);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} - ${errorText || response.statusText}`);
      }

      const data = await response.json();
      console.log('Prediction data:', data);
      setPredictions(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Request failed:', err);
      setError(err.message || 'Failed to fetch predictions');
    } finally {
      setLoading(false);
    }
  };

  const renderPredictionsTable = () => {
    if (!predictions || predictions.length === 0) {
      return null;
    }

    console.log('Rendering predictions table with:', {
      predictions: predictions,
      productNames: productNames,
    });

    return (
      <TableContainer component={Paper}>
        <Table>
          <StyledTableHead>
            <TableRow>
              <TableCell>Item No</TableCell>
              <TableCell>Product Name</TableCell>
              <TableCell>Previous Enter Rate</TableCell>
              <TableCell>Predicted Retail Rate</TableCell>
              <TableCell>Adjusted Price</TableCell>
              <TableCell>Cost Rate</TableCell>
              <TableCell>MRP</TableCell>
            </TableRow>
          </StyledTableHead>
          <TableBody>
            {predictions.map((item, index) => {
              const itemNo = item.item_no;
              const productName = productNames[itemNo];

              return (
                <TableRow key={index}>
                  <TableCell>{itemNo}</TableCell>
                  <TableCell>
                    {productName !== undefined ? (
                      productName
                    ) : (
                      <Box display="flex" alignItems="center">
                        <CircularProgress size={16} sx={{ mr: 1 }} />
                        Loading...
                      </Box>
                    )}
                  </TableCell>
                  <TableCell>{item.previous_enter_rate?.toFixed(2)}</TableCell>
                  <TableCell>{item.predicted_retail_rate?.toFixed(2)}</TableCell>
                  <TableCell>{item.adjusted_price?.toFixed(2)}</TableCell>
                  <TableCell>{item.cost_rate?.toFixed(2)}</TableCell>
                  <TableCell>{item.mrp?.toFixed(2)}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  return (
    <Container>
      <Typography variant="h4" gutterBottom style={{ fontWeight: 'normal' }}>
        Dynamic Price Predictor
      </Typography>

      <FormPaper>
        <form onSubmit={handleSubmit}>
          <Box mb={3}>
            <Typography component="label" style={{ display: 'block', marginBottom: '8px' }}>
              Godown Code *
            </Typography>
            <StyledFormControl fullWidth variant="outlined">
              {godownsLoading ? (
                <CircularProgress size={24} sx={{ margin: '14px auto' }} />
              ) : (
                <Select
                  name="godown_code"
                  value={formData.godown_code}
                  onChange={handleChange}
                  displayEmpty
                  inputProps={{
                    style: {
                      backgroundColor: 'white',
                      fontSize: '14px',
                    },
                  }}
                >
                  <MenuItem value="" disabled>
                    Select a godown
                  </MenuItem>
                  {godownCodes.map((option) => (
                    <MenuItem key={option.code} value={option.code}>
                      {option.code} ({option.count} items)
                    </MenuItem>
                  ))}
                </Select>
              )}
            </StyledFormControl>
          </Box>

          <Box mb={3}>
            <Typography component="label" style={{ display: 'block', marginBottom: '8px' }}>
              Date *
            </Typography>
            <StyledTextField
              fullWidth
              type="date"
              name="date"
              value={formData.date}
              onChange={handleChange}
              variant="outlined"
              inputProps={{
                style: {
                  backgroundColor: 'white',
                  fontSize: '14px',
                },
              }}
            />
          </Box>

          <Box mt={4}>
            <PredictButton
              type="submit"
              variant="contained"
              fullWidth
              disabled={loading || !formData.godown_code}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'PREDICT PRICES'}
            </PredictButton>
          </Box>
        </form>
      </FormPaper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {renderPredictionsTable()}
    </Container>
  );
};

export default DynamicPricingForm;