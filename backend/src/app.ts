import express from 'express';
import type { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { MLServiceAdapter, GodownCode } from './adapters/ml-service.adapter';
import { createLogger, transports, format } from 'winston';
import { PricingResult } from './domain/models/pricing.model';

type PricingQuery = {
    godown?: string;
    date?: string;
};

type PredictionBody = {
    godown_code: string;
    date: string;
};

type ProductNamesBody = {
    item_numbers: string[];
};

const app = express();
const port = 4000;

// Configure CORS
app.use(cors({
  origin: 'http://localhost:3000',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Accept', 'Origin'],
  exposedHeaders: ['Content-Type'],
  credentials: false,
  maxAge: 86400
}));

// Add OPTIONS handling for preflight requests
app.options('*', cors());

// Increase JSON payload limit
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ limit: '10mb', extended: true }));

// Add response type middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  // Add CORS headers to every response
  res.header('Access-Control-Allow-Origin', 'http://localhost:3000');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Accept, Origin');
  res.type('application/json');
  next();
});

// Create ML service adapter
const mlService = MLServiceAdapter.getInstance();

// Set up Winston logger
const logger = createLogger({
    level: 'info',
    format: format.combine(
        format.timestamp(),
        format.colorize(),
        format.simple()
    ),
    transports: [
        new transports.Console()
    ]
});

// Example usage of logger
logger.info('Logger initialized successfully');

// Health check endpoint
app.get('/health', (req, res) => {
  try {
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: '1.0.0'
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Health check error:', errorMessage);
    res.status(500).json({
      status: 'error',
      message: errorMessage
    });
  }
});

app.get('/api/godowns', async (_req: Request, res: Response, next: NextFunction) => {
    try {
        logger.info('Fetching godown codes');
        const godownCodes = await mlService.getGodownCodes();
        logger.info(`Retrieved ${godownCodes.length} godown codes`);
        res.json(godownCodes);
    } catch (error) {
        logger.error('Error fetching godown codes:', error);
        next(error);
    }
});

app.get('/api/pricing', async (req: Request<{}, any, any, PricingQuery>, res: Response, next: NextFunction) => {
    try {
        const { godown, date } = req.query;
        if (!godown || !date) {
            res.status(400).json({ error: 'Missing required parameters' });
            return;
        }
        const results = await mlService.getPricing(godown, date);
        res.json(results);
    } catch (error) {
        logger.error('Error in pricing handler:', error);
        next(error);
    }
});

app.post('/api/predict', async (req: Request<{}, any, PredictionBody>, res: Response, next: NextFunction) => {
    try {
        const { godown_code, date } = req.body;
        logger.info('Received prediction request:', { godown_code, date });
        
        if (!godown_code || !date) {
            logger.warn('Missing required parameters');
            res.status(400).json({ 
                error: 'Missing required parameters',
                detail: 'Both godown_code and date are required'
            });
            return;
        }
        
        const results = await mlService.getPredictions(godown_code, date);
        logger.info('Prediction successful');
        res.json(results);
    } catch (error) {
        logger.error('Prediction failed:', error);
        next(error);
    }
});

// Product names endpoint
app.post('/api/product-names', async (req: Request<{}, any, ProductNamesBody>, res: Response, next: NextFunction) => {
  try {
    const { item_numbers } = req.body;
    
    if (!Array.isArray(item_numbers) || item_numbers.length === 0) {
      return res.status(400).json({
        error: 'Invalid request: item_numbers must be a non-empty array'
      });
    }

    console.log('Received product names request for items:', item_numbers);
    const productNames = await mlService.getProductNames(item_numbers);
    
    console.log('Found product names:', {
      count: Object.keys(productNames).length,
      sample: Object.entries(productNames).slice(0, 3)
    });

    res.json(productNames);
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Error fetching product names:', errorMessage);
    res.status(500).json({
      error: `Failed to fetch product names: ${errorMessage}`
    });
  }
});

app.get('/api/model-metrics', async (_req: Request, res: Response, next: NextFunction) => {
    try {
        logger.info('Providing updated model metrics data');
        
        // Set explicit CORS headers
        res.header('Access-Control-Allow-Origin', '*');
        res.header('Access-Control-Allow-Methods', 'GET');
        res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
        
        // Updated model metrics data with the correct values
        const metrics = {
            "model_type": "XGBRegressor",
            "features": ["hour", "day_of_week", "demand", "margin", "godown_code_encoded", "cost_rate", "retail_rate", "mrp"],
            "feature_importance": [
                {"feature": "margin", "importance": 0.3245},
                {"feature": "demand", "importance": 0.2512},
                {"feature": "retail_rate", "importance": 0.1876},
                {"feature": "mrp", "importance": 0.0987},
                {"feature": "cost_rate", "importance": 0.0765},
                {"feature": "hour", "importance": 0.0432},
                {"feature": "godown_code_encoded", "importance": 0.0134},
                {"feature": "day_of_week", "importance": 0.0049}
            ],
            "metrics": {
                "rmse": 19.8764,
                "mae": 1.7897,
                "r2": 0.9142,
                "accuracy_within_5_percent": 0.8676,
                "accuracy_within_10_percent": 0.9292,
                "accuracy_within_20_percent": 0.9588,
                "sample_size": 10000
            },
            "data_stats": {
                "total_records": 17392805,
                "unique_items": 17929
            }
        };
        logger.info('Updated model metrics provided successfully');
        res.json(metrics);
    } catch (error) {
        logger.error('Error providing model metrics:', error);
        next(error);
    }
});

// Error handling middleware
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
    logger.error('Unhandled error:', err);
    res.status(500).json({ 
        error: 'Internal server error',
        detail: err.message || 'Unknown error'
    });
});

app.listen(port, () => {
    logger.info(`Backend service running on port ${port}`);
});

export { PricingResult };

export interface PricingPort {
    getPricing(godownCode: string, date: string): Promise<PricingResult[]>;
} 