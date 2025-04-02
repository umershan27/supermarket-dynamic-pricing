"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const ml_service_adapter_1 = require("./adapters/ml-service.adapter");
const winston_1 = require("winston");
const app = (0, express_1.default)();
const port = 4000;
// Configure CORS
app.use((0, cors_1.default)({
    origin: 'http://localhost:3000',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Accept', 'Origin'],
    exposedHeaders: ['Content-Type'],
    credentials: false,
    maxAge: 86400
}));
// Add OPTIONS handling for preflight requests
app.options('*', (0, cors_1.default)());
// Increase JSON payload limit
app.use(express_1.default.json({ limit: '10mb' }));
app.use(express_1.default.urlencoded({ limit: '10mb', extended: true }));
// Add response type middleware
app.use((req, res, next) => {
    // Add CORS headers to every response
    res.header('Access-Control-Allow-Origin', 'http://localhost:3000');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type, Accept, Origin');
    res.type('application/json');
    next();
});
// Create ML service adapter
const mlService = ml_service_adapter_1.MLServiceAdapter.getInstance();
// Set up Winston logger
const logger = (0, winston_1.createLogger)({
    level: 'info',
    format: winston_1.format.combine(winston_1.format.timestamp(), winston_1.format.colorize(), winston_1.format.simple()),
    transports: [
        new winston_1.transports.Console()
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
    }
    catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        console.error('Health check error:', errorMessage);
        res.status(500).json({
            status: 'error',
            message: errorMessage
        });
    }
});
app.get('/api/godowns', (_req, res, next) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        logger.info('Fetching godown codes');
        const godownCodes = yield mlService.getGodownCodes();
        logger.info(`Retrieved ${godownCodes.length} godown codes`);
        res.json(godownCodes);
    }
    catch (error) {
        logger.error('Error fetching godown codes:', error);
        next(error);
    }
}));
app.get('/api/pricing', (req, res, next) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const { godown, date } = req.query;
        if (!godown || !date) {
            res.status(400).json({ error: 'Missing required parameters' });
            return;
        }
        const results = yield mlService.getPricing(godown, date);
        res.json(results);
    }
    catch (error) {
        logger.error('Error in pricing handler:', error);
        next(error);
    }
}));
app.post('/api/predict', (req, res, next) => __awaiter(void 0, void 0, void 0, function* () {
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
        const results = yield mlService.getPredictions(godown_code, date);
        logger.info('Prediction successful');
        res.json(results);
    }
    catch (error) {
        logger.error('Prediction failed:', error);
        next(error);
    }
}));
// Product names endpoint
app.post('/api/product-names', (req, res, next) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const { item_numbers } = req.body;
        if (!Array.isArray(item_numbers) || item_numbers.length === 0) {
            return res.status(400).json({
                error: 'Invalid request: item_numbers must be a non-empty array'
            });
        }
        console.log('Received product names request for items:', item_numbers);
        const productNames = yield mlService.getProductNames(item_numbers);
        console.log('Found product names:', {
            count: Object.keys(productNames).length,
            sample: Object.entries(productNames).slice(0, 3)
        });
        res.json(productNames);
    }
    catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        console.error('Error fetching product names:', errorMessage);
        res.status(500).json({
            error: `Failed to fetch product names: ${errorMessage}`
        });
    }
}));
app.get('/api/model-metrics', (_req, res, next) => __awaiter(void 0, void 0, void 0, function* () {
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
                { "feature": "margin", "importance": 0.3245 },
                { "feature": "demand", "importance": 0.2512 },
                { "feature": "retail_rate", "importance": 0.1876 },
                { "feature": "mrp", "importance": 0.0987 },
                { "feature": "cost_rate", "importance": 0.0765 },
                { "feature": "hour", "importance": 0.0432 },
                { "feature": "godown_code_encoded", "importance": 0.0134 },
                { "feature": "day_of_week", "importance": 0.0049 }
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
    }
    catch (error) {
        logger.error('Error providing model metrics:', error);
        next(error);
    }
}));
// Error handling middleware
app.use((err, _req, res, _next) => {
    logger.error('Unhandled error:', err);
    res.status(500).json({
        error: 'Internal server error',
        detail: err.message || 'Unknown error'
    });
});
app.listen(port, () => {
    logger.info(`Backend service running on port ${port}`);
});
