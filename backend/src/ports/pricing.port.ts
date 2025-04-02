import { createLogger, transports, format } from 'winston';
import { PricingResult } from '../domain/models/pricing.model';
import { GodownCode } from '../adapters/ml-service.adapter';

// Set up Winston logger
const logger = createLogger({
  level: 'info',
  format: format.combine(
    format.colorize(),
    format.simple()
  ),
  transports: [
    new transports.Console()
  ]
});

// Example usage of logger
logger.info('Logger initialized successfully');

export interface PricingPort {
  getPricing(godownCode: string, date: string): Promise<PricingResult[]>;
  getPredictions(godownCode: string, date: string): Promise<PricingResult[]>;
  getGodownCodes(): Promise<GodownCode[]>;
  getModelMetrics(): Promise<any>;
}