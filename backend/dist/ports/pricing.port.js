"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const winston_1 = require("winston");
// Set up Winston logger
const logger = (0, winston_1.createLogger)({
    level: 'info',
    format: winston_1.format.combine(winston_1.format.colorize(), winston_1.format.simple()),
    transports: [
        new winston_1.transports.Console()
    ]
});
// Example usage of logger
logger.info('Logger initialized successfully');
