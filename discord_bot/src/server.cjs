const express = require('express');
const stripeWebhook = require('./webhooks/stripe.cjs');

const app = express();

// Stripe webhook needs raw body for signature verification
app.use('/stripe/webhook', express.raw({ type: 'application/json' }));

// For other routes, use regular JSON parsing
app.use(express.json());

// Basic request logging
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
    next();
});

// Use the Stripe webhook router
app.use('/stripe', stripeWebhook);

// Health check endpoint
app.get('/health', (req, res) => {
    console.log('Health check requested');
    res.status(200).json({ 
        status: 'ok',
        timestamp: new Date().toISOString()
    });
});

// Error handling
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({ error: 'Internal Server Error' });
});

const PORT = process.env.PORT || 3000;
const HOST = '0.0.0.0';

app.listen(PORT, HOST, () => {
    console.log(`Server started at ${new Date().toISOString()}`);
    console.log(`Webhook server listening on http://${HOST}:${PORT}`);
    console.log(`Health check available at http://${HOST}:${PORT}/health`);
});

module.exports = app; 