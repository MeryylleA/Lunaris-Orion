const express = require('express');
const { createLogger, format, transports } = require('winston');
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.File({ filename: 'logs/stripe-webhook.log' })
    ]
});

const router = express.Router();

// Processa eventos do Stripe
router.post('/webhook', express.raw({ type: 'application/json' }), async (req, res) => {
    const sig = req.headers['stripe-signature'];
    let event;

    try {
        // Verifica assinatura do webhook
        event = stripe.webhooks.constructEvent(
            req.body,
            sig,
            process.env.STRIPE_WEBHOOK_SECRET
        );
    } catch (err) {
        logger.error('Erro na assinatura do webhook:', err.message);
        return res.status(400).send(`Webhook Error: ${err.message}`);
    }

    try {
        // Processa eventos
        switch (event.type) {
            case 'checkout.session.completed': {
                const session = event.data.object;
                const userId = session.client_reference_id;

                // Atualiza status da assinatura
                await global.db.createSubscription({
                    userId,
                    stripeCustomerId: session.customer,
                    stripeSubscriptionId: session.subscription,
                    status: 'active',
                    currentPeriodEnd: new Date(session.subscription_data.current_period_end * 1000)
                });

                // Gera nova API key premium
                const apiKey = await global.db.generateApiKey(userId, true);

                logger.info('Assinatura premium ativada:', {
                    userId,
                    subscriptionId: session.subscription
                });

                break;
            }

            case 'customer.subscription.updated': {
                const subscription = event.data.object;
                const userId = subscription.metadata.discord_id;

                await global.db.updateSubscription({
                    userId,
                    status: subscription.status,
                    currentPeriodEnd: new Date(subscription.current_period_end * 1000)
                });

                logger.info('Assinatura atualizada:', {
                    userId,
                    status: subscription.status
                });

                break;
            }

            case 'customer.subscription.deleted': {
                const subscription = event.data.object;
                const userId = subscription.metadata.discord_id;

                // Revoga API key premium
                await global.db.revokeApiKey(userId);

                // Atualiza status
                await global.db.updateSubscription({
                    userId,
                    status: 'canceled',
                    currentPeriodEnd: new Date(subscription.current_period_end * 1000)
                });

                logger.info('Assinatura cancelada:', {
                    userId,
                    subscriptionId: subscription.id
                });

                break;
            }

            case 'invoice.payment_failed': {
                const invoice = event.data.object;
                const userId = invoice.subscription.metadata.discord_id;

                await global.db.updateSubscription({
                    userId,
                    status: 'past_due'
                });

                logger.warn('Falha no pagamento:', {
                    userId,
                    invoiceId: invoice.id
                });

                break;
            }

            default:
                logger.info(`Evento não processado: ${event.type}`);
        }

        res.json({ received: true });
    } catch (err) {
        logger.error('Erro ao processar webhook:', err);
        res.status(500).send('Erro ao processar webhook');
    }
});

module.exports = router; 