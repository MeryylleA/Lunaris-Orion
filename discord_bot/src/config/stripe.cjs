const stripe = require('stripe');
const { createLogger, format, transports } = require('winston');

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.File({ filename: 'logs/stripe.log' })
    ]
});

// Configuração do Stripe
const stripeClient = stripe(process.env.STRIPE_SECRET_KEY, {
    apiVersion: '2023-10-16',
    typescript: false,
    maxNetworkRetries: 2,
    timeout: 10000
});

// Produtos e preços
const PRODUCTS = {
    PREMIUM: {
        id: process.env.STRIPE_PREMIUM_PRODUCT_ID,
        name: 'Premium',
        description: 'Plano premium com gerações ilimitadas',
        features: [
            'Gerações ilimitadas',
            'Acesso a recursos beta',
            'Prioridade na fila',
            'Suporte premium'
        ],
        price: {
            id: process.env.STRIPE_PREMIUM_PRICE_ID,
            amount: 1000, // R$ 10,00
            currency: 'brl',
            interval: 'month'
        }
    }
};

// Funções auxiliares
async function createCheckoutSession(userId, email, username) {
    try {
        const session = await stripeClient.checkout.sessions.create({
            payment_method_types: ['card'],
            line_items: [{
                price: PRODUCTS.PREMIUM.price.id,
                quantity: 1
            }],
            mode: 'subscription',
            success_url: `${process.env.DISCORD_BOT_URL}/success?session_id={CHECKOUT_SESSION_ID}`,
            cancel_url: `${process.env.DISCORD_BOT_URL}/cancel`,
            client_reference_id: userId,
            customer_email: email,
            metadata: {
                discord_id: userId,
                username: username
            }
        });

        logger.info('Sessão de checkout criada:', {
            userId,
            sessionId: session.id
        });

        return session;
    } catch (error) {
        logger.error('Erro ao criar sessão de checkout:', error);
        throw error;
    }
}

async function cancelSubscription(subscriptionId) {
    try {
        const subscription = await stripeClient.subscriptions.update(subscriptionId, {
            cancel_at_period_end: true
        });

        logger.info('Assinatura cancelada:', {
            subscriptionId: subscription.id
        });

        return subscription;
    } catch (error) {
        logger.error('Erro ao cancelar assinatura:', error);
        throw error;
    }
}

async function getSubscription(subscriptionId) {
    try {
        const subscription = await stripeClient.subscriptions.retrieve(subscriptionId);
        return subscription;
    } catch (error) {
        logger.error('Erro ao buscar assinatura:', error);
        throw error;
    }
}

module.exports = {
    stripe: stripeClient,
    PRODUCTS,
    createCheckoutSession,
    cancelSubscription,
    getSubscription
}; 