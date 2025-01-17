import Stripe from 'stripe';
import { createLogger, format, transports } from 'winston';
import database from './database.js';

const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.Console({
            format: format.combine(
                format.colorize(),
                format.simple()
            )
        })
    ]
});

class PaymentSystem {
    constructor() {
        this.db = database;
        this.stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
        
        this.plans = {
            free: {
                name: 'Gratuito',
                price: 0,
                features: [
                    '50 gerações por dia',
                    'Resolução básica',
                    'Estilos padrão',
                    'Suporte via comunidade'
                ]
            },
            premium: {
                id: 'price_premium_monthly', // ID do produto no Stripe
                name: 'Premium',
                price: 1000, // $10.00
                interval: 'month',
                features: [
                    'Gerações ilimitadas',
                    'Todas as resoluções',
                    'Acesso a recursos beta',
                    'Prioridade na fila',
                    'Suporte prioritário',
                    'Sem marca d\'água',
                    'Acesso antecipado a novos recursos'
                ]
            }
        };
    }
    
    /**
     * Cria ou atualiza cliente no Stripe.
     * @param {string} userId - ID do usuário no Discord
     * @param {string} email - Email do usuário
     * @returns {Promise<string>} - ID do cliente no Stripe
     */
    async createCustomer(userId, email) {
        try {
            // Verifica se já existe
            const userData = await this.db.getUserData(userId);
            
            if (userData?.stripe_customer_id) {
                // Atualiza cliente existente
                const customer = await this.stripe.customers.update(
                    userData.stripe_customer_id,
                    { email }
                );
                return customer.id;
            }
            
            // Cria novo cliente
            const customer = await this.stripe.customers.create({
                email,
                metadata: {
                    discord_id: userId
                }
            });
            
            // Salva ID do cliente
            await this.db.updateUser(userId, {
                stripe_customer_id: customer.id
            });
            
            return customer.id;
            
        } catch (error) {
            logger.error('Erro ao criar/atualizar cliente:', error);
            throw error;
        }
    }
    
    /**
     * Cria sessão de checkout do Stripe.
     * @param {string} userId - ID do usuário no Discord
     * @param {string} email - Email do usuário
     * @returns {Promise<string>} - URL da sessão de checkout
     */
    async createCheckoutSession(userId, email) {
        try {
            // Cria/atualiza cliente
            const customerId = await this.createCustomer(userId, email);
            
            // Cria sessão
            const session = await this.stripe.checkout.sessions.create({
                customer: customerId,
                payment_method_types: ['card'],
                line_items: [{
                    price: this.plans.premium.id,
                    quantity: 1
                }],
                mode: 'subscription',
                success_url: `${process.env.WEBSITE_URL}/success?session_id={CHECKOUT_SESSION_ID}`,
                cancel_url: `${process.env.WEBSITE_URL}/cancel`,
                metadata: {
                    discord_id: userId,
                    plan_id: 'premium'
                },
                allow_promotion_codes: true,
                billing_address_collection: 'required'
            });
            
            // Salva sessão no banco
            await this.db.saveCheckoutSession({
                userId,
                sessionId: session.id,
                planId: 'premium',
                status: 'pending'
            });
            
            return session.url;
            
        } catch (error) {
            logger.error('Erro ao criar sessão de checkout:', error);
            throw error;
        }
    }
    
    /**
     * Processa eventos do webhook do Stripe.
     * @param {Object} event - Evento do Stripe
     * @returns {Promise<boolean>} - True se processado com sucesso
     */
    async handleWebhookEvent(event) {
        try {
            const { type, data } = event;
            
            switch (type) {
                case 'checkout.session.completed':
                    return await this._handleCheckoutCompleted(data.object);
                    
                case 'customer.subscription.updated':
                    return await this._handleSubscriptionUpdated(data.object);
                    
                case 'customer.subscription.deleted':
                    return await this._handleSubscriptionDeleted(data.object);
                    
                case 'invoice.payment_failed':
                    return await this._handlePaymentFailed(data.object);
                    
                default:
                    logger.info(`Evento não processado: ${type}`);
                    return true;
            }
            
        } catch (error) {
            logger.error('Erro ao processar webhook:', error);
            return false;
        }
    }
    
    /**
     * Processa checkout concluído.
     * @private
     */
    async _handleCheckoutCompleted(data) {
        try {
            // Busca dados da sessão
            const sessionData = await this.db.getCheckoutSession(data.id);
            if (!sessionData) {
                throw new Error(`Sessão não encontrada: ${data.id}`);
            }
            
            // Atualiza assinatura
            await this.db.updateSubscription({
                userId: sessionData.user_id,
                stripeSubscriptionId: data.subscription,
                planId: 'premium',
                status: 'active',
                periodStart: new Date(data.period_start * 1000),
                periodEnd: new Date(data.period_end * 1000)
            });
            
            // Atualiza sessão
            await this.db.updateCheckoutSession(data.id, {
                status: 'completed',
                completedAt: new Date()
            });
            
            return true;
            
        } catch (error) {
            logger.error('Erro ao processar checkout completado:', error);
            return false;
        }
    }
    
    /**
     * Cancela assinatura do usuário.
     * @param {string} userId - ID do usuário no Discord
     * @returns {Promise<boolean>} - True se cancelado com sucesso
     */
    async cancelSubscription(userId) {
        try {
            const userData = await this.db.getUserData(userId);
            if (!userData?.subscription?.stripe_subscription_id) {
                return false;
            }
            
            // Cancela no Stripe
            await this.stripe.subscriptions.update(
                userData.subscription.stripe_subscription_id,
                { cancel_at_period_end: true }
            );
            
            // Atualiza banco
            await this.db.updateSubscription({
                userId: userData.id,
                status: 'canceling',
                canceledAt: new Date()
            });
            
            return true;
            
        } catch (error) {
            logger.error('Erro ao cancelar assinatura:', error);
            throw error;
        }
    }
    
    /**
     * Retorna status detalhado da assinatura.
     * @param {string} userId - ID do usuário no Discord
     * @returns {Promise<Object>} - Dados da assinatura
     */
    async getSubscriptionStatus(userId) {
        try {
            const userData = await this.db.getUserData(userId);
            if (!userData?.subscription) {
                return {
                    active: false,
                    plan: this.plans.free,
                    status: 'free',
                    dailyGenerationsLeft: 50 // Limite diário do plano gratuito
                };
            }
            
            return {
                active: userData.subscription.status === 'active',
                plan: this.plans.premium,
                status: userData.subscription.status,
                expiresAt: userData.subscription.current_period_end,
                unlimited: true // Gerações ilimitadas no plano premium
            };
            
        } catch (error) {
            logger.error('Erro ao buscar status da assinatura:', error);
            throw error;
        }
    }
    
    /**
     * Verifica se o usuário pode gerar mais imagens hoje.
     * @param {string} userId - ID do usuário no Discord
     * @returns {Promise<boolean>} - True se pode gerar mais imagens
     */
    async canGenerateMore(userId) {
        try {
            const status = await this.getSubscriptionStatus(userId);
            
            // Plano premium tem gerações ilimitadas
            if (status.unlimited) {
                return true;
            }
            
            // Plano gratuito: verifica limite diário
            const today = new Date().toISOString().split('T')[0];
            const generationsToday = await this.db.getGenerationsCount(userId, today);
            
            return generationsToday < 50; // Limite diário do plano gratuito
            
        } catch (error) {
            logger.error('Erro ao verificar limite de gerações:', error);
            return false;
        }
    }
}

export default new PaymentSystem(); 