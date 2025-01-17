const Stripe = require('stripe');
const { PLANS } = require('../config/plans');

class StripeService {
    constructor() {
        this.stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
    }

    async createCustomer(user) {
        try {
            return await this.stripe.customers.create({
                email: user.email,
                metadata: {
                    discord_id: user.discord_id,
                    user_id: user.id
                }
            });
        } catch (error) {
            console.error('Erro ao criar cliente no Stripe:', error);
            throw error;
        }
    }

    async createCheckoutSession(user, planId, customerId) {
        try {
            const plan = PLANS[planId];
            if (!plan) {
                throw new Error('Plano inválido');
            }

            return await this.stripe.checkout.sessions.create({
                customer: customerId,
                success_url: `${process.env.DISCORD_BOT_URL}/success?session_id={CHECKOUT_SESSION_ID}`,
                cancel_url: `${process.env.DISCORD_BOT_URL}/cancel`,
                mode: 'subscription',
                payment_method_types: ['card'],
                line_items: [{
                    price: plan.price_id,
                    quantity: 1,
                }],
                metadata: {
                    discord_id: user.discord_id,
                    user_id: user.id,
                    plan_id: planId
                },
                subscription_data: {
                    metadata: {
                        discord_id: user.discord_id,
                        user_id: user.id,
                        plan_id: planId
                    }
                }
            });
        } catch (error) {
            console.error('Erro ao criar sessão de checkout:', error);
            throw error;
        }
    }

    async cancelSubscription(subscriptionId) {
        try {
            return await this.stripe.subscriptions.cancel(subscriptionId);
        } catch (error) {
            console.error('Erro ao cancelar assinatura:', error);
            throw error;
        }
    }

    async getSubscription(subscriptionId) {
        try {
            return await this.stripe.subscriptions.retrieve(subscriptionId);
        } catch (error) {
            console.error('Erro ao obter assinatura:', error);
            throw error;
        }
    }

    async updateSubscription(subscriptionId, planId) {
        try {
            const plan = PLANS[planId];
            if (!plan) {
                throw new Error('Plano inválido');
            }

            return await this.stripe.subscriptions.update(subscriptionId, {
                items: [{
                    price: plan.price_id
                }],
                metadata: {
                    plan_id: planId
                }
            });
        } catch (error) {
            console.error('Erro ao atualizar assinatura:', error);
            throw error;
        }
    }

    async getPaymentHistory(customerId) {
        try {
            return await this.stripe.paymentIntents.list({
                customer: customerId,
                limit: 10
            });
        } catch (error) {
            console.error('Erro ao obter histórico de pagamentos:', error);
            throw error;
        }
    }

    async createPortalSession(customerId) {
        try {
            return await this.stripe.billingPortal.sessions.create({
                customer: customerId,
                return_url: process.env.DISCORD_BOT_URL
            });
        } catch (error) {
            console.error('Erro ao criar sessão do portal:', error);
            throw error;
        }
    }
}

module.exports = new StripeService(); 