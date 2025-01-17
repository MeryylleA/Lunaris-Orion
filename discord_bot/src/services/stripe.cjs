const Stripe = require('stripe');
const { PLANS } = require('../config/plans.cjs');

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

            const guildId = process.env.DISCORD_GUILD_ID;
            const channelId = process.env.DISCORD_CHANNEL_ID;
            const successUrl = `https://discord.com/channels/${guildId}/${channelId}?success=true&session_id={CHECKOUT_SESSION_ID}`;
            const cancelUrl = `https://discord.com/channels/${guildId}/${channelId}?canceled=true`;

            return await this.stripe.checkout.sessions.create({
                customer: customerId,
                success_url: successUrl,
                cancel_url: cancelUrl,
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
            console.error('Error creating checkout session:', error);
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

    async handleWebhook(event) {
        try {
            switch (event.type) {
                case 'customer.subscription.created':
                case 'customer.subscription.updated':
                    const subscription = event.data.object;
                    // Atualizar status da assinatura no banco
                    return {
                        status: subscription.status,
                        customerId: subscription.customer,
                        subscriptionId: subscription.id,
                        planId: subscription.metadata.plan_id,
                        currentPeriodStart: new Date(subscription.current_period_start * 1000),
                        currentPeriodEnd: new Date(subscription.current_period_end * 1000)
                    };

                case 'customer.subscription.deleted':
                    const canceledSubscription = event.data.object;
                    // Marcar assinatura como cancelada no banco
                    return {
                        status: 'canceled',
                        customerId: canceledSubscription.customer,
                        subscriptionId: canceledSubscription.id
                    };

                case 'invoice.payment_succeeded':
                    const invoice = event.data.object;
                    // Registrar pagamento bem-sucedido
                    return {
                        type: 'payment_success',
                        customerId: invoice.customer,
                        subscriptionId: invoice.subscription,
                        amount: invoice.amount_paid,
                        invoiceId: invoice.id
                    };

                case 'invoice.payment_failed':
                    const failedInvoice = event.data.object;
                    // Registrar falha no pagamento
                    return {
                        type: 'payment_failed',
                        customerId: failedInvoice.customer,
                        subscriptionId: failedInvoice.subscription,
                        invoiceId: failedInvoice.id
                    };

                default:
                    console.log(`Evento não tratado: ${event.type}`);
                    return null;
            }
        } catch (error) {
            console.error('Erro ao processar webhook:', error);
            throw error;
        }
    }
}

module.exports = new StripeService(); 