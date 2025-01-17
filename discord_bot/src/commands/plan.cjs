const { SlashCommandBuilder, EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle, PermissionFlagsBits } = require('discord.js');
const { Pool } = require('pg');
const stripeService = require('../services/stripe.cjs');
const { PLANS, SUBSCRIPTION_STATUS } = require('../config/plans.cjs');

const pool = new Pool({
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    ssl: {
        rejectUnauthorized: false
    }
});

async function getUserSubscription(userId) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'SELECT * FROM subscriptions WHERE user_id = $1::bigint ORDER BY created_at DESC LIMIT 1',
            [userId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function createSubscriptionRecord(userId, stripeCustomerId, stripeSubscriptionId, planId, status) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            `INSERT INTO subscriptions (user_id, stripe_customer_id, stripe_subscription_id, plan_id, status)
             VALUES ($1, $2, $3, $4, $5)
             RETURNING *`,
            [userId, stripeCustomerId, stripeSubscriptionId, planId, status]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function updateSubscriptionStatus(subscriptionId, status) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'UPDATE subscriptions SET status = $1 WHERE id = $2 RETURNING *',
            [status, subscriptionId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function handleSubscribe(interaction, planId) {
    try {
        const plan = PLANS[planId];
        if (!plan) {
            await interaction.reply({ 
                content: 'Invalid plan selected!', 
                ephemeral: true 
            });
            return;
        }

        // Check if user already has an active subscription
        const existingSubscription = await getUserSubscription(interaction.user.id);
        if (existingSubscription && existingSubscription.status === 'active') {
            await interaction.reply({
                content: 'You already have an active subscription! Use `/plan cancel` to cancel it first.',
                ephemeral: true
            });
            return;
        }

        await interaction.deferReply({ ephemeral: true });

        // Create or retrieve Stripe customer
        let stripeCustomerId = existingSubscription?.stripe_customer_id;
        if (!stripeCustomerId) {
            const customer = await stripeService.createCustomer({
                email: interaction.user.email,
                discord_id: interaction.user.id,
                id: interaction.user.id
            });
            stripeCustomerId = customer.id;
        }

        // Create checkout session
        const session = await stripeService.createCheckoutSession({
            discord_id: interaction.user.id,
            id: interaction.user.id
        }, planId, stripeCustomerId);

        const row = new ActionRowBuilder()
            .addComponents(
                new ButtonBuilder()
                    .setLabel('Proceed to Checkout')
                    .setStyle(ButtonStyle.Link)
                    .setURL(session.url)
            );

        const embed = new EmbedBuilder()
            .setColor('#0099ff')
            .setTitle('🌟 Premium Plan Checkout')
            .setDescription(`You're about to subscribe to the ${plan.name} plan`)
            .addFields([
                { 
                    name: '💰 Price', 
                    value: `$${plan.price.toFixed(2)}/month`, 
                    inline: true 
                },
                { 
                    name: '✨ Features', 
                    value: plan.features.map(f => `• ${f}`).join('\n'), 
                    inline: false 
                }
            ])
            .setFooter({ 
                text: 'This checkout link will expire in 24 hours',
                iconURL: interaction.client.user.displayAvatarURL()
            })
            .setTimestamp();

        await interaction.editReply({
            embeds: [embed],
            components: [row],
            ephemeral: true
        });

    } catch (error) {
        console.error('Error processing subscription:', error);
        await interaction.editReply({
            content: 'An error occurred while processing your subscription. Please try again later.',
            ephemeral: true
        });
    }
}

async function handleCancel(interaction) {
    try {
        const subscription = await getUserSubscription(interaction.user.id);
        if (!subscription || subscription.status !== 'active') {
            await interaction.reply({
                content: 'You don\'t have an active subscription to cancel!',
                ephemeral: true
            });
            return;
        }

        await interaction.deferReply({ ephemeral: true });

        // Cancel subscription in Stripe
        await stripeService.cancelSubscription(subscription.stripe_subscription_id);

        // Update status in database
        await updateSubscriptionStatus(subscription.id, 'canceled');

        const embed = new EmbedBuilder()
            .setColor('#ff0000')
            .setTitle('Subscription Canceled')
            .setDescription('Your subscription has been successfully canceled.')
            .addFields([
                { 
                    name: '📅 Access Period', 
                    value: 'Your premium access will remain active until the end of your current billing period.' 
                },
                {
                    name: '🔄 Resubscribe',
                    value: 'You can resubscribe at any time using `/plan subscribe`'
                }
            ])
            .setFooter({ 
                text: 'Thank you for trying our premium features!',
                iconURL: interaction.client.user.displayAvatarURL()
            })
            .setTimestamp();

        await interaction.editReply({
            embeds: [embed],
            ephemeral: true
        });

    } catch (error) {
        console.error('Error canceling subscription:', error);
        await interaction.editReply({
            content: 'An error occurred while canceling your subscription. Please try again later.',
            ephemeral: true
        });
    }
}

async function handleStatus(interaction) {
    try {
        await interaction.deferReply({ ephemeral: true });

        const subscription = await getUserSubscription(interaction.user.id);
        if (!subscription) {
            const embed = new EmbedBuilder()
                .setColor('#95a5a6')
                .setTitle('No Active Subscription')
                .setDescription('You currently don\'t have any subscription.')
                .addFields([
                    {
                        name: '🆓 Free Tier',
                        value: 'You are currently on the free tier with limited features.',
                        inline: false
                    },
                    {
                        name: '⭐ Upgrade Now',
                        value: 'Use `/plan subscribe` to upgrade to premium and unlock all features!',
                        inline: false
                    }
                ])
                .setFooter({ 
                    text: 'Unlock unlimited generations with premium!',
                    iconURL: interaction.client.user.displayAvatarURL()
                })
                .setTimestamp();

            await interaction.editReply({
                embeds: [embed],
                ephemeral: true
            });
            return;
        }

        // Get updated subscription details from Stripe
        const stripeSubscription = await stripeService.getSubscription(subscription.stripe_subscription_id);
        const plan = PLANS[subscription.plan_id];
        const status = SUBSCRIPTION_STATUS[stripeSubscription.status];

        const embed = new EmbedBuilder()
            .setColor(status.color)
            .setTitle('Subscription Status')
            .addFields([
                { 
                    name: '📦 Plan', 
                    value: plan.name, 
                    inline: true 
                },
                { 
                    name: '📊 Status', 
                    value: `${status.emoji} ${status.name}`, 
                    inline: true 
                },
                { 
                    name: '💰 Price', 
                    value: `$${plan.price.toFixed(2)}/month`, 
                    inline: true 
                }
            ]);

        if (stripeSubscription.status === 'active') {
            embed.addFields([
                { 
                    name: '📅 Next Payment', 
                    value: `<t:${Math.floor(stripeSubscription.current_period_end)}:R>`, 
                    inline: true 
                }
            ]);
        }

        // Add manage subscription button
        const portalSession = await stripeService.createPortalSession(subscription.stripe_customer_id);
        const row = new ActionRowBuilder()
            .addComponents(
                new ButtonBuilder()
                    .setLabel('Manage Subscription')
                    .setStyle(ButtonStyle.Link)
                    .setURL(portalSession.url)
            );

        await interaction.editReply({
            embeds: [embed],
            components: [row],
            ephemeral: true
        });

    } catch (error) {
        console.error('Error getting subscription status:', error);
        await interaction.editReply({
            content: 'An error occurred while fetching your subscription status. Please try again later.',
            ephemeral: true
        });
    }
}

async function handleHistory(interaction) {
    try {
        const subscription = await getUserSubscription(interaction.user.id);
        if (!subscription) {
            await interaction.reply({
                content: 'You don\'t have any subscription history yet!',
                ephemeral: true
            });
            return;
        }

        await interaction.deferReply({ ephemeral: true });

        // Get payment history
        const payments = await stripeService.getPaymentHistory(subscription.stripe_customer_id);

        const embed = new EmbedBuilder()
            .setColor('#0099ff')
            .setTitle('Payment History')
            .setDescription('Your recent payment history:')
            .addFields(
                payments.data.map(payment => ({
                    name: `Payment on ${new Date(payment.created * 1000).toLocaleDateString()}`,
                    value: `Status: ${payment.status}\nAmount: $${(payment.amount / 100).toFixed(2)}`,
                    inline: false
                }))
            )
            .setFooter({ 
                text: 'Showing your 10 most recent payments',
                iconURL: interaction.client.user.displayAvatarURL()
            })
            .setTimestamp();

        await interaction.editReply({
            embeds: [embed],
            ephemeral: true
        });

    } catch (error) {
        console.error('Error getting payment history:', error);
        await interaction.editReply({
            content: 'An error occurred while fetching your payment history. Please try again later.',
            ephemeral: true
        });
    }
}

module.exports = {
    data: new SlashCommandBuilder()
        .setName('plan')
        .setDescription('Manage your subscription plan')
        .setDMPermission(false)
        .setDefaultMemberPermissions(PermissionFlagsBits.SendMessages)
        .addSubcommand(subcommand =>
            subcommand
                .setName('subscribe')
                .setDescription('Subscribe to the premium plan'))
        .addSubcommand(subcommand =>
            subcommand
                .setName('cancel')
                .setDescription('Cancel your current subscription'))
        .addSubcommand(subcommand =>
            subcommand
                .setName('status')
                .setDescription('Check your subscription status'))
        .addSubcommand(subcommand =>
            subcommand
                .setName('history')
                .setDescription('View your payment history')),

    async execute(interaction) {
        if (!interaction.guild) {
            await interaction.reply({ 
                content: 'This command can only be used in servers!', 
                ephemeral: true 
            });
            return;
        }

        const subcommand = interaction.options.getSubcommand();

        switch (subcommand) {
            case 'subscribe':
                await handleSubscribe(interaction, 'premium');
                break;

            case 'cancel':
                await handleCancel(interaction);
                break;

            case 'status':
                await handleStatus(interaction);
                break;

            case 'history':
                await handleHistory(interaction);
                break;

            default:
                await interaction.reply({
                    content: 'Invalid command!',
                    ephemeral: true
                });
        }
    },
}; 