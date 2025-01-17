const { SlashCommandBuilder } = require('@discordjs/builders');
const { EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle } = require('discord.js');
const { createLogger, format, transports } = require('winston');

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.File({ filename: 'logs/premium.log' })
    ]
});

// Preços e produtos
const PREMIUM_PRICE = 1000; // R$ 10,00
const PREMIUM_PRODUCT = 'price_premium_monthly';

module.exports = {
    data: new SlashCommandBuilder()
        .setName('premium')
        .setDescription('Gerencia sua assinatura premium')
        .addSubcommand(subcommand =>
            subcommand
                .setName('assinar')
                .setDescription('Assina o plano premium')
        )
        .addSubcommand(subcommand =>
            subcommand
                .setName('status')
                .setDescription('Verifica o status da sua assinatura')
        )
        .addSubcommand(subcommand =>
            subcommand
                .setName('cancelar')
                .setDescription('Cancela sua assinatura premium')
        ),

    async execute(interaction, client) {
        const subcommand = interaction.options.getSubcommand();
        const userId = interaction.user.id;

        try {
            switch (subcommand) {
                case 'assinar': {
                    // Verifica se já tem assinatura ativa
                    const subscription = await client.db.getSubscriptionStatus(userId);
                    if (subscription && subscription.status === 'active') {
                        return interaction.reply({
                            content: '✨ Você já possui uma assinatura premium ativa!',
                            ephemeral: true
                        });
                    }

                    // Cria sessão de checkout do Stripe
                    const session = await client.stripe.checkout.sessions.create({
                        payment_method_types: ['card'],
                        line_items: [{
                            price: PREMIUM_PRODUCT,
                            quantity: 1,
                        }],
                        mode: 'subscription',
                        success_url: `${process.env.DISCORD_BOT_URL}/success?session_id={CHECKOUT_SESSION_ID}`,
                        cancel_url: `${process.env.DISCORD_BOT_URL}/cancel`,
                        client_reference_id: userId,
                        customer_email: interaction.user.email,
                        metadata: {
                            discord_id: userId,
                            username: interaction.user.tag
                        }
                    });

                    // Cria embed com informações
                    const embed = new EmbedBuilder()
                        .setColor('#FFD700')
                        .setTitle('✨ Assinatura Premium')
                        .setDescription('Clique no botão abaixo para assinar!')
                        .addFields(
                            { name: 'Preço', value: 'R$ 10,00/mês' },
                            { name: 'Benefícios', value: '• Gerações ilimitadas\n• Acesso a recursos beta\n• Prioridade na fila\n• Suporte premium' }
                        );

                    // Cria botão de pagamento
                    const row = new ActionRowBuilder()
                        .addComponents(
                            new ButtonBuilder()
                                .setLabel('Assinar Premium')
                                .setStyle(ButtonStyle.Link)
                                .setURL(session.url)
                                .setEmoji('✨')
                        );

                    return interaction.reply({
                        embeds: [embed],
                        components: [row],
                        ephemeral: true
                    });
                }

                case 'status': {
                    const subscription = await client.db.getSubscriptionStatus(userId);
                    if (!subscription) {
                        const embed = new EmbedBuilder()
                            .setColor('#FF0000')
                            .setTitle('Status da Assinatura')
                            .setDescription('Você não possui uma assinatura premium.')
                            .addFields(
                                { name: 'Plano Atual', value: '🔰 Gratuito' },
                                { name: 'Limite', value: '50 gerações por dia' }
                            );

                        return interaction.reply({
                            embeds: [embed],
                            ephemeral: true
                        });
                    }

                    const embed = new EmbedBuilder()
                        .setColor('#FFD700')
                        .setTitle('Status da Assinatura')
                        .setDescription('✨ Assinatura Premium Ativa')
                        .addFields(
                            { name: 'Status', value: subscription.status === 'active' ? '✅ Ativa' : '❌ Inativa' },
                            { name: 'Renovação', value: new Date(subscription.current_period_end).toLocaleDateString() },
                            { name: 'Plano', value: '✨ Premium' },
                            { name: 'Gerações', value: 'Ilimitadas' }
                        );

                    return interaction.reply({
                        embeds: [embed],
                        ephemeral: true
                    });
                }

                case 'cancelar': {
                    const subscription = await client.db.getSubscriptionStatus(userId);
                    if (!subscription || subscription.status !== 'active') {
                        return interaction.reply({
                            content: 'Você não possui uma assinatura premium ativa para cancelar.',
                            ephemeral: true
                        });
                    }

                    // Cancela a assinatura no Stripe
                    await client.stripe.subscriptions.update(subscription.stripe_subscription_id, {
                        cancel_at_period_end: true
                    });

                    // Atualiza no banco
                    await client.db.updateSubscriptionStatus(userId, 'canceling');

                    // Log do cancelamento
                    logger.info('Assinatura cancelada', {
                        userId,
                        subscriptionId: subscription.stripe_subscription_id
                    });

                    const embed = new EmbedBuilder()
                        .setColor('#FF0000')
                        .setTitle('Assinatura Cancelada')
                        .setDescription('Sua assinatura premium foi cancelada.')
                        .addFields(
                            { name: 'Acesso até', value: new Date(subscription.current_period_end).toLocaleDateString() },
                            { name: 'Observação', value: 'Você ainda terá acesso premium até o fim do período atual.' }
                        );

                    return interaction.reply({
                        embeds: [embed],
                        ephemeral: true
                    });
                }
            }
        } catch (error) {
            logger.error('Erro ao executar comando premium:', error);
            return interaction.reply({
                content: 'Ocorreu um erro ao processar o comando. Por favor, tente novamente.',
                ephemeral: true
            });
        }
    }
}; 