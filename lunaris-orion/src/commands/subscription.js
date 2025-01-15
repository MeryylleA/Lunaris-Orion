const { SlashCommandBuilder, EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle } = require('discord.js');
const { createLogger, format, transports } = require('winston');

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.Console(),
        new transports.File({ filename: 'logs/commands.log' })
    ]
});

// Configuração dos planos
const PLANOS = {
    free: {
        name: 'Gratuito',
        description: 'Plano básico do Lunaris Orion',
        price: 'Grátis',
        features: [
            '50 gerações por dia',
            'Resolução básica',
            'Estilos padrão',
            'Suporte via comunidade'
        ]
    },
    premium: {
        name: 'Premium',
        description: 'Acesso completo ao Lunaris Orion',
        price: 'R$ 10,00/mês',
        price_id: process.env.STRIPE_PREMIUM_PRICE_ID,
        features: [
            'Gerações ilimitadas',
            'Acesso a recursos beta',
            'Prioridade na fila',
            'Suporte prioritário',
            'Sem marca d\'água',
            'Acesso antecipado a novos recursos'
        ]
    }
};

module.exports = {
    data: new SlashCommandBuilder()
        .setName('plano')
        .setDescription('Gerencia sua assinatura do Lunaris Orion')
        .addSubcommand(subcommand =>
            subcommand
                .setName('info')
                .setDescription('Mostra informações do seu plano atual'))
        .addSubcommand(subcommand =>
            subcommand
                .setName('assinar')
                .setDescription('Assina ou atualiza seu plano')),

    async execute(interaction, client) {
        try {
            const subcommand = interaction.options.getSubcommand();

            switch (subcommand) {
                case 'info':
                    await handlePlanInfo(interaction, client);
                    break;
                case 'assinar':
                    await handleSubscribe(interaction, client);
                    break;
            }
        } catch (error) {
            logger.error('Erro ao executar comando de plano:', error);
            await interaction.reply({
                content: 'Erro ao processar comando. Tente novamente mais tarde.',
                ephemeral: true
            });
        }
    }
};

async function handlePlanInfo(interaction, client) {
    try {
        const subscription = await client.getSubscriptionStatus(interaction.user.id);

        const embed = new EmbedBuilder()
            .setTitle('Seu Plano Atual')
            .setColor(subscription?.status === 'active' ? '#00ff00' : '#ff9900');

        if (subscription?.status === 'active') {
            const plano = PLANOS[subscription.tier];
            embed.addFields(
                { name: 'Plano', value: plano.name, inline: true },
                { name: 'Status', value: '✅ Ativo', inline: true },
                { name: 'Expira em', value: new Date(subscription.expires_at).toLocaleDateString('pt-BR'), inline: true },
                { name: 'Recursos Disponíveis', value: plano.features.join('\n'), inline: false }
            );

            // Adiciona botão para cancelar
            const row = new ActionRowBuilder()
                .addComponents(
                    new ButtonBuilder()
                        .setCustomId('cancel_subscription')
                        .setLabel('Cancelar Assinatura')
                        .setStyle(ButtonStyle.Danger)
                );

            await interaction.reply({ embeds: [embed], components: [row], ephemeral: true });
        } else {
            const plano = PLANOS.free;
            embed.addFields(
                { name: 'Plano', value: plano.name, inline: true },
                { name: 'Status', value: '✅ Ativo', inline: true },
                { name: 'Recursos Disponíveis', value: plano.features.join('\n'), inline: false },
                { name: '\u200B', value: 'Use `/plano assinar` para fazer upgrade para o plano Premium!', inline: false }
            );

            await interaction.reply({ embeds: [embed], ephemeral: true });
        }
    } catch (error) {
        logger.error('Erro ao buscar informações do plano:', error);
        throw error;
    }
}

async function handleSubscribe(interaction, client) {
    try {
        const subscription = await client.getSubscriptionStatus(interaction.user.id);

        if (subscription?.status === 'active') {
            await interaction.reply({
                content: 'Você já possui uma assinatura ativa. Use `/plano info` para ver os detalhes.',
                ephemeral: true
            });
            return;
        }

        const plano = PLANOS.premium;
        const embed = new EmbedBuilder()
            .setTitle('Assinar Plano Premium')
            .setDescription(plano.description)
            .setColor('#0099ff')
            .addFields(
                { name: 'Preço', value: plano.price, inline: true },
                { name: 'Recursos Incluídos', value: plano.features.join('\n'), inline: false }
            );

        // Cria sessão do Stripe
        const session = await client.stripe.checkout.sessions.create({
            payment_method_types: ['card'],
            line_items: [{
                price: plano.price_id,
                quantity: 1,
            }],
            mode: 'subscription',
            success_url: `${process.env.WEBSITE_URL}/success?session_id={CHECKOUT_SESSION_ID}`,
            cancel_url: `${process.env.WEBSITE_URL}/cancel`,
            customer_email: interaction.user.email,
            metadata: {
                discord_id: interaction.user.id
            }
        });

        // Salva sessão no banco
        await client.pool.query(
            `INSERT INTO checkout_sessions (user_id, session_id, plan_id, created_at)
            VALUES ((SELECT id FROM users WHERE discord_id = $1), $2, $3, NOW())`,
            [interaction.user.id, session.id, 'premium']
        );

        const row = new ActionRowBuilder()
            .addComponents(
                new ButtonBuilder()
                    .setLabel('Assinar Agora')
                    .setStyle(ButtonStyle.Link)
                    .setURL(session.url)
            );

        await interaction.reply({
            embeds: [embed],
            components: [row],
            ephemeral: true
        });

    } catch (error) {
        logger.error('Erro ao processar assinatura:', error);
        throw error;
    }
} 