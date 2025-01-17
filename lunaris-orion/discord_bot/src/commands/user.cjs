const { SlashCommandBuilder, EmbedBuilder } = require('discord.js');
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

module.exports = {
    data: new SlashCommandBuilder()
        .setName('user')
        .setDescription('Exibe informações do usuário'),

    async execute(interaction, client) {
        try {
            // Busca dados do usuário
            const userData = await client.pool.query(
                `SELECT u.*, s.tier, s.status, s.expires_at, k.key_hash
                FROM users u
                LEFT JOIN subscriptions s ON u.id = s.user_id
                LEFT JOIN api_keys k ON u.id = k.user_id
                WHERE u.discord_id = $1`,
                [interaction.user.id]
            );

            let user = userData.rows[0];

            if (!user) {
                // Cria novo usuário
                const newUser = await client.pool.query(
                    `INSERT INTO users (discord_id, created_at)
                    VALUES ($1, NOW())
                    RETURNING *`,
                    [interaction.user.id]
                );
                user = newUser.rows[0];
            }

            // Cria embed informativo
            const embed = new EmbedBuilder()
                .setTitle('Informações do Usuário')
                .setColor('#0099ff');

            embed.addFields({
                name: 'Status da Assinatura',
                value: user.status || 'Sem assinatura',
                inline: false
            });

            if (user.status === 'active') {
                embed.addFields(
                    {
                        name: 'Plano',
                        value: user.tier === 'premium' ? 'Premium' : 'Gratuito',
                        inline: true
                    },
                    {
                        name: 'Expira em',
                        value: new Date(user.expires_at).toLocaleDateString('pt-BR'),
                        inline: true
                    }
                );
            }

            embed.addFields({
                name: 'Chave API',
                value: user.key_hash ? '✅ Ativa' : '❌ Não gerada',
                inline: false
            });

            // Verifica uso diário (para plano gratuito)
            if (!user.status || user.status !== 'active') {
                const today = new Date().toISOString().split('T')[0];
                const usageData = await client.pool.query(
                    `SELECT COUNT(*) as count
                    FROM generations
                    WHERE user_id = $1
                    AND DATE(created_at) = $2`,
                    [user.id, today]
                );

                embed.addFields({
                    name: 'Gerações Hoje',
                    value: `${usageData.rows[0].count}/50`,
                    inline: true
                });
            }

            await interaction.reply({ embeds: [embed], ephemeral: true });

        } catch (error) {
            logger.error('Erro ao buscar informações do usuário:', error);
            await interaction.reply({
                content: 'Erro ao buscar informações. Tente novamente mais tarde.',
                ephemeral: true
            });
        }
    },
}; 