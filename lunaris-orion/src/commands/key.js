const { SlashCommandBuilder } = require('@discordjs/builders');
const { EmbedBuilder } = require('discord.js');
const crypto = require('crypto');
const { createLogger, format, transports } = require('winston');
const apiService = require('../config/api');

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.File({ filename: 'logs/keys.log' })
    ]
});

// Função para gerar chave API segura
function generateApiKey(userId, isPremium = false) {
    const key = crypto.randomBytes(32).toString('hex');
    const prefix = isPremium ? 'lop_' : 'lo_';
    return `${prefix}${key}`;
}

module.exports = {
    data: new SlashCommandBuilder()
        .setName('key')
        .setDescription('Gerencia sua chave API do Lunaris Orion')
        .addSubcommand(subcommand =>
            subcommand
                .setName('gerar')
                .setDescription('Gera uma nova chave API')
        )
        .addSubcommand(subcommand =>
            subcommand
                .setName('status')
                .setDescription('Verifica o status da sua chave API')
        )
        .addSubcommand(subcommand =>
            subcommand
                .setName('revogar')
                .setDescription('Revoga sua chave API atual')
        ),

    async execute(interaction, client) {
        const subcommand = interaction.options.getSubcommand();
        const userId = interaction.user.id;

        try {
            switch (subcommand) {
                case 'gerar': {
                    // Verifica se o usuário já tem uma chave ativa
                    const currentKey = await client.db.getUserApiKey(userId);
                    if (currentKey && currentKey.active) {
                        return interaction.reply({
                            content: 'Você já possui uma chave API ativa. Use `/key revogar` primeiro se deseja gerar uma nova.',
                            ephemeral: true
                        });
                    }

                    // Verifica status da assinatura
                    const subscription = await client.db.getSubscriptionStatus(userId);
                    const isPremium = subscription && subscription.status === 'active';

                    // Gera nova chave
                    const apiKey = generateApiKey(userId, isPremium);
                    
                    // Sincroniza com o servidor de geração
                    await apiService.syncApiKey(userId, apiKey, isPremium);
                    
                    // Salva no banco local
                    await client.db.saveApiKey(userId, apiKey, isPremium);

                    // Log da geração
                    logger.info('Nova chave API gerada e sincronizada', {
                        userId,
                        isPremium,
                        keyPrefix: apiKey.substring(0, 4)
                    });

                    // Cria embed com a chave
                    const embed = new EmbedBuilder()
                        .setColor(isPremium ? '#FFD700' : '#00FF00')
                        .setTitle('🔑 Sua Nova Chave API')
                        .setDescription('Guarde esta chave em um local seguro!')
                        .addFields(
                            { name: 'Chave', value: `\`${apiKey}\`` },
                            { name: 'Tipo', value: isPremium ? '✨ Premium' : '🔰 Padrão' },
                            { name: 'Limite Diário', value: isPremium ? '♾️ Ilimitado' : '50 gerações' },
                            { name: 'Instruções', value: 'Cole esta chave nas configurações do plugin do Aseprite' }
                        )
                        .setFooter({ text: 'Esta mensagem será excluída em 1 minuto por segurança' });

                    // Envia a chave em mensagem privada que se auto-destrói
                    await interaction.user.send({ embeds: [embed] })
                        .then(msg => {
                            setTimeout(() => msg.delete(), 60000);
                        });

                    return interaction.reply({
                        content: '✅ Chave API gerada e sincronizada com sucesso! Verifique suas mensagens privadas.',
                        ephemeral: true
                    });
                }

                case 'status': {
                    const key = await client.db.getUserApiKey(userId);
                    if (!key) {
                        return interaction.reply({
                            content: 'Você não possui uma chave API. Use `/key gerar` para criar uma.',
                            ephemeral: true
                        });
                    }

                    // Verifica status no servidor de geração
                    const apiStatus = await apiService.checkKeyStatus(key.key);
                    const subscription = await client.db.getSubscriptionStatus(userId);

                    const embed = new EmbedBuilder()
                        .setColor(key.is_premium ? '#FFD700' : '#00FF00')
                        .setTitle('🔑 Status da Chave API')
                        .addFields(
                            { name: 'Status', value: key.active ? '✅ Ativa' : '❌ Revogada' },
                            { name: 'Tipo', value: key.is_premium ? '✨ Premium' : '🔰 Padrão' },
                            { name: 'Gerações Hoje', value: `${apiStatus.daily_uses || 0}/${key.is_premium ? '∞' : '50'}` },
                            { name: 'Criada em', value: new Date(key.created_at).toLocaleDateString() }
                        );

                    if (subscription) {
                        embed.addFields({
                            name: 'Assinatura',
                            value: `✨ Premium até ${new Date(subscription.current_period_end).toLocaleDateString()}`
                        });
                    }

                    return interaction.reply({
                        embeds: [embed],
                        ephemeral: true
                    });
                }

                case 'revogar': {
                    const key = await client.db.getUserApiKey(userId);
                    if (!key || !key.active) {
                        return interaction.reply({
                            content: 'Você não possui uma chave API ativa para revogar.',
                            ephemeral: true
                        });
                    }

                    // Revoga no servidor de geração
                    await apiService.revokeApiKey(key.key);
                    
                    // Revoga no banco local
                    await client.db.revokeApiKey(userId);

                    // Log da revogação
                    logger.info('Chave API revogada em todos os sistemas', {
                        userId,
                        keyPrefix: key.key.substring(0, 4)
                    });

                    return interaction.reply({
                        content: '✅ Sua chave API foi revogada com sucesso em todos os sistemas. Use `/key gerar` para criar uma nova.',
                        ephemeral: true
                    });
                }
            }
        } catch (error) {
            logger.error('Erro ao executar comando key:', error);
            return interaction.reply({
                content: 'Ocorreu um erro ao processar o comando. Por favor, tente novamente.',
                ephemeral: true
            });
        }
    }
}; 