const { SlashCommandBuilder, EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle, ModalBuilder, TextInputBuilder, TextInputStyle } = require('discord.js');
const { Pool } = require('pg');
const { createLogger, format, transports } = require('winston');

// Configuração do logger
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
        }),
        new transports.File({ filename: 'logs/generations.log' })
    ]
});

// Configuração do pool de conexão
const pool = new Pool({
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    ssl: process.env.DB_SSL === 'true'
});

// Estilos disponíveis
const STYLES = {
    pixel_art: { name: 'Pixel Art', emoji: '🎮' },
    pixel_art_16bit: { name: '16-bit', emoji: '🕹️' },
    pixel_art_32bit: { name: '32-bit', emoji: '🖼️' },
    isometric: { name: 'Isométrico', emoji: '📐' },
    cyberpunk: { name: 'Cyberpunk', emoji: '🌆' },
    fantasy: { name: 'Fantasia', emoji: '🏰' },
    anime: { name: 'Anime', emoji: '🎭' }
};

// Configuração do comando
const data = new SlashCommandBuilder()
    .setName('gerar')
    .setDescription('Gera uma imagem pixel art usando IA')
    .addStringOption(option =>
        option
            .setName('prompt')
            .setDescription('Descrição da imagem que você quer gerar')
            .setRequired(true))
    .addStringOption(option =>
        option
            .setName('estilo')
            .setDescription('Estilo da pixel art')
            .addChoices(
                ...Object.entries(STYLES).map(([id, style]) => ({
                    name: `${style.emoji} ${style.name}`,
                    value: id
                }))
            ))
    .addIntegerOption(option =>
        option
            .setName('largura')
            .setDescription('Largura da imagem (16-128)')
            .setMinValue(16)
            .setMaxValue(128))
    .addIntegerOption(option =>
        option
            .setName('altura')
            .setDescription('Altura da imagem (16-128)')
            .setMinValue(16)
            .setMaxValue(128))
    .addStringOption(option =>
        option
            .setName('prompt_negativo')
            .setDescription('O que você não quer que apareça na imagem'));

async function checkUserPermissions(userId) {
    try {
        // Verifica se o usuário tem permissão
        const result = await pool.query(
            'SELECT is_premium, credits FROM users WHERE discord_id = $1',
            [userId]
        );

        if (result.rows.length === 0) {
            return { allowed: false, reason: 'not_registered' };
        }

        const user = result.rows[0];
        
        // Se for premium, tem permissão total
        if (user.is_premium) {
            return { allowed: true, isPremium: true };
        }

        // Verifica limite diário para usuários gratuitos
        const today = new Date().toISOString().split('T')[0];
        const usageResult = await pool.query(
            'SELECT COUNT(*) FROM generations WHERE user_id = $1 AND DATE(created_at) = $2',
            [userId, today]
        );

        const dailyUses = parseInt(usageResult.rows[0].count);
        if (dailyUses >= 50) {
            return { allowed: false, reason: 'daily_limit', uses: dailyUses };
        }

        return { allowed: true, isPremium: false, dailyUses };
    } catch (error) {
        logger.error('Erro ao verificar permissões:', error);
        throw error;
    }
}

async function createGenerationEmbed(interaction, status = 'queued') {
    const prompt = interaction.options.getString('prompt');
    const style = interaction.options.getString('estilo') || 'pixel_art';
    const width = interaction.options.getInteger('largura') || 64;
    const height = interaction.options.getInteger('altura') || 64;
    const negativePrompt = interaction.options.getString('prompt_negativo');

    const embed = new EmbedBuilder()
        .setColor(status === 'queued' ? '#FFA500' : '#00FF00')
        .setTitle(`${STYLES[style].emoji} Geração de Pixel Art`)
        .setDescription(prompt)
        .addFields(
            { name: '🎨 Estilo', value: STYLES[style].name, inline: true },
            { name: '📏 Dimensões', value: `${width}x${height}`, inline: true },
            { name: '⚙️ Status', value: status === 'queued' ? '⏳ Na fila...' : '✅ Concluído', inline: true }
        );

    if (negativePrompt) {
        embed.addFields({ name: '❌ Prompt Negativo', value: negativePrompt, inline: false });
    }

    return embed;
}

async function handleGeneration(interaction) {
    try {
        // Verifica permissões
        const permissions = await checkUserPermissions(interaction.user.id);
        
        if (!permissions.allowed) {
            const errorEmbed = new EmbedBuilder()
                .setColor('#FF0000')
                .setTitle('❌ Erro na Geração');

            switch (permissions.reason) {
                case 'not_registered':
                    errorEmbed.setDescription('Você precisa se registrar primeiro! Use `/conta registrar`');
                    break;
                case 'daily_limit':
                    errorEmbed.setDescription(`Você atingiu o limite diário de 50 gerações.\nConsidere assinar o plano premium para gerações ilimitadas!`);
                    break;
            }

            const premiumButton = new ActionRowBuilder()
                .addComponents(
                    new ButtonBuilder()
                        .setCustomId('view_premium')
                        .setLabel('💎 Ver Plano Premium')
                        .setStyle(ButtonStyle.Primary)
                );

            await interaction.reply({ embeds: [errorEmbed], components: [premiumButton], ephemeral: true });
            return;
        }

        // Cria embed inicial
        const embed = await createGenerationEmbed(interaction);
        
        // Botões de controle
        const controlRow = new ActionRowBuilder()
            .addComponents(
                new ButtonBuilder()
                    .setCustomId('cancel_generation')
                    .setLabel('❌ Cancelar')
                    .setStyle(ButtonStyle.Danger),
                new ButtonBuilder()
                    .setCustomId('regenerate')
                    .setLabel('🔄 Regenerar')
                    .setStyle(ButtonStyle.Secondary),
                new ButtonBuilder()
                    .setCustomId('share_prompt')
                    .setLabel('📤 Compartilhar Prompt')
                    .setStyle(ButtonStyle.Success)
            );

        // Envia mensagem inicial
        await interaction.reply({ embeds: [embed], components: [controlRow] });

        // Simula geração (substitua pelo código real de geração)
        await new Promise(resolve => setTimeout(resolve, 3000));

        // Atualiza com resultado
        const resultEmbed = await createGenerationEmbed(interaction, 'completed');
        
        // Adiciona botões de ação
        const actionRow = new ActionRowBuilder()
            .addComponents(
                new ButtonBuilder()
                    .setCustomId('download')
                    .setLabel('📥 Download')
                    .setStyle(ButtonStyle.Primary),
                new ButtonBuilder()
                    .setCustomId('variations')
                    .setLabel('🎲 Variações')
                    .setStyle(ButtonStyle.Secondary),
                new ButtonBuilder()
                    .setCustomId('upscale')
                    .setLabel('📐 Upscale')
                    .setStyle(ButtonStyle.Success),
                new ButtonBuilder()
                    .setCustomId('share_prompt')
                    .setLabel('📤 Compartilhar')
                    .setStyle(ButtonStyle.Secondary)
            );

        await interaction.editReply({
            embeds: [resultEmbed],
            components: [actionRow]
        });

        // Atualiza créditos se não for premium
        if (!permissions.isPremium) {
            await pool.query(
                'UPDATE users SET credits = credits - 1 WHERE discord_id = $1',
                [interaction.user.id]
            );
        }

    } catch (error) {
        logger.error('Erro na geração:', error);
        
        const errorEmbed = new EmbedBuilder()
            .setColor('#FF0000')
            .setTitle('❌ Erro na Geração')
            .setDescription('Ocorreu um erro durante a geração. Por favor, tente novamente.')
            .addFields(
                { name: 'Erro', value: error.message }
            );

        await interaction.editReply({
            embeds: [errorEmbed],
            components: []
        });
    }
}

module.exports = {
    data,
    async execute(interaction) {
        try {
            await handleGeneration(interaction);
        } catch (error) {
            logger.error('Erro ao executar comando gerar:', error);
            await interaction.reply({ 
                content: 'Ocorreu um erro inesperado. Por favor, tente novamente.',
                ephemeral: true 
            });
        }
    }
};