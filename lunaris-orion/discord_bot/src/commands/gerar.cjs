const { SlashCommandBuilder, EmbedBuilder } = require('discord.js');
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
        })
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

// Configuração do comando
const data = new SlashCommandBuilder()
    .setName('gerar')
    .setDescription('Gera uma imagem em pixel art usando IA')
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
                { name: 'Retro', value: 'retro' },
                { name: 'Cyberpunk', value: 'cyberpunk' },
                { name: 'Fantasy', value: 'fantasy' },
                { name: 'Minimal', value: 'minimal' },
                { name: 'Anime', value: 'anime' }
            ))
    .addStringOption(option =>
        option
            .setName('resolucao')
            .setDescription('Tamanho da imagem')
            .addChoices(
                { name: '16x16', value: '16' },
                { name: '32x32', value: '32' },
                { name: '64x64', value: '64' },
                { name: '128x128', value: '128' }
            ))
    .addStringOption(option =>
        option
            .setName('prompt_negativo')
            .setDescription('O que você NÃO quer que apareça na imagem'));

async function handleGeneration(interaction) {
    try {
        await interaction.deferReply();
        
        const userId = interaction.user.id;
        const prompt = interaction.options.getString('prompt');
        const style = interaction.options.getString('estilo') || 'retro';
        const size = interaction.options.getString('resolucao') || '64';
        const negativePrompt = interaction.options.getString('prompt_negativo') || '';
        
        // Verifica se usuário tem permissão
        const userResult = await pool.query(
            'SELECT * FROM users WHERE id = $1',
            [userId]
        );
        
        if (!userResult.rows[0]) {
            return interaction.editReply('Você precisa registrar uma conta primeiro! Use `/conta registrar`');
        }
        
        // Verifica limites de uso
        const usageResult = await pool.query(
            'SELECT COUNT(*) FROM generations WHERE user_id = $1 AND created_at > NOW() - INTERVAL \'24 hours\'',
            [userId]
        );
        
        const dailyUsage = parseInt(usageResult.rows[0].count);
        const dailyLimit = 5; // Limite para usuários gratuitos
        
        if (dailyUsage >= dailyLimit) {
            return interaction.editReply('Você atingiu seu limite diário de gerações. Considere assinar um plano premium!');
        }
        
        // Registra a geração
        const generationResult = await pool.query(
            `INSERT INTO generations (
                user_id, prompt, negative_prompt, width, height,
                steps, style, status, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            RETURNING id`,
            [userId, prompt, negativePrompt, size, size, 20, style, 'pending']
        );
        
        const generationId = generationResult.rows[0].id;
        
        // Aqui você deve implementar a chamada para sua API de geração
        // Por enquanto, vamos simular uma resposta
        
        const embed = new EmbedBuilder()
            .setColor('#0099ff')
            .setTitle('Geração Iniciada')
            .addFields(
                { name: 'Prompt', value: prompt },
                { name: 'Estilo', value: style },
                { name: 'Tamanho', value: `${size}x${size}` },
                { name: 'Status', value: 'Em processamento...' }
            )
            .setTimestamp();
        
        await interaction.editReply({ embeds: [embed] });
        
        // Aqui você implementaria a lógica de webhook para atualizar quando a imagem estiver pronta
        
    } catch (error) {
        logger.error('Erro ao gerar imagem:', error);
        throw new Error('Erro ao gerar imagem: ' + error.message);
    }
}

module.exports = {
    data,
    async execute(interaction) {
        try {
            await handleGeneration(interaction);
        } catch (error) {
            logger.error('Erro ao processar comando de geração:', error);
            const message = interaction.deferred ? 
                interaction.editReply : 
                interaction.reply;
            
            await message.call(interaction, {
                content: 'Ocorreu um erro ao processar seu comando.',
                ephemeral: true
            });
        }
    }
}; 