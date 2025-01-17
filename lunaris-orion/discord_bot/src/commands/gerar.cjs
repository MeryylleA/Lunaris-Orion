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
    .setDescription('Gera uma imagem pixel art usando IA')
    .addStringOption(option =>
        option.setName('prompt')
            .setDescription('Descrição da imagem que você quer gerar')
            .setRequired(true));

async function handleGeneration(interaction) {
    try {
        // Envia mensagem inicial
        await interaction.editReply(' Iniciando geração da imagem...');

        // Verifica se o usuário tem permissão
        const userId = interaction.user.id;
        const result = await pool.query(
            'SELECT is_premium, credits FROM users WHERE discord_id = $1',
            [userId]
        );

        if (result.rows.length === 0) {
            return interaction.editReply(' Você precisa se registrar primeiro! Use /conta para criar sua conta.');
        }

        const user = result.rows[0];
        if (!user.is_premium && user.credits <= 0) {
            return interaction.editReply(' Você não tem créditos suficientes! Adquira o plano premium ou mais créditos.');
        }

        // Obtém o prompt
        const prompt = interaction.options.getString('prompt');

        // Aqui você chamaria sua API de geração
        // Por enquanto, vamos simular uma geração
        await new Promise(resolve => setTimeout(resolve, 2000)); // Simula o tempo de geração

        // Cria embed com resultado
        const embed = new EmbedBuilder()
            .setColor('#00ff00')
            .setTitle('Imagem Gerada!')
            .setDescription(`Prompt: ${prompt}`)
            .setTimestamp();

        // Atualiza a resposta com o embed
        await interaction.editReply({
            content: ' Imagem gerada com sucesso!',
            embeds: [embed]
        });

        // Atualiza créditos do usuário se não for premium
        if (!user.is_premium) {
            await pool.query(
                'UPDATE users SET credits = credits - 1 WHERE discord_id = $1',
                [userId]
            );
        }

    } catch (error) {
        logger.error('Erro na geração:', error);
        await interaction.editReply(' Ocorreu um erro durante a geração. Por favor, tente novamente.');
    }
}

module.exports = {
    data,
    async execute(interaction) {
        try {
            await handleGeneration(interaction);
        } catch (error) {
            logger.error('Erro ao executar comando gerar:', error);
            await interaction.editReply(' Ocorreu um erro inesperado. Por favor, tente novamente.');
        }
    }
};