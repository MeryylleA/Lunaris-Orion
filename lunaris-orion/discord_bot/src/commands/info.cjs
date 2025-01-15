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

const data = new SlashCommandBuilder()
    .setName('info')
    .setDescription('Mostra informações sobre o status do sistema');

async function getSystemStats() {
    try {
        // Total de usuários
        const usersResult = await pool.query('SELECT COUNT(*) FROM users');
        const totalUsers = parseInt(usersResult.rows[0].count);
        
        // Total de gerações nas últimas 24h
        const generationsResult = await pool.query(
            'SELECT COUNT(*) FROM generations WHERE created_at > NOW() - INTERVAL \'24 hours\''
        );
        const dailyGenerations = parseInt(generationsResult.rows[0].count);
        
        // Usuários premium
        const premiumResult = await pool.query(
            'SELECT COUNT(DISTINCT user_id) FROM subscriptions WHERE status = \'active\''
        );
        const premiumUsers = parseInt(premiumResult.rows[0].count);
        
        // Tempo médio de geração
        const avgTimeResult = await pool.query(`
            SELECT AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_time
            FROM generations 
            WHERE status = 'completed'
            AND created_at > NOW() - INTERVAL '24 hours'
        `);
        const avgGenerationTime = Math.round(parseFloat(avgTimeResult.rows[0].avg_time || 0));
        
        return {
            totalUsers,
            dailyGenerations,
            premiumUsers,
            avgGenerationTime
        };
    } catch (error) {
        logger.error('Erro ao obter estatísticas:', error);
        throw error;
    }
}

module.exports = {
    data,
    async execute(interaction) {
        try {
            await interaction.deferReply();
            
            const stats = await getSystemStats();
            
            const embed = new EmbedBuilder()
                .setColor('#0099ff')
                .setTitle('Status do Sistema')
                .addFields(
                    { name: 'Usuários Registrados', value: stats.totalUsers.toString() },
                    { name: 'Usuários Premium', value: stats.premiumUsers.toString() },
                    { name: 'Gerações (24h)', value: stats.dailyGenerations.toString() },
                    { name: 'Tempo Médio', value: `${stats.avgGenerationTime}s` }
                )
                .setTimestamp();
            
            await interaction.editReply({ embeds: [embed] });
            
        } catch (error) {
            logger.error('Erro ao processar comando de info:', error);
            await interaction.editReply({
                content: 'Ocorreu um erro ao obter informações do sistema.',
                ephemeral: true
            });
        }
    }
}; 