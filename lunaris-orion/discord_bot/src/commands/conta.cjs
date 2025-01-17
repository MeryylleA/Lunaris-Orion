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
    .setName('conta')
    .setDescription('Gerencia sua conta Lunaris')
    .addSubcommand(subcommand =>
        subcommand
            .setName('info')
            .setDescription('Mostra informações da sua conta'))
    .addSubcommand(subcommand =>
        subcommand
            .setName('registrar')
            .setDescription('Registra uma nova conta'))
    .addSubcommand(subcommand =>
        subcommand
            .setName('chave')
            .setDescription('Gerencia suas chaves de API')
            .addStringOption(option =>
                option
                    .setName('ação')
                    .setDescription('O que você quer fazer com a chave')
                    .setRequired(true)
                    .addChoices(
                        { name: 'Gerar Nova', value: 'gerar' },
                        { name: 'Revogar', value: 'revogar' },
                        { name: 'Listar', value: 'listar' }
                    )));

async function handleAccountInfo(interaction) {
    try {
        const userId = interaction.user.id;
        
        // Busca informações do usuário
        const userResult = await pool.query(
            'SELECT * FROM users WHERE id = $1',
            [userId]
        );
        
        if (!userResult.rows[0]) {
            return interaction.reply({
                content: 'Você ainda não tem uma conta registrada. Use `/conta registrar` para criar uma.',
                ephemeral: true
            });
        }
        
        const user = userResult.rows[0];
        
        // Busca informações da assinatura
        const subscriptionResult = await pool.query(
            'SELECT * FROM subscriptions WHERE user_id = $1',
            [userId]
        );
        
        // Busca chaves API ativas
        const apiKeysResult = await pool.query(
            'SELECT COUNT(*) FROM api_keys WHERE user_id = $1 AND active = true',
            [userId]
        );
        
        const embed = new EmbedBuilder()
            .setColor('#0099ff')
            .setTitle('Informações da Conta')
            .addFields(
                { name: 'Username', value: user.username },
                { name: 'Email', value: user.email || 'Não configurado' },
                { name: 'Criada em', value: new Date(user.created_at).toLocaleDateString() },
                { name: 'Plano', value: subscriptionResult.rows[0]?.status || 'Gratuito' },
                { name: 'Chaves API Ativas', value: apiKeysResult.rows[0].count.toString() }
            )
            .setTimestamp();
        
        return interaction.reply({ embeds: [embed], ephemeral: true });
        
    } catch (error) {
        logger.error('Erro ao obter informações da conta:', error);
        throw new Error('Erro ao obter informações da conta: ' + error.message);
    }
}

async function handleRegister(interaction) {
    try {
        const userId = interaction.user.id;
        const username = interaction.user.username;
        
        // Verifica se já existe conta
        const existingUser = await pool.query(
            'SELECT id FROM users WHERE id = $1',
            [userId]
        );
        
        if (existingUser.rows[0]) {
            return interaction.reply({
                content: 'Você já tem uma conta registrada!',
                ephemeral: true
            });
        }
        
        // Cria nova conta
        await pool.query(
            'INSERT INTO users (id, username, created_at) VALUES ($1, $2, NOW())',
            [userId, username]
        );
        
        return interaction.reply({
            content: 'Conta criada com sucesso! Use `/conta info` para ver suas informações.',
            ephemeral: true
        });
        
    } catch (error) {
        logger.error('Erro ao registrar conta:', error);
        throw new Error('Erro ao registrar conta: ' + error.message);
    }
}

async function handleApiKey(interaction) {
    const action = interaction.options.getString('ação');
    
    try {
        switch (action) {
            case 'gerar':
                // Implementar geração de chave
                break;
            case 'revogar':
                // Implementar revogação de chave
                break;
            case 'listar':
                // Implementar listagem de chaves
                break;
        }
    } catch (error) {
        logger.error('Erro ao gerenciar chave API:', error);
        throw new Error('Erro ao gerenciar chave API: ' + error.message);
    }
}

module.exports = {
    data,
    async execute(interaction) {
        try {
            const subcommand = interaction.options.getSubcommand();
            
            switch (subcommand) {
                case 'info':
                    await handleAccountInfo(interaction);
                    break;
                case 'registrar':
                    await handleRegister(interaction);
                    break;
                case 'chave':
                    await handleApiKey(interaction);
                    break;
            }
        } catch (error) {
            logger.error('Erro ao processar comando de conta:', error);
            await interaction.reply({
                content: 'Ocorreu um erro ao processar seu comando.',
                ephemeral: true
            });
        }
    }
}; 