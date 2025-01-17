const { Client, GatewayIntentBits, Collection } = require('discord.js');
const { config } = require('dotenv');
const { join } = require('path');
const { readdirSync } = require('fs');
const { createLogger, format, transports } = require('winston');
const Stripe = require('stripe');
require('./server.cjs'); // Start the webhook server

// Carrega variáveis de ambiente
config();

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.printf(({ timestamp, level, message, ...rest }) => {
            const restString = Object.keys(rest).length ? `\n${JSON.stringify(rest, null, 2)}` : '';
            return `${timestamp} ${level}: ${message}${restString}`;
        })
    ),
    transports: [
        new transports.Console({
            format: format.combine(
                format.colorize(),
                format.simple()
            )
        }),
        new transports.File({ 
            filename: 'logs/error.log', 
            level: 'error' 
        }),
        new transports.File({ 
            filename: 'logs/debug.log',
            level: 'debug'
        })
    ]
});

// Inicialização do Stripe
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);

// Cria o cliente Discord
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent
    ]
});

// Coleções para comandos e dados
client.commands = new Collection();
client.cooldowns = new Collection();
client.subscriptions = new Collection();
client.stripe = stripe;

// Carrega comandos
const commandsPath = join(__dirname, 'commands');
const commandFiles = readdirSync(commandsPath).filter(file => file.endsWith('.cjs'));

for (const file of commandFiles) {
    const filePath = join(commandsPath, file);
    try {
        const command = require(filePath);
        client.commands.set(command.data.name, command);
        logger.info(`Comando carregado: ${command.data.name}`);
    } catch (error) {
        logger.error(`Erro ao carregar comando ${file}:`, error);
    }
}

// Evento ready
client.once('ready', () => {
    logger.info(`Bot online! Logado como ${client.user.tag}`);
    logger.info(`Comandos carregados: ${[...client.commands.keys()].join(', ')}`);
});

// Manipulador de interações
client.on('interactionCreate', async interaction => {
    if (!interaction.isChatInputCommand()) return;

    const command = client.commands.get(interaction.commandName);
    if (!command) return;

    // Sistema de cooldown
    const { cooldowns } = client;
    if (!cooldowns.has(command.data.name)) {
        cooldowns.set(command.data.name, new Collection());
    }

    const now = Date.now();
    const timestamps = cooldowns.get(command.data.name);
    const cooldownAmount = (command.cooldown || 3) * 1000;

    if (timestamps.has(interaction.user.id)) {
        const expirationTime = timestamps.get(interaction.user.id) + cooldownAmount;

        if (now < expirationTime) {
            const timeLeft = (expirationTime - now) / 1000;
            return interaction.reply({ 
                content: `Por favor, aguarde ${timeLeft.toFixed(1)} segundos antes de usar o comando \`${command.data.name}\` novamente.`,
                ephemeral: true 
            });
        }
    }

    timestamps.set(interaction.user.id, now);
    setTimeout(() => timestamps.delete(interaction.user.id), cooldownAmount);

    try {
        logger.info(`Executando comando: ${interaction.commandName}`);
        const startTime = Date.now();
        
        // Executa o comando
        await command.execute(interaction, client);
        
        const executionTime = Date.now() - startTime;
        logger.info(`Comando ${interaction.commandName} executado com sucesso em ${executionTime}ms`);
    } catch (error) {
        logger.error('Erro ao executar comando:', {
            command: interaction.commandName,
            error: error.message,
            stack: error.stack
        });

        const errorMessage = {
            content: 'Ocorreu um erro ao executar este comando.',
            ephemeral: true
        };

        try {
            if (interaction.deferred) {
                await interaction.editReply(errorMessage);
            } else if (!interaction.replied) {
                await interaction.reply(errorMessage);
            }
        } catch (e) {
            logger.error('Erro ao enviar mensagem de erro:', e);
        }
    }
});

// Manipulador de erros não tratados
process.on('unhandledRejection', (error) => {
    logger.error('Erro não tratado:', {
        error: error.message,
        stack: error.stack
    });
});

// Login to Discord
client.login(process.env.DISCORD_TOKEN);