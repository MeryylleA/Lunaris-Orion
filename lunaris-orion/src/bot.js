const { Client, GatewayIntentBits, Collection } = require('discord.js');
const { REST } = require('@discordjs/rest');
const { Routes } = require('discord-api-types/v10');
const { config } = require('dotenv');
const { createLogger, format, transports } = require('winston');
const Stripe = require('stripe');
const crypto = require('crypto');
const path = require('path');
const fs = require('fs');

// Configuração do ambiente
config();

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.File({ filename: 'logs/error.log', level: 'error' }),
        new transports.File({ filename: 'logs/combined.log' }),
        new transports.Console({
            format: format.combine(
                format.colorize(),
                format.simple()
            )
        })
    ]
});

// Inicialização do Stripe
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);

// Inicialização do cliente Discord
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

// Carrega comandos
const commandsPath = path.join(__dirname, 'commands');
const commandFiles = fs.readdirSync(commandsPath).filter(file => file.endsWith('.js'));

for (const file of commandFiles) {
    const filePath = path.join(commandsPath, file);
    const command = require(filePath);
    client.commands.set(command.data.name, command);
}

// Evento de inicialização
client.once('ready', () => {
    logger.info(`Bot iniciado como ${client.user.tag}`);
    
    // Registra comandos slash
    const commands = [];
    client.commands.forEach(command => {
        commands.push(command.data.toJSON());
    });

    const rest = new REST({ version: '10' }).setToken(process.env.DISCORD_TOKEN);

    rest.put(
        Routes.applicationCommands(client.user.id),
        { body: commands }
    )
    .then(() => logger.info('Comandos slash registrados com sucesso'))
    .catch(error => logger.error('Erro ao registrar comandos:', error));
});

// Manipulador de interações
client.on('interactionCreate', async interaction => {
    if (!interaction.isCommand()) return;

    const command = client.commands.get(interaction.commandName);
    if (!command) return;

    try {
        await command.execute(interaction, client);
    } catch (error) {
        logger.error(`Erro ao executar comando ${interaction.commandName}:`, error);
        
        const errorMessage = {
            content: 'Ocorreu um erro ao executar este comando.',
            ephemeral: true
        };

        if (interaction.replied || interaction.deferred) {
            await interaction.followUp(errorMessage);
        } else {
            await interaction.reply(errorMessage);
        }
    }
});

// Sistema de cooldown global
client.on('interactionCreate', async interaction => {
    if (!interaction.isCommand()) return;

    const { cooldowns } = client;
    const command = client.commands.get(interaction.commandName);

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
});

// Tratamento de erros
process.on('unhandledRejection', error => {
    logger.error('Erro não tratado:', error);
});

// Inicia o bot
client.login(process.env.DISCORD_TOKEN)
    .then(() => logger.info('Bot conectado ao Discord'))
    .catch(error => logger.error('Erro ao conectar:', error)); 