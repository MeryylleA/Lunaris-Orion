const { Client, GatewayIntentBits, Collection } = require('discord.js');
const { config } = require('dotenv');
const { join } = require('path');
const { readdirSync } = require('fs');
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

// Carrega variáveis de ambiente
config();

// Cria o cliente Discord
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages
    ]
});

// Coleção para armazenar comandos
client.commands = new Collection();

// Carrega comandos
const commandsPath = join(__dirname, 'commands');
const commandFiles = readdirSync(commandsPath).filter(file => file.endsWith('.cjs'));

for (const file of commandFiles) {
    const filePath = join(commandsPath, file);
    const command = require(filePath);
    
    if ('data' in command && 'execute' in command) {
        client.commands.set(command.data.name, command);
        logger.info(`Comando carregado: ${command.data.name}`);
    } else {
        logger.warn(`Comando em ${filePath} está faltando 'data' ou 'execute' obrigatórios`);
    }
}

// Evento ready
client.once('ready', () => {
    logger.info(`Bot online como ${client.user.tag}!`);
});

// Manipulador de interações
client.on('interactionCreate', async interaction => {
    if (!interaction.isChatInputCommand()) return;

    const command = client.commands.get(interaction.commandName);

    if (!command) {
        logger.error(`Comando ${interaction.commandName} não encontrado`);
        return;
    }

    try {
        await command.execute(interaction);
    } catch (error) {
        logger.error(`Erro ao executar comando ${interaction.commandName}:`, error);
        
        const errorMessage = {
            content: 'Ocorreu um erro ao executar este comando!',
            ephemeral: true
        };
        
        if (interaction.replied || interaction.deferred) {
            await interaction.followUp(errorMessage);
        } else {
            await interaction.reply(errorMessage);
        }
    }
});

// Login do bot
client.login(process.env.DISCORD_TOKEN)
    .catch(error => {
        logger.error('Erro ao fazer login:', error);
    }); 