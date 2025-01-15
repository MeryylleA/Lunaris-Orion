const { REST, Routes } = require('discord.js');
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

async function deployCommands() {
    const commands = [];
    const commandsPath = join(__dirname, 'commands');

    try {
        // Carrega todos os comandos
        const commandFiles = readdirSync(commandsPath).filter(file => file.endsWith('.cjs'));
        
        for (const file of commandFiles) {
            const filePath = join(commandsPath, file);
            const command = require(filePath);
            
            if ('data' in command && 'execute' in command) {
                commands.push(command.data.toJSON());
                logger.info(`Comando carregado para deploy: ${command.data.name}`);
            } else {
                logger.warn(`Comando em ${filePath} está faltando 'data' ou 'execute' obrigatórios`);
            }
        }
        
        // Configura o cliente REST
        const rest = new REST().setToken(process.env.DISCORD_TOKEN);
        
        // Registra os comandos
        logger.info('Iniciando deploy dos comandos...');
        
        const data = await rest.put(
            Routes.applicationCommands(process.env.DISCORD_CLIENT_ID),
            { body: commands }
        );
        
        logger.info(`Deploy realizado com sucesso! ${data.length} comandos registrados.`);
        
    } catch (error) {
        logger.error('Erro durante o deploy dos comandos:', error);
    }
}

// Executa o deploy
deployCommands();
