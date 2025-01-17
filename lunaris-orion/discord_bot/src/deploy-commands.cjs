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
        logger.info(`Carregando comandos do diretório: ${commandsPath}`);
        
        // Verifica se o diretório existe
        if (!readdirSync(commandsPath)) {
            throw new Error(`Diretório de comandos não encontrado: ${commandsPath}`);
        }

        // Lista todos os arquivos de comando
        const commandFiles = readdirSync(commandsPath).filter(file => file.endsWith('.cjs'));
        logger.info(`Encontrados ${commandFiles.length} arquivos de comando: ${commandFiles.join(', ')}`);

        // Carrega cada comando
        for (const file of commandFiles) {
            const filePath = join(commandsPath, file);
            logger.info(`Carregando comando: ${file}`);
            
            try {
                const command = require(filePath);
                if ('data' in command && 'execute' in command) {
                    commands.push(command.data.toJSON());
                    logger.info(`Comando carregado com sucesso: ${command.data.name}`);
                } else {
                    logger.warn(`Comando inválido em ${file}: Faltando propriedades obrigatórias`);
                }
            } catch (error) {
                logger.error(`Erro ao carregar comando ${file}:`, error);
            }
        }

        // Verifica se temos o token e client ID
        if (!process.env.DISCORD_TOKEN) {
            throw new Error('DISCORD_TOKEN não encontrado no arquivo .env');
        }
        if (!process.env.DISCORD_CLIENT_ID) {
            throw new Error('DISCORD_CLIENT_ID não encontrado no arquivo .env');
        }

        logger.info(`Iniciando registro de ${commands.length} comandos...`);

        // Configura o cliente REST
        const rest = new REST().setToken(process.env.DISCORD_TOKEN);

        // Registra os comandos
        logger.info('Iniciando deploy dos comandos...');
        const data = await rest.put(
            Routes.applicationCommands(process.env.DISCORD_CLIENT_ID),
            { body: commands },
        );

        logger.info(`Comandos registrados com sucesso! ${data.length} comandos deployados.`);
        
        // Lista os comandos registrados
        logger.info('Comandos registrados:');
        data.forEach(cmd => logger.info(`- ${cmd.name}`));

    } catch (error) {
        logger.error('Erro durante o deploy dos comandos:', error);
        process.exit(1);
    }
}

// Executa o deploy
deployCommands();
