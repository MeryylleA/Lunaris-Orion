import { Events } from 'discord.js';
import { createLogger, format, transports } from 'winston';

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

export const name = Events.ClientReady;
export const once = true;

export async function execute(client) {
    logger.info(`Bot iniciado como ${client.user.tag}`);
    
    // Configura status do bot
    client.user.setPresence({
        activities: [{
            name: '/ajuda | Gerando Pixel Art',
            type: 3 // Watching
        }],
        status: 'online'
    });
    
    // Registra informações sobre os servidores
    logger.info(`Presente em ${client.guilds.cache.size} servidores`);
    
    // Verifica configuração das GPUs
    try {
        const gpuInfo = await checkGPUs();
        logger.info('Informações das GPUs:', gpuInfo);
    } catch (error) {
        logger.error('Erro ao verificar GPUs:', error);
    }
}

async function checkGPUs() {
    // Esta função será implementada para integrar com o sistema de GPUs
    return {
        available: true,
        count: 6,
        models: ['H100', 'A100', 'L4', 'L40S', 'V100', 'V100S']
    };
} 