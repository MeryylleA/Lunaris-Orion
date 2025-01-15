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

export const name = Events.InteractionCreate;
export const once = false;

export async function execute(interaction) {
    // Registra informações sobre a interação
    logger.info(`Nova interação de ${interaction.user.tag} em ${interaction.guild?.name || 'DM'}`);
    
    try {
        if (interaction.isChatInputCommand()) {
            // Processa comandos slash
            const command = interaction.client.commands.get(interaction.commandName);
            
            if (!command) {
                logger.warn(`Comando não encontrado: ${interaction.commandName}`);
                await interaction.reply({
                    content: 'Desculpe, não encontrei este comando.',
                    ephemeral: true
                });
                return;
            }
            
            await command.execute(interaction);
            
        } else if (interaction.isButton()) {
            // Processa interações de botão
            const [commandName, ...args] = interaction.customId.split('_');
            const command = interaction.client.commands.get(commandName);
            
            if (!command || !command.handleButton) {
                logger.warn(`Manipulador de botão não encontrado: ${commandName}`);
                await interaction.reply({
                    content: 'Desculpe, ocorreu um erro ao processar este botão.',
                    ephemeral: true
                });
                return;
            }
            
            await command.handleButton(interaction, args);
            
        } else if (interaction.isModalSubmit()) {
            // Processa envios de modal
            const [commandName, ...args] = interaction.customId.split('_');
            const command = interaction.client.commands.get(commandName);
            
            if (!command || !command.handleModal) {
                logger.warn(`Manipulador de modal não encontrado: ${commandName}`);
                await interaction.reply({
                    content: 'Desculpe, ocorreu um erro ao processar este formulário.',
                    ephemeral: true
                });
                return;
            }
            
            await command.handleModal(interaction, args);
        }
        
    } catch (error) {
        logger.error('Erro ao processar interação:', error);
        
        const errorMessage = {
            content: 'Desculpe, ocorreu um erro ao processar sua solicitação.',
            ephemeral: true
        };
        
        if (interaction.replied || interaction.deferred) {
            await interaction.followUp(errorMessage);
        } else {
            await interaction.reply(errorMessage);
        }
    }
} 