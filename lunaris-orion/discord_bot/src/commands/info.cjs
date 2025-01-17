const { SlashCommandBuilder, EmbedBuilder } = require('discord.js');
const { createLogger, format, transports } = require('winston');

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.printf(({ timestamp, level, message }) => {
            return `${timestamp} ${level}: ${message}`;
        })
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

// Configuração do comando
const data = new SlashCommandBuilder()
    .setName('info')
    .setDescription('Mostra informações sobre o bot');

module.exports = {
    data,
    async execute(interaction) {
        logger.info('Comando info iniciado');
        
        try {
            logger.debug('Criando embed de informações');
            
            const embed = new EmbedBuilder()
                .setColor('#0099ff')
                .setTitle('Lunaris Orion - Informações')
                .setDescription('Bot para geração de pixel art usando IA')
                .addFields(
                    { name: 'Status', value: '🟢 Online', inline: true },
                    { name: 'Versão', value: '1.0.0', inline: true }
                )
                .setTimestamp();

            logger.debug('Embed criado, enviando resposta');
            
            // Como já fizemos deferReply no index.cjs, usamos editReply aqui
            await interaction.editReply({ embeds: [embed] });
            
            logger.info('Comando info executado com sucesso');
        } catch (error) {
            logger.error('Erro ao executar comando info:', error);
            await interaction.editReply({ 
                content: 'Ocorreu um erro ao buscar as informações.',
                ephemeral: true 
            });
        }
    }
};