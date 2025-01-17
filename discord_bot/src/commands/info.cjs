const { SlashCommandBuilder, EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle } = require('discord.js');
const { createLogger, format, transports } = require('winston');
const os = require('os');

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

const data = new SlashCommandBuilder()
    .setName('info')
    .setDescription('Mostra informações sobre o Lunaris Orion')
    .addSubcommand(subcommand =>
        subcommand
            .setName('sistema')
            .setDescription('Informações sobre o sistema'))
    .addSubcommand(subcommand =>
        subcommand
            .setName('bot')
            .setDescription('Informações sobre o bot'))
    .addSubcommand(subcommand =>
        subcommand
            .setName('recursos')
            .setDescription('Lista de recursos disponíveis'));

async function formatUptime(uptime) {
    const days = Math.floor(uptime / (24 * 60 * 60 * 1000));
    const hours = Math.floor((uptime % (24 * 60 * 60 * 1000)) / (60 * 60 * 1000));
    const minutes = Math.floor((uptime % (60 * 60 * 1000)) / (60 * 1000));
    return `${days}d ${hours}h ${minutes}m`;
}

async function handleSystemInfo(interaction, client) {
    const systemUptime = os.uptime() * 1000;
    const botUptime = client.uptime;
    const memoryUsage = process.memoryUsage();

    const embed = new EmbedBuilder()
        .setColor('#0099ff')
        .setTitle('🖥️ Informações do Sistema')
        .setDescription('Status e métricas do sistema Lunaris Orion')
        .addFields(
            { name: '⚡ Status', value: '```\n🟢 Operacional\n```', inline: true },
            { name: '⏰ Uptime Bot', value: `\`\`\`\n${await formatUptime(botUptime)}\n\`\`\``, inline: true },
            { name: '⌛ Uptime Sistema', value: `\`\`\`\n${await formatUptime(systemUptime)}\n\`\`\``, inline: true },
            { name: '💾 Uso de Memória', value: `\`\`\`\nHeap: ${Math.round(memoryUsage.heapUsed / 1024 / 1024)}MB / ${Math.round(memoryUsage.heapTotal / 1024 / 1024)}MB\nRSS: ${Math.round(memoryUsage.rss / 1024 / 1024)}MB\n\`\`\``, inline: false },
            { name: '🔄 Latência', value: `\`\`\`\nAPI: ${Math.round(client.ws.ping)}ms\n\`\`\``, inline: true }
        )
        .setFooter({ text: 'Lunaris Orion • Sistema de Monitoramento', iconURL: client.user.displayAvatarURL() })
        .setTimestamp();

    const row = new ActionRowBuilder()
        .addComponents(
            new ButtonBuilder()
                .setCustomId('refresh_system')
                .setLabel('🔄 Atualizar')
                .setStyle(ButtonStyle.Secondary),
            new ButtonBuilder()
                .setCustomId('view_details')
                .setLabel('📊 Detalhes')
                .setStyle(ButtonStyle.Primary)
        );

    await interaction.reply({ embeds: [embed], components: [row] });
}

async function handleBotInfo(interaction, client) {
    const embed = new EmbedBuilder()
        .setColor('#FFD700')
        .setTitle('🤖 Lunaris Orion')
        .setDescription('Sistema avançado de geração de pixel art usando IA')
        .setThumbnail(client.user.displayAvatarURL())
        .addFields(
            { 
                name: '📊 Estatísticas', 
                value: '```\n' +
                      '• Usuários: 1,234\n' +
                      '• Servidores: 56\n' +
                      '• Gerações: 45,678\n' +
                      '```',
                inline: false 
            },
            { 
                name: '🛠️ Versão', 
                value: '```\n' +
                      'Bot: v1.2.0\n' +
                      'Discord.js: v14.11.0\n' +
                      'Node: ' + process.version + '\n' +
                      '```',
                inline: true 
            },
            { 
                name: '👥 Equipe', 
                value: '```\n' +
                      '• Desenvolvedor: Seu Nome\n' +
                      '• Suporte: Equipe Lunaris\n' +
                      '```',
                inline: true 
            }
        )
        .setFooter({ text: 'Lunaris Orion • Powered by Stable Diffusion', iconURL: client.user.displayAvatarURL() });

    const row = new ActionRowBuilder()
        .addComponents(
            new ButtonBuilder()
                .setLabel('🌐 Website')
                .setURL('https://lunaris-orion.com')
                .setStyle(ButtonStyle.Link),
            new ButtonBuilder()
                .setLabel('📚 Documentação')
                .setURL('https://docs.lunaris-orion.com')
                .setStyle(ButtonStyle.Link),
            new ButtonBuilder()
                .setLabel('💬 Suporte')
                .setURL('https://discord.gg/lunaris')
                .setStyle(ButtonStyle.Link)
        );

    await interaction.reply({ embeds: [embed], components: [row] });
}

async function handleResourcesInfo(interaction) {
    const embed = new EmbedBuilder()
        .setColor('#00FF00')
        .setTitle('🎨 Recursos do Lunaris Orion')
        .setDescription('Explore todos os recursos disponíveis')
        .addFields(
            {
                name: '🤖 Comandos Principais',
                value: '```\n' +
                      '/gerar - Gera pixel art\n' +
                      '/prompt - Gerencia prompts\n' +
                      '/conta - Gerencia sua conta\n' +
                      '/key - Gerencia chaves API\n' +
                      '```'
            },
            {
                name: '💎 Recursos Premium',
                value: '```\n' +
                      '• Gerações ilimitadas\n' +
                      '• Resolução até 128x128\n' +
                      '• Prioridade na fila\n' +
                      '• Estilos exclusivos\n' +
                      '```'
            },
            {
                name: '🛠️ Ferramentas',
                value: '```\n' +
                      '• Plugin Aseprite\n' +
                      '• API REST\n' +
                      '• Biblioteca SDK\n' +
                      '• Compartilhamento de prompts\n' +
                      '```'
            }
        )
        .setFooter({ text: 'Use /premium para conhecer nossos planos' });

    const row = new ActionRowBuilder()
        .addComponents(
            new ButtonBuilder()
                .setCustomId('view_commands')
                .setLabel('📚 Ver Comandos')
                .setStyle(ButtonStyle.Primary),
            new ButtonBuilder()
                .setCustomId('view_premium')
                .setLabel('💎 Ver Premium')
                .setStyle(ButtonStyle.Success),
            new ButtonBuilder()
                .setCustomId('view_tools')
                .setLabel('🛠️ Ver Ferramentas')
                .setStyle(ButtonStyle.Secondary)
        );

    await interaction.reply({ embeds: [embed], components: [row] });
}

module.exports = {
    data,
    async execute(interaction) {
        logger.info(`Comando info iniciado por ${interaction.user.tag}`);
        
        try {
            const subcommand = interaction.options.getSubcommand();
            
            switch (subcommand) {
                case 'sistema':
                    await handleSystemInfo(interaction, interaction.client);
                    break;
                case 'bot':
                    await handleBotInfo(interaction, interaction.client);
                    break;
                case 'recursos':
                    await handleResourcesInfo(interaction);
                    break;
            }
            
            logger.info(`Comando info (${subcommand}) executado com sucesso`);
        } catch (error) {
            logger.error('Erro ao executar comando info:', error);
            await interaction.reply({ 
                content: 'Ocorreu um erro ao buscar as informações. Por favor, tente novamente.',
                ephemeral: true 
            });
        }
    }
};