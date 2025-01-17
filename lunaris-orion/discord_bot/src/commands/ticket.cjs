const { SlashCommandBuilder, EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle, ModalBuilder, TextInputBuilder, TextInputStyle } = require('discord.js');
const { createLogger, format, transports } = require('winston');

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.Console(),
        new transports.File({ filename: 'logs/commands.log' })
    ]
});

module.exports = {
    data: new SlashCommandBuilder()
        .setName('ticket')
        .setDescription('Sistema de suporte técnico')
        .addSubcommand(subcommand =>
            subcommand
                .setName('criar')
                .setDescription('Cria um novo ticket de suporte'))
        .addSubcommand(subcommand =>
            subcommand
                .setName('listar')
                .setDescription('Lista seus tickets ativos'))
        .addSubcommand(subcommand =>
            subcommand
                .setName('fechar')
                .setDescription('Fecha um ticket existente')
                .addStringOption(option =>
                    option
                        .setName('ticket_id')
                        .setDescription('ID do ticket para fechar')
                        .setRequired(true))),

    async execute(interaction, client) {
        try {
            const subcommand = interaction.options.getSubcommand();

            switch (subcommand) {
                case 'criar':
                    await handleCreateTicket(interaction, client);
                    break;
                case 'listar':
                    await handleListTickets(interaction, client);
                    break;
                case 'fechar':
                    await handleCloseTicket(interaction, client);
                    break;
            }
        } catch (error) {
            logger.error('Erro ao executar comando de ticket:', error);
            await interaction.reply({
                content: 'Erro ao processar comando. Tente novamente mais tarde.',
                ephemeral: true
            });
        }
    }
};

async function handleCreateTicket(interaction, client) {
    try {
        // Verifica se já tem muitos tickets abertos
        const activeTickets = await client.pool.query(
            `SELECT COUNT(*) as count
            FROM tickets
            WHERE user_id = (SELECT id FROM users WHERE discord_id = $1)
            AND status = 'open'`,
            [interaction.user.id]
        );

        if (activeTickets.rows[0].count >= 3) {
            await interaction.reply({
                content: 'Você já possui 3 tickets abertos. Por favor, feche algum antes de criar um novo.',
                ephemeral: true
            });
            return;
        }

        // Cria modal para detalhes do ticket
        const modal = new ModalBuilder()
            .setCustomId('ticket_create_modal')
            .setTitle('Criar Ticket de Suporte');

        const titleInput = new TextInputBuilder()
            .setCustomId('ticket_title')
            .setLabel('Título do Problema')
            .setStyle(TextInputStyle.Short)
            .setPlaceholder('Ex: Erro na geração de pixel art')
            .setRequired(true);

        const descriptionInput = new TextInputBuilder()
            .setCustomId('ticket_description')
            .setLabel('Descrição Detalhada')
            .setStyle(TextInputStyle.Paragraph)
            .setPlaceholder('Descreva seu problema em detalhes...')
            .setRequired(true);

        const firstRow = new ActionRowBuilder().addComponents(titleInput);
        const secondRow = new ActionRowBuilder().addComponents(descriptionInput);

        modal.addComponents(firstRow, secondRow);
        await interaction.showModal(modal);

    } catch (error) {
        logger.error('Erro ao criar ticket:', error);
        throw error;
    }
}

async function handleListTickets(interaction, client) {
    try {
        // Busca tickets do usuário
        const tickets = await client.pool.query(
            `SELECT t.*, u.discord_id
            FROM tickets t
            JOIN users u ON t.user_id = u.id
            WHERE u.discord_id = $1
            ORDER BY t.created_at DESC`,
            [interaction.user.id]
        );

        if (tickets.rows.length === 0) {
            await interaction.reply({
                content: 'Você não possui nenhum ticket.',
                ephemeral: true
            });
            return;
        }

        // Cria embed com lista de tickets
        const embed = new EmbedBuilder()
            .setTitle('🎫 Seus Tickets')
            .setColor('#0099ff');

        tickets.rows.forEach(ticket => {
            embed.addFields({
                name: `#${ticket.id} - ${ticket.title}`,
                value: `Status: ${ticket.status === 'open' ? '🟢 Aberto' : '🔴 Fechado'}\n` +
                       `Criado em: ${ticket.created_at.toLocaleDateString('pt-BR')}\n` +
                       `Última atualização: ${ticket.updated_at.toLocaleDateString('pt-BR')}`,
                inline: false
            });
        });

        await interaction.reply({ embeds: [embed], ephemeral: true });

    } catch (error) {
        logger.error('Erro ao listar tickets:', error);
        throw error;
    }
}

async function handleCloseTicket(interaction, client) {
    try {
        const ticketId = interaction.options.getString('ticket_id');

        // Verifica se o ticket existe e pertence ao usuário
        const ticket = await client.pool.query(
            `SELECT t.*, u.discord_id
            FROM tickets t
            JOIN users u ON t.user_id = u.id
            WHERE t.id = $1 AND u.discord_id = $2`,
            [ticketId, interaction.user.id]
        );

        if (!ticket.rows[0]) {
            await interaction.reply({
                content: 'Ticket não encontrado ou você não tem permissão para fechá-lo.',
                ephemeral: true
            });
            return;
        }

        if (ticket.rows[0].status === 'closed') {
            await interaction.reply({
                content: 'Este ticket já está fechado.',
                ephemeral: true
            });
            return;
        }

        // Fecha o ticket
        await client.pool.query(
            `UPDATE tickets
            SET status = 'closed', updated_at = NOW()
            WHERE id = $1`,
            [ticketId]
        );

        // Notifica o usuário
        const embed = new EmbedBuilder()
            .setTitle('🎫 Ticket Fechado')
            .setColor('#00ff00')
            .setDescription(`O ticket #${ticketId} foi fechado com sucesso.`)
            .addFields({
                name: 'Feedback',
                value: 'Como foi sua experiência com o suporte?',
                inline: false
            });

        const row = new ActionRowBuilder()
            .addComponents(
                new ButtonBuilder()
                    .setCustomId(`feedback_positive_${ticketId}`)
                    .setLabel('👍 Bom')
                    .setStyle(ButtonStyle.Success),
                new ButtonBuilder()
                    .setCustomId(`feedback_negative_${ticketId}`)
                    .setLabel('👎 Ruim')
                    .setStyle(ButtonStyle.Danger)
            );

        await interaction.reply({
            embeds: [embed],
            components: [row],
            ephemeral: true
        });

    } catch (error) {
        logger.error('Erro ao fechar ticket:', error);
        throw error;
    }
} 