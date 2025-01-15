const { SlashCommandBuilder, EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle } = require('discord.js');
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
        .setName('prompt')
        .setDescription('Sistema de compartilhamento de prompts')
        .addSubcommand(subcommand =>
            subcommand
                .setName('compartilhar')
                .setDescription('Compartilha um prompt bem sucedido')
                .addStringOption(option =>
                    option
                        .setName('generation_id')
                        .setDescription('ID da geração para compartilhar')
                        .setRequired(true)))
        .addSubcommand(subcommand =>
            subcommand
                .setName('explorar')
                .setDescription('Explora prompts compartilhados pela comunidade')
                .addStringOption(option =>
                    option
                        .setName('categoria')
                        .setDescription('Categoria de prompts para explorar')
                        .addChoices(
                            { name: 'Personagens', value: 'character' },
                            { name: 'Itens', value: 'item' },
                            { name: 'Cenários', value: 'scenario' },
                            { name: 'Todos', value: 'all' }
                        )))
        .addSubcommand(subcommand =>
            subcommand
                .setName('favoritos')
                .setDescription('Lista seus prompts favoritos')),

    async execute(interaction, client) {
        try {
            const subcommand = interaction.options.getSubcommand();

            switch (subcommand) {
                case 'compartilhar':
                    await handleShare(interaction, client);
                    break;
                case 'explorar':
                    await handleExplore(interaction, client);
                    break;
                case 'favoritos':
                    await handleFavorites(interaction, client);
                    break;
            }
        } catch (error) {
            logger.error('Erro ao executar comando de prompt:', error);
            await interaction.reply({
                content: 'Erro ao processar comando. Tente novamente mais tarde.',
                ephemeral: true
            });
        }
    }
};

async function handleShare(interaction, client) {
    try {
        const generationId = interaction.options.getString('generation_id');

        // Busca dados da geração
        const generation = await client.pool.query(
            `SELECT g.*, p.prompt, p.negative_prompt, u.discord_id
            FROM generations g
            JOIN prompts p ON g.prompt_id = p.id
            JOIN users u ON g.user_id = u.id
            WHERE g.id = $1 AND u.discord_id = $2`,
            [generationId, interaction.user.id]
        );

        if (!generation.rows[0]) {
            await interaction.reply({
                content: 'Geração não encontrada ou você não tem permissão para compartilhá-la.',
                ephemeral: true
            });
            return;
        }

        // Compartilha o prompt
        const gen = generation.rows[0];
        await client.pool.query(
            `INSERT INTO shared_prompts (
                generation_id, user_id, prompt, negative_prompt,
                width, height, created_at
            ) VALUES (
                $1, (SELECT id FROM users WHERE discord_id = $2),
                $3, $4, $5, $6, NOW()
            )`,
            [
                gen.id,
                interaction.user.id,
                gen.prompt,
                gen.negative_prompt,
                gen.width,
                gen.height
            ]
        );

        // Notifica o usuário
        const embed = new EmbedBuilder()
            .setTitle('🎨 Prompt Compartilhado')
            .setColor('#00ff00')
            .addFields(
                {
                    name: 'Prompt',
                    value: gen.prompt,
                    inline: false
                },
                {
                    name: 'Prompt Negativo',
                    value: gen.negative_prompt || 'Nenhum',
                    inline: false
                },
                {
                    name: 'Resolução',
                    value: `${gen.width}x${gen.height}`,
                    inline: true
                }
            );

        await interaction.reply({
            content: 'Prompt compartilhado com sucesso!',
            embeds: [embed],
            ephemeral: true
        });

    } catch (error) {
        logger.error('Erro ao compartilhar prompt:', error);
        throw error;
    }
}

async function handleExplore(interaction, client) {
    try {
        const categoria = interaction.options.getString('categoria') || 'all';

        // Busca prompts compartilhados
        let query = `
            SELECT sp.*, u.discord_id,
            (SELECT COUNT(*) FROM prompt_likes WHERE prompt_id = sp.id) as likes
            FROM shared_prompts sp
            JOIN users u ON sp.user_id = u.id
        `;

        if (categoria !== 'all') {
            query += ` WHERE sp.category = $1`;
        }

        query += ` ORDER BY likes DESC, sp.created_at DESC LIMIT 10`;

        const prompts = await client.pool.query(
            query,
            categoria !== 'all' ? [categoria] : []
        );

        if (prompts.rows.length === 0) {
            await interaction.reply({
                content: 'Nenhum prompt encontrado nesta categoria.',
                ephemeral: true
            });
            return;
        }

        // Cria embed com lista de prompts
        const embed = new EmbedBuilder()
            .setTitle('🎨 Prompts da Comunidade')
            .setColor('#0099ff')
            .setDescription(`Mostrando os prompts mais populares ${categoria !== 'all' ? `da categoria ${categoria}` : ''}`);

        prompts.rows.forEach((prompt, index) => {
            embed.addFields({
                name: `#${index + 1} - Por ${prompt.discord_id}`,
                value: `Prompt: ${prompt.prompt}\n` +
                       `Negativo: ${prompt.negative_prompt || 'Nenhum'}\n` +
                       `Resolução: ${prompt.width}x${prompt.height}\n` +
                       `❤️ ${prompt.likes} curtidas`,
                inline: false
            });
        });

        // Adiciona botões de interação
        const row = new ActionRowBuilder()
            .addComponents(
                new ButtonBuilder()
                    .setCustomId('prompt_prev')
                    .setLabel('⬅️ Anterior')
                    .setStyle(ButtonStyle.Secondary)
                    .setDisabled(true),
                new ButtonBuilder()
                    .setCustomId('prompt_next')
                    .setLabel('Próximo ➡️')
                    .setStyle(ButtonStyle.Secondary)
            );

        await interaction.reply({
            embeds: [embed],
            components: [row],
            ephemeral: true
        });

    } catch (error) {
        logger.error('Erro ao explorar prompts:', error);
        throw error;
    }
}

async function handleFavorites(interaction, client) {
    try {
        // Busca prompts favoritos
        const favorites = await client.pool.query(
            `SELECT sp.*, u.discord_id
            FROM shared_prompts sp
            JOIN users u ON sp.user_id = u.id
            JOIN prompt_likes pl ON sp.id = pl.prompt_id
            WHERE pl.user_id = (SELECT id FROM users WHERE discord_id = $1)
            ORDER BY pl.created_at DESC`,
            [interaction.user.id]
        );

        if (favorites.rows.length === 0) {
            await interaction.reply({
                content: 'Você ainda não tem prompts favoritos.',
                ephemeral: true
            });
            return;
        }

        // Cria embed com lista de favoritos
        const embed = new EmbedBuilder()
            .setTitle('❤️ Seus Prompts Favoritos')
            .setColor('#ff0066');

        favorites.rows.forEach((prompt, index) => {
            embed.addFields({
                name: `#${index + 1} - Por ${prompt.discord_id}`,
                value: `Prompt: ${prompt.prompt}\n` +
                       `Negativo: ${prompt.negative_prompt || 'Nenhum'}\n` +
                       `Resolução: ${prompt.width}x${prompt.height}`,
                inline: false
            });
        });

        await interaction.reply({
            embeds: [embed],
            ephemeral: true
        });

    } catch (error) {
        logger.error('Erro ao listar favoritos:', error);
        throw error;
    }
} 