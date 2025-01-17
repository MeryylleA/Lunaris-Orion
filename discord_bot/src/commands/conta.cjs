const { SlashCommandBuilder, EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle } = require('discord.js');
const { Pool } = require('pg');
const crypto = require('crypto');

const pool = new Pool({
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    ssl: {
        rejectUnauthorized: false
    }
});

async function getUserByDiscordId(discordId, username) {
    const client = await pool.connect();
    try {
        let result = await client.query(
            'SELECT * FROM users WHERE discord_id = $1',
            [discordId]
        );

        if (!result.rows[0]) {
            result = await client.query(
                'INSERT INTO users (discord_id, username, created_at) VALUES ($1, $2, NOW()) RETURNING *',
                [discordId, username]
            );
        }

        return result.rows[0];
    } finally {
        client.release();
    }
}

async function getUserApiKey(userId) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'SELECT * FROM api_keys WHERE user_id = $1 AND NOT revoked AND expires_at > NOW() ORDER BY created_at DESC LIMIT 1',
            [userId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function generateApiKey(discordId, username) {
    const client = await pool.connect();
    try {
        // Get user info
        const user = await getUserByDiscordId(discordId, username);
        
        let key;
        let isUnique = false;
        let attempts = 0;
        const MAX_ATTEMPTS = 5;

        // Keep generating keys until we find a unique one or reach max attempts
        while (!isUnique && attempts < MAX_ATTEMPTS) {
            // Generate user-friendly part (6 chars)
            const userPart = username.replace(/[^a-zA-Z0-9]/g, '').toUpperCase().slice(0, 6).padEnd(6, '0');
            
            // Generate secure hash (6 chars)
            const hash = crypto.createHash('sha256')
                .update(discordId + Date.now().toString() + attempts.toString())
                .digest('hex')
                .toUpperCase()
                .slice(0, 6);
            
            // Combine parts with prefix
            key = `LUN-${userPart}-${hash}`;

            // Check if key already exists
            const existingKey = await client.query(
                'SELECT id FROM api_keys WHERE key = $1',
                [key]
            );

            if (existingKey.rows.length === 0) {
                isUnique = true;
            } else {
                attempts++;
            }
        }

        if (!isUnique) {
            throw new Error('Failed to generate a unique API key after multiple attempts');
        }
        
        // Define expiration (30 days)
        const expiresAt = new Date();
        expiresAt.setDate(expiresAt.getDate() + 30);

        // Save to database
        const result = await client.query(
            'INSERT INTO api_keys (user_id, key, expires_at, created_at) VALUES ($1, $2, $3, NOW()) RETURNING *',
            [user.id, key, expiresAt]
        );

        return { key, expiresAt };
    } finally {
        client.release();
    }
}

async function handleAccountInfo(interaction) {
    try {
        await interaction.deferReply({ ephemeral: true });

        const user = await getUserByDiscordId(interaction.user.id, interaction.user.username);
        const apiKey = await getUserApiKey(user.id);
        
        const embed = new EmbedBuilder()
            .setColor('#2b2d31')
            .setTitle('🌙 Account Information')
            .setDescription(`Welcome back, **${interaction.user.username}**! Here's your account information.`)
            .addFields([
                {
                    name: '👤 Account Details',
                    value: [
                        `**Discord ID:** \`${interaction.user.id}\``,
                        `**Username:** ${interaction.user.username}`,
                        `**Created At:** <t:${Math.floor(new Date(user.created_at).getTime() / 1000)}:R>`
                    ].join('\n'),
                    inline: false
                },
                {
                    name: '🎨 Aseprite Extension Access',
                    value: apiKey 
                        ? [
                            '**Status:** ✅ Active',
                            `**Expires:** <t:${Math.floor(new Date(apiKey.expires_at).getTime() / 1000)}:R>`
                          ].join('\n')
                        : '**Status:** ❌ No active access key\nUse `/account api generate` to create one',
                    inline: false
                }
            ])
            .setFooter({ 
                text: 'Lunaris Orion • AI Image Generation for Aseprite',
                iconURL: interaction.client.user.displayAvatarURL()
            })
            .setTimestamp();

        await interaction.editReply({
            embeds: [embed],
            ephemeral: true
        });

    } catch (error) {
        console.error('Error getting account information:', error);
        await interaction.editReply({
            content: 'An error occurred while fetching your account information. Please try again later.',
            ephemeral: true
        });
    }
}

async function handleApiCommand(interaction) {
    try {
        await interaction.deferReply({ ephemeral: true });

        const action = interaction.options.getString('action');
        const user = await getUserByDiscordId(interaction.user.id, interaction.user.username);

        switch (action) {
            case 'generate': {
                const existingKey = await getUserApiKey(user.id);
                if (existingKey) {
                    const embed = new EmbedBuilder()
                        .setColor('#ff3838')
                        .setTitle('❌ Active Key Found')
                        .setDescription('You already have an active access key. Please revoke it first if you want to generate a new one.')
                        .addFields({
                            name: '⏰ Current Key Expires',
                            value: `<t:${Math.floor(new Date(existingKey.expires_at).getTime() / 1000)}:R>`
                        })
                        .setFooter({ 
                            text: 'Use /account api revoke to revoke your current key',
                            iconURL: interaction.client.user.displayAvatarURL()
                        });

                    await interaction.editReply({
                        embeds: [embed],
                        ephemeral: true
                    });
                    return;
                }

                const { key, expiresAt } = await generateApiKey(interaction.user.id, interaction.user.username);
                const embed = new EmbedBuilder()
                    .setColor('#43b581')
                    .setTitle('🎨 New Access Key Generated')
                    .setDescription('This key will be used to authenticate your Lunaris Orion extension in Aseprite. Keep it safe!')
                    .addFields([
                        {
                            name: '🔐 Your Access Key',
                            value: `\`${key}\``,
                            inline: false
                        },
                        {
                            name: '⚡ Features',
                            value: [
                                '• Access to Lunaris Orion in Aseprite',
                                '• AI-powered image generation',
                                '• Seamless integration with your workflow',
                                '• Generation limits based on your plan'
                            ].join('\n'),
                            inline: true
                        },
                        {
                            name: '⏰ Key Details',
                            value: [
                                `**Created:** <t:${Math.floor(Date.now() / 1000)}:R>`,
                                `**Expires:** <t:${Math.floor(expiresAt.getTime() / 1000)}:R>`
                            ].join('\n'),
                            inline: true
                        }
                    ])
                    .setFooter({ 
                        text: 'Enter this key in your Lunaris Orion extension settings',
                        iconURL: interaction.client.user.displayAvatarURL()
                    })
                    .setTimestamp();

                await interaction.editReply({
                    embeds: [embed],
                    ephemeral: true
                });
                break;
            }

            case 'revoke': {
                const client = await pool.connect();
                try {
                    await client.query(
                        'UPDATE api_keys SET revoked = true WHERE user_id = $1 AND NOT revoked',
                        [user.id]
                    );

                    const embed = new EmbedBuilder()
                        .setColor('#ff7f50')
                        .setTitle('🔒 Access Key Revoked')
                        .setDescription('Your access key has been successfully revoked. Your Aseprite extension will no longer work with this key.')
                        .addFields({
                            name: '📝 Next Steps',
                            value: 'You can generate a new access key using `/account api generate`'
                        })
                        .setFooter({ 
                            text: 'Lunaris Orion • Security First',
                            iconURL: interaction.client.user.displayAvatarURL()
                        })
                        .setTimestamp();

                    await interaction.editReply({
                        embeds: [embed],
                        ephemeral: true
                    });
                } finally {
                    client.release();
                }
                break;
            }

            default:
                await interaction.editReply({
                    content: 'Invalid action.',
                    ephemeral: true
                });
        }

    } catch (error) {
        console.error('Error processing API command:', error);
        await interaction.editReply({
            content: 'An error occurred while processing your request. Please try again later.',
            ephemeral: true
        });
    }
}

module.exports = {
    data: new SlashCommandBuilder()
        .setName('account')
        .setDescription('Manage your account')
        .addSubcommand(subcommand =>
            subcommand
                .setName('info')
                .setDescription('Show your account information'))
        .addSubcommand(subcommand =>
            subcommand
                .setName('api')
                .setDescription('Manage your Aseprite extension access key')
                .addStringOption(option =>
                    option
                        .setName('action')
                        .setDescription('Action to perform')
                        .setRequired(true)
                        .addChoices(
                            { name: 'Generate New Key', value: 'generate' },
                            { name: 'Revoke Current Key', value: 'revoke' }
                        ))),

    async execute(interaction) {
        const subcommand = interaction.options.getSubcommand();

        switch (subcommand) {
            case 'info':
                await handleAccountInfo(interaction);
                break;
            case 'api':
                await handleApiCommand(interaction);
                break;
            default:
                await interaction.reply({
                    content: 'Invalid subcommand.',
                    ephemeral: true
                });
        }
    },
}; 