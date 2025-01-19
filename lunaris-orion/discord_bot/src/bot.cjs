const { Client, GatewayIntentBits, Collection } = require('discord.js');
const fs = require('node:fs');
const path = require('node:path');
require('dotenv').config();

const client = new Client({ intents: [GatewayIntentBits.Guilds] });
client.commands = new Collection();

// Load commands
const commandsPath = path.join(__dirname, 'commands');
const commandFiles = fs.readdirSync(commandsPath).filter(file => file.endsWith('.cjs'));

for (const file of commandFiles) {
    const filePath = path.join(commandsPath, file);
    const command = require(filePath);
    if ('data' in command && 'execute' in command) {
        client.commands.set(command.data.name, command);
    }
}

client.once('ready', () => {
    console.log('Lunaris Orion Bot is ready for pixel art generation!');
});

client.on('interactionCreate', async interaction => {
    if (!interaction.isChatInputCommand()) return;

    const command = client.commands.get(interaction.commandName);
    if (!command) return;

    try {
        await command.execute(interaction);
    } catch (error) {
        console.error(error);
        
        // Check if interaction has been deferred or replied to
        if (interaction.deferred || interaction.replied) {
            await interaction.editReply({ 
                content: 'There was an error while executing this command!',
                ephemeral: true 
            }).catch(console.error);
        } else {
            await interaction.reply({ 
                content: 'There was an error while executing this command!',
                ephemeral: true 
            }).catch(console.error);
        }
    }
});

client.login(process.env.DISCORD_TOKEN); 