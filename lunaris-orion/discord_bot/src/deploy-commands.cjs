require('dotenv').config();
const { REST, Routes } = require('discord.js');
const fs = require('fs');
const path = require('path');

// Logs de diagnóstico
console.log('Current working directory:', process.cwd());
console.log('Environment variables loaded:', {
    tokenExists: !!process.env.DISCORD_TOKEN,
    tokenLength: process.env.DISCORD_TOKEN ? process.env.DISCORD_TOKEN.length : 0,
    clientIdExists: !!process.env.CLIENT_ID,
    guildIdExists: !!process.env.GUILD_ID
});

const commands = [];
const commandsPath = path.join(__dirname, 'commands');
const commandFiles = fs.readdirSync(commandsPath).filter(file => file.endsWith('.cjs'));

for (const file of commandFiles) {
    const filePath = path.join(commandsPath, file);
    const command = require(filePath);
    if ('data' in command) {
        commands.push(command.data.toJSON());
    }
}

const rest = new REST().setToken(process.env.DISCORD_TOKEN);

(async () => {
    try {
        console.log(`Started refreshing ${commands.length} application (/) commands.`);

        const data = await rest.put(
            Routes.applicationGuildCommands(process.env.CLIENT_ID, process.env.GUILD_ID),
            { body: commands },
        );

        console.log(`Successfully reloaded ${data.length} application (/) commands.`);
    } catch (error) {
        console.error(error);
    }
})(); 