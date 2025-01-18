const { SlashCommandBuilder } = require('discord.js');
const path = require('path');
const { spawn } = require('child_process');

module.exports = {
    data: new SlashCommandBuilder()
        .setName('generate')
        .setDescription('Generate pixel art from a text prompt')
        .addStringOption(option =>
            option.setName('prompt')
                .setDescription('Text description of the pixel art you want to generate')
                .setRequired(true))
        .addNumberOption(option =>
            option.setName('temperature')
                .setDescription('Controls randomness (0.1 to 1.0, default 0.7)')
                .setMinValue(0.1)
                .setMaxValue(1.0)
                .setRequired(false)),

    async execute(interaction) {
        await interaction.deferReply();

        try {
            const prompt = interaction.options.getString('prompt');
            const temperature = interaction.options.getNumber('temperature') || 0.7;
            
            // Create unique filename for this generation
            const timestamp = Date.now();
            const outputPath = path.join(__dirname, '..', '..', 'generated', `${timestamp}.png`);
            
            // Prepare Python script path
            const scriptPath = path.join(__dirname, '..', '..', '..', 'models', 'mini', 'inference.py');
            const checkpointPath = path.join(__dirname, '..', '..', '..', 'models', 'mini', 'checkpoints', 'best_model.pt');
            
            // Run Python script
            const pythonProcess = spawn('python', [
                scriptPath,
                '--prompt', prompt,
                '--output', outputPath,
                '--checkpoint', checkpointPath,
                '--temperature', temperature.toString()
            ]);

            let errorOutput = '';

            pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            pythonProcess.on('close', async (code) => {
                if (code !== 0) {
                    console.error('Generation failed:', errorOutput);
                    await interaction.editReply('Sorry, there was an error generating your pixel art. Please try again later.');
                    return;
                }

                await interaction.editReply({
                    content: `Here's your pixel art for: "${prompt}"`,
                    files: [outputPath]
                });
            });

        } catch (error) {
            console.error('Error in generate command:', error);
            await interaction.editReply('Sorry, there was an error processing your request. Please try again later.');
        }
    },
}; 