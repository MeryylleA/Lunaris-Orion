const { SlashCommandBuilder } = require('discord.js');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Function to find the most recent run directory
function findLatestRunDirectory(modelDir) {
    const runsPath = path.join(__dirname, '..', '..', '..', 'models', modelDir, 'runs');
    if (!fs.existsSync(runsPath)) {
        throw new Error(`No runs directory found for ${modelDir} model`);
    }

    const runDirs = fs.readdirSync(runsPath)
        .filter(dir => dir.startsWith('run_'))
        .map(dir => ({
            name: dir,
            path: path.join(runsPath, dir),
            time: fs.statSync(path.join(runsPath, dir)).mtime.getTime()
        }))
        .sort((a, b) => b.time - a.time);

    if (runDirs.length === 0) {
        throw new Error(`No run directories found for ${modelDir} model`);
    }

    return runDirs[0].path;
}

// Function to find the best checkpoint
function findBestCheckpoint(runDir) {
    const checkpointsDir = path.join(runDir, 'checkpoints');
    if (!fs.existsSync(checkpointsDir)) {
        throw new Error('No checkpoints directory found');
    }

    const checkpoints = fs.readdirSync(checkpointsDir)
        .filter(file => file.endsWith('.pt'))
        .map(file => ({
            name: file,
            path: path.join(checkpointsDir, file),
            time: fs.statSync(path.join(checkpointsDir, file)).mtime.getTime()
        }))
        .sort((a, b) => b.time - a.time);

    if (checkpoints.length === 0) {
        throw new Error('No checkpoint files found');
    }

    return checkpoints[0].path;
}

// Queue system for managing generation requests
class GenerationQueue {
    constructor() {
        this.queues = {
            mini: [],
            large: []
        };
        this.processing = {
            mini: new Set(),
            large: new Set()
        };
        this.maxConcurrent = {
            mini: 3,  // Allow 3 concurrent Mini generations
            large: 1  // Only 1 Large generation at a time
        };
        this.timeoutMs = 10 * 60 * 1000; // 10 minutes timeout
        this.avgProcessingTime = {
            mini: 30 * 1000,  // 30 seconds for Mini
            large: 90 * 1000  // 90 seconds for Large
        };
    }

    // Add a request to the queue
    addToQueue(model, interaction, prompt, temperature) {
        const queue = this.queues[model];
        const timestamp = Date.now();
        
        // Create queue entry
        const entry = {
            interaction,
            prompt,
            temperature,
            timestamp,
            userId: interaction.user.id
        };

        // Add to appropriate queue
        queue.push(entry);
        
        // Calculate position and estimated time
        const position = queue.length;
        const activeJobs = this.processing[model].size;
        const estimatedWaitSeconds = Math.round(
            ((position - 1) / this.maxConcurrent[model] * this.avgProcessingTime[model] +
            (activeJobs > 0 ? this.avgProcessingTime[model] : 0)) / 1000
        );

        // Return queue information
        return {
            position,
            estimatedWaitSeconds
        };
    }

    // Process next items in queue
    async processQueue(model) {
        const queue = this.queues[model];
        const processing = this.processing[model];
        
        // Remove timed out requests
        const now = Date.now();
        this.queues[model] = queue.filter(entry => {
            if (now - entry.timestamp > this.timeoutMs) {
                entry.interaction.editReply({
                    content: 'Your generation request has timed out. Please try again.',
                    ephemeral: true
                }).catch(console.error);
                return false;
            }
            return true;
        });

        // Process as many items as we can
        while (this.queues[model].length > 0 && processing.size < this.maxConcurrent[model]) {
            const entry = this.queues[model].shift();
            processing.add(entry.userId);
            
            try {
                await this.generateImage(model, entry);
            } finally {
                processing.delete(entry.userId);
                // Continue processing queue
                this.processQueue(model);
            }
        }
    }

    // Generate image for a queue entry
    async generateImage(model, entry) {
        const { interaction, prompt, temperature } = entry;
        
        try {
            // Find latest run directory and best checkpoint
            const runDir = findLatestRunDirectory(model);
            const checkpointPath = findBestCheckpoint(runDir);
            
            // Create unique filename
            const timestamp = Date.now();
            const outputPath = path.join(__dirname, '..', '..', 'generated', `${timestamp}.png`);
            
            // Prepare paths
            const modelDir = model;
            const scriptPath = path.join(__dirname, '..', '..', '..', 'models', modelDir, 'inference.py');

            return new Promise((resolve, reject) => {
                console.log('Using checkpoint:', checkpointPath);
                
                const pythonProcess = spawn('/workspace/.miniconda3/bin/python3', [
                    scriptPath,
                    '--prompt', prompt,
                    '--output', outputPath,
                    '--checkpoint', checkpointPath,
                    '--temperature', temperature.toString()
                ], {
                    env: { 
                        ...process.env, 
                        PYTHONPATH: process.env.PYTHONPATH || '',
                        PATH: `/workspace/.miniconda3/bin:${process.env.PATH || ''}`
                    },
                    stdio: ['pipe', 'pipe', 'pipe']
                });

                let errorOutput = '';
                let stdOutput = '';

                pythonProcess.stderr.on('data', (data) => {
                    errorOutput += data.toString();
                    console.error(`Python Error: ${data}`);
                });

                pythonProcess.stdout.on('data', (data) => {
                    stdOutput += data.toString();
                    console.log(`Python Output: ${data}`);
                });

                pythonProcess.on('close', async (code) => {
                    if (code === 0) {
                        await interaction.editReply({
                            content: `Generated pixel art using ${model.toUpperCase()} model for prompt: "${prompt}"`,
                            files: [outputPath]
                        });
                        resolve();
                    } else {
                        console.error('Full Python Error:', errorOutput);
                        console.error('Full Python Output:', stdOutput);
                        await interaction.editReply({
                            content: `Failed to generate pixel art with ${model.toUpperCase()} model. Please try again.`,
                            ephemeral: true
                        });
                        reject(new Error('Generation failed'));
                    }
                });

                pythonProcess.on('error', async (error) => {
                    console.error('Process Error:', error);
                    console.error('Python Error Output:', errorOutput);
                    await interaction.editReply({
                        content: `Failed to start generation process with ${model.toUpperCase()} model. Please try again.`,
                        ephemeral: true
                    });
                    reject(error);
                });
            });
        } catch (error) {
            console.error('Setup Error:', error);
            await interaction.editReply({
                content: `Failed to setup generation process: ${error.message}`,
                ephemeral: true
            });
            throw error;
        }
    }
}

// Create global queue instance
const generationQueue = new GenerationQueue();

module.exports = {
    data: new SlashCommandBuilder()
        .setName('generate')
        .setDescription('Generate pixel art from a text prompt')
        .addStringOption(option =>
            option.setName('prompt')
                .setDescription('Text description of the pixel art you want to generate')
                .setRequired(true))
        .addStringOption(option =>
            option.setName('model')
                .setDescription('Model to use for generation')
                .setRequired(true)
                .addChoices(
                    { name: 'Mini - Fast, lightweight model', value: 'mini' },
                    { name: 'Large - High quality, slower model', value: 'large' }
                ))
        .addNumberOption(option =>
            option.setName('temperature')
                .setDescription('Controls randomness in generation (0.1 to 1.0)')
                .setMinValue(0.1)
                .setMaxValue(1.0)
                .setRequired(false)),

    async execute(interaction) {
        try {
            await interaction.deferReply();

            const prompt = interaction.options.getString('prompt');
            const model = interaction.options.getString('model');
            const temperature = interaction.options.getNumber('temperature') || 0.7;

            // Add request to queue and get position info
            const queueInfo = generationQueue.addToQueue(model, interaction, prompt, temperature);

            // Inform user about queue position
            if (queueInfo.position > 1 || generationQueue.processing[model].size > 0) {
                await interaction.editReply({
                    content: `Your request has been queued! Position: ${queueInfo.position}\nEstimated wait time: ${queueInfo.estimatedWaitSeconds} seconds\nUsing ${model.toUpperCase()} model for prompt: "${prompt}"`,
                });
            }

            // Start processing queue if not already processing
            if (generationQueue.processing[model].size < generationQueue.maxConcurrent[model]) {
                generationQueue.processQueue(model);
            }

        } catch (error) {
            console.error('Command Error:', error);
            await interaction.editReply({
                content: 'An unexpected error occurred. Please try again.',
                ephemeral: true
            });
        }
    },
}; 