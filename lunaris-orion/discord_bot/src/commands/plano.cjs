const { SlashCommandBuilder, EmbedBuilder } = require('discord.js');
const { Pool } = require('pg');
const Stripe = require('stripe');
const { createLogger, format, transports } = require('winston');

// Configuração do logger
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

// Configuração do pool de conexão
const pool = new Pool({
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    ssl: process.env.DB_SSL === 'true'
});

// Configuração do Stripe
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);

const PLANOS = {
    basic: {
        nome: 'Básico',
        preco: 'R$ 10/mês',
        recursos: [
            '100 gerações por dia',
            'Acesso a todos os estilos',
            'Resolução até 64x64'
        ]
    },
    pro: {
        nome: 'Pro',
        preco: 'R$ 25/mês',
        recursos: [
            'Gerações ilimitadas',
            'Acesso a estilos exclusivos',
            'Resolução até 128x128',
            'Prioridade na fila'
        ]
    },
    enterprise: {
        nome: 'Enterprise',
        preco: 'R$ 100/mês',
        recursos: [
            'Tudo do plano Pro',
            'API dedicada',
            'Suporte prioritário',
            'Personalização de estilos'
        ]
    }
};

const data = new SlashCommandBuilder()
    .setName('plano')
    .setDescription('Gerencia seu plano de assinatura')
    .addSubcommand(subcommand =>
        subcommand
            .setName('info')
            .setDescription('Mostra informações sobre seu plano atual'))
    .addSubcommand(subcommand =>
        subcommand
            .setName('listar')
            .setDescription('Lista todos os planos disponíveis'))
    .addSubcommand(subcommand =>
        subcommand
            .setName('assinar')
            .setDescription('Assina ou atualiza seu plano')
            .addStringOption(option =>
                option
                    .setName('tipo')
                    .setDescription('Tipo do plano')
                    .setRequired(true)
                    .addChoices(
                        { name: 'Básico', value: 'basic' },
                        { name: 'Pro', value: 'pro' },
                        { name: 'Enterprise', value: 'enterprise' }
                    )));

async function handlePlanInfo(interaction) {
    try {
        const userId = interaction.user.id;
        
        // Busca informações da assinatura
        const subscriptionResult = await pool.query(
            'SELECT * FROM subscriptions WHERE user_id = $1',
            [userId]
        );
        
        const subscription = subscriptionResult.rows[0];
        
        if (!subscription) {
            return interaction.reply({
                content: 'Você não tem nenhum plano ativo. Use `/plano listar` para ver os planos disponíveis.',
                ephemeral: true
            });
        }
        
        const embed = new EmbedBuilder()
            .setColor('#0099ff')
            .setTitle('Seu Plano')
            .addFields(
                { name: 'Plano Atual', value: PLANOS[subscription.plan_id].nome },
                { name: 'Status', value: subscription.status },
                { name: 'Próxima Cobrança', value: new Date(subscription.current_period_end).toLocaleDateString() }
            )
            .setTimestamp();
        
        return interaction.reply({ embeds: [embed], ephemeral: true });
        
    } catch (error) {
        logger.error('Erro ao obter informações do plano:', error);
        throw error;
    }
}

async function handlePlanList(interaction) {
    try {
        const embeds = Object.entries(PLANOS).map(([id, plano]) => {
            return new EmbedBuilder()
                .setColor('#0099ff')
                .setTitle(plano.nome)
                .setDescription(plano.preco)
                .addFields(
                    { name: 'Recursos', value: plano.recursos.join('\n') }
                );
        });
        
        return interaction.reply({ embeds, ephemeral: true });
        
    } catch (error) {
        logger.error('Erro ao listar planos:', error);
        throw error;
    }
}

async function handlePlanSubscribe(interaction) {
    try {
        const userId = interaction.user.id;
        const planType = interaction.options.getString('tipo');
        
        // Verifica se usuário existe
        const userResult = await pool.query(
            'SELECT * FROM users WHERE id = $1',
            [userId]
        );
        
        if (!userResult.rows[0]) {
            return interaction.reply({
                content: 'Você precisa registrar uma conta primeiro! Use `/conta registrar`',
                ephemeral: true
            });
        }
        
        // Cria sessão de checkout do Stripe
        const session = await stripe.checkout.sessions.create({
            customer_email: userResult.rows[0].email,
            client_reference_id: userId,
            payment_method_types: ['card'],
            mode: 'subscription',
            line_items: [{
                price: process.env.STRIPE_PREMIUM_PRICE_ID,
                quantity: 1,
            }],
            success_url: `${process.env.DISCORD_BOT_URL}/success?session_id={CHECKOUT_SESSION_ID}`,
            cancel_url: `${process.env.DISCORD_BOT_URL}/cancel`,
            metadata: {
                userId,
                planType
            }
        });
        
        const embed = new EmbedBuilder()
            .setColor('#0099ff')
            .setTitle('Assinar Plano')
            .setDescription(`Clique no botão abaixo para assinar o plano ${PLANOS[planType].nome}`)
            .addFields(
                { name: 'Preço', value: PLANOS[planType].preco },
                { name: 'Link de Pagamento', value: session.url }
            );
        
        return interaction.reply({ embeds: [embed], ephemeral: true });
        
    } catch (error) {
        logger.error('Erro ao processar assinatura:', error);
        throw error;
    }
}

module.exports = {
    data,
    async execute(interaction) {
        try {
            const subcommand = interaction.options.getSubcommand();
            
            switch (subcommand) {
                case 'info':
                    await handlePlanInfo(interaction);
                    break;
                case 'listar':
                    await handlePlanList(interaction);
                    break;
                case 'assinar':
                    await handlePlanSubscribe(interaction);
                    break;
            }
        } catch (error) {
            logger.error('Erro ao processar comando de plano:', error);
            await interaction.reply({
                content: 'Ocorreu um erro ao processar seu comando.',
                ephemeral: true
            });
        }
    }
}; 