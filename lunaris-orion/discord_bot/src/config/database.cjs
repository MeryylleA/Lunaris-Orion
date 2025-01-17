const { Pool } = require('pg');
const { createLogger, format, transports } = require('winston');

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.File({ filename: 'logs/database.log' }),
        new transports.Console({
            format: format.combine(
                format.colorize(),
                format.simple()
            )
        })
    ]
});

class Database {
    constructor() {
        this.pool = new Pool({
            host: process.env.DB_HOST,
            port: process.env.DB_PORT || 5432,
            database: process.env.DB_NAME || 'lunaris_orion',
            user: process.env.DB_USER || 'lunaris_app',
            password: process.env.DB_PASSWORD,
            ssl: process.env.DB_SSL === 'true'
        });

        this.pool.on('error', (err) => {
            logger.error('Erro inesperado no pool do banco:', err);
        });
    }

    async getUserApiKey(userId) {
        try {
            const result = await this.pool.query(
                `SELECT ak.*, du.count as daily_uses
                 FROM api_keys ak
                 LEFT JOIN daily_usage du ON 
                    du.api_key_id = ak.id AND 
                    du.date = CURRENT_DATE
                 WHERE ak.user_id = $1 AND ak.active = true`,
                [userId]
            );
            return result.rows[0];
        } catch (error) {
            logger.error('Erro ao buscar API key do usuário:', error);
            throw error;
        }
    }

    async saveApiKey(userId, apiKey, isPremium) {
        try {
            // Primeiro, desativa todas as chaves existentes do usuário
            await this.pool.query(
                'UPDATE api_keys SET active = false WHERE user_id = $1',
                [userId]
            );

            // Insere a nova chave
            const result = await this.pool.query(
                `INSERT INTO api_keys 
                 (user_id, key, is_premium, active, daily_limit)
                 VALUES ($1, $2, $3, true, $4)
                 RETURNING *`,
                [userId, apiKey, isPremium, isPremium ? -1 : 50]
            );

            // Cria o registro de uso diário
            await this.pool.query(
                `INSERT INTO daily_usage (user_id, api_key_id, date, count)
                 VALUES ($1, $2, CURRENT_DATE, 0)`,
                [userId, result.rows[0].id]
            );

            return result.rows[0];
        } catch (error) {
            logger.error('Erro ao salvar API key:', error);
            throw error;
        }
    }

    async revokeApiKey(userId) {
        try {
            await this.pool.query(
                'UPDATE api_keys SET active = false WHERE user_id = $1 AND active = true',
                [userId]
            );
        } catch (error) {
            logger.error('Erro ao revogar API key:', error);
            throw error;
        }
    }

    async getSubscriptionStatus(userId) {
        try {
            const result = await this.pool.query(
                'SELECT * FROM subscriptions WHERE user_id = $1 AND status = $2',
                [userId, 'active']
            );
            return result.rows[0];
        } catch (error) {
            logger.error('Erro ao buscar status da assinatura:', error);
            throw error;
        }
    }

    async incrementDailyUses(apiKeyId, userId) {
        try {
            const result = await this.pool.query(
                `INSERT INTO daily_usage (user_id, api_key_id, date, count)
                 VALUES ($1, $2, CURRENT_DATE, 1)
                 ON CONFLICT (user_id, api_key_id, date)
                 DO UPDATE SET count = daily_usage.count + 1
                 RETURNING count`,
                [userId, apiKeyId]
            );
            return result.rows[0].count;
        } catch (error) {
            logger.error('Erro ao incrementar uso diário:', error);
            throw error;
        }
    }

    async close() {
        await this.pool.end();
    }
}

module.exports = new Database();
