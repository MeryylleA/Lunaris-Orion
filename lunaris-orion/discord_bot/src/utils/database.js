import pg from 'pg';
import { createLogger, format, transports } from 'winston';

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

class Database {
    constructor() {
        this.pool = new pg.Pool({
            host: process.env.DB_HOST,
            port: process.env.DB_PORT,
            database: process.env.DB_NAME,
            user: process.env.DB_USER,
            password: process.env.DB_PASSWORD,
            max: 20,
            idleTimeoutMillis: 30000,
            connectionTimeoutMillis: 2000,
        });
        
        // Teste de conexão
        this.pool.on('connect', () => {
            logger.info('Conexão com banco de dados estabelecida');
        });
        
        this.pool.on('error', (err) => {
            logger.error('Erro no pool de conexões:', err);
        });
    }
    
    async getUserData(discordId) {
        try {
            const result = await this.pool.query(
                `SELECT u.*, s.tier, s.status, s.expires_at, k.key_hash
                FROM users u
                LEFT JOIN subscriptions s ON u.id = s.user_id
                LEFT JOIN api_keys k ON u.id = k.user_id
                WHERE u.discord_id = $1`,
                [discordId]
            );
            
            return result.rows[0] || null;
            
        } catch (error) {
            logger.error('Erro ao buscar dados do usuário:', error);
            throw error;
        }
    }
    
    async createUser(discordId) {
        try {
            const result = await this.pool.query(
                `INSERT INTO users (discord_id, created_at)
                VALUES ($1, NOW())
                RETURNING *`,
                [discordId]
            );
            
            return result.rows[0];
            
        } catch (error) {
            logger.error('Erro ao criar usuário:', error);
            throw error;
        }
    }
    
    async updateUserSubscription(userId, subscriptionData) {
        try {
            const result = await this.pool.query(
                `INSERT INTO subscriptions (
                    user_id, stripe_subscription_id, plan_id,
                    status, current_period_start, current_period_end
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id) DO UPDATE
                SET stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                    plan_id = EXCLUDED.plan_id,
                    status = EXCLUDED.status,
                    current_period_start = EXCLUDED.current_period_start,
                    current_period_end = EXCLUDED.current_period_end
                RETURNING *`,
                [
                    userId,
                    subscriptionData.stripeSubscriptionId,
                    subscriptionData.planId,
                    subscriptionData.status,
                    subscriptionData.periodStart,
                    subscriptionData.periodEnd
                ]
            );
            
            return result.rows[0];
            
        } catch (error) {
            logger.error('Erro ao atualizar assinatura:', error);
            throw error;
        }
    }
    
    async saveApiKey(userId, keyData) {
        try {
            const result = await this.pool.query(
                `INSERT INTO api_keys (
                    user_id, key_hash, expires_at, usage_limit
                ) VALUES ($1, $2, $3, $4)
                RETURNING *`,
                [
                    userId,
                    keyData.keyHash,
                    keyData.expiresAt,
                    keyData.usageLimit
                ]
            );
            
            return result.rows[0];
            
        } catch (error) {
            logger.error('Erro ao salvar chave API:', error);
            throw error;
        }
    }
    
    async revokeApiKey(userId, keyId) {
        try {
            const result = await this.pool.query(
                `UPDATE api_keys
                SET revoked_at = NOW(), active = false
                WHERE user_id = $1 AND id = $2
                RETURNING *`,
                [userId, keyId]
            );
            
            return result.rows[0];
            
        } catch (error) {
            logger.error('Erro ao revogar chave API:', error);
            throw error;
        }
    }
    
    async listApiKeys(userId) {
        try {
            const result = await this.pool.query(
                `SELECT id, created_at, expires_at, usage_limit, usage_count, active
                FROM api_keys
                WHERE user_id = $1
                ORDER BY created_at DESC`,
                [userId]
            );
            
            return result.rows;
            
        } catch (error) {
            logger.error('Erro ao listar chaves API:', error);
            throw error;
        }
    }
    
    async close() {
        try {
            await this.pool.end();
            logger.info('Conexão com banco de dados encerrada');
        } catch (error) {
            logger.error('Erro ao encerrar conexão:', error);
            throw error;
        }
    }
}

export default new Database(); 