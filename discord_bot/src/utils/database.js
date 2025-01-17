import pkg from 'pg';
const { Pool } = pkg;
import winston from 'winston';

// Configuração do pool de conexão
const pool = new Pool({
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    ssl: process.env.DB_SSL === 'true' ? {
        rejectUnauthorized: false
    } : false,
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
});

// Função para gerar chave API única
const generateApiKey = () => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    const prefix = 'LUN';
    let key = prefix;
    for (let i = 0; i < 29; i++) {
        key += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return key;
};

// Funções de utilidade para o banco de dados
const db = {
    // Verifica se um usuário existe
    async userExists(userId) {
        const result = await pool.query('SELECT id FROM users WHERE discord_id = $1', [userId]);
        return result.rows.length > 0;
    },

    // Obtém informações do usuário
    async getUserInfo(userId) {
        const result = await pool.query(`
            SELECT u.*, 
                   COUNT(DISTINCT ak.id) as total_keys,
                   COUNT(DISTINCT CASE WHEN ak.expires_at > NOW() THEN ak.id END) as active_keys,
                   s.plan_type,
                   s.expires_at as subscription_expires
            FROM users u
            LEFT JOIN api_keys ak ON u.id = ak.user_id
            LEFT JOIN subscriptions s ON u.id = s.user_id AND s.expires_at > NOW()
            WHERE u.discord_id = $1
            GROUP BY u.id, s.plan_type, s.expires_at
        `, [userId]);
        return result.rows[0];
    },

    // Cria um novo usuário
    async createUser(userId, username) {
        return await pool.query(
            'INSERT INTO users (discord_id, username, created_at) VALUES ($1, $2, NOW()) RETURNING *',
            [userId, username]
        );
    },

    // Gera uma nova chave API
    async generateNewApiKey(userId) {
        const apiKey = generateApiKey();
        const expiresAt = new Date();
        expiresAt.setDate(expiresAt.getDate() + 30); // 30 dias de validade

        const result = await pool.query(
            'INSERT INTO api_keys (user_id, key, created_at, expires_at) VALUES ($1, $2, NOW(), $3) RETURNING *',
            [userId, apiKey, expiresAt]
        );
        return result.rows[0];
    },

    // Lista as chaves API ativas do usuário
    async listActiveApiKeys(userId) {
        return await pool.query(
            'SELECT * FROM api_keys WHERE user_id = $1 AND expires_at > NOW() ORDER BY created_at DESC',
            [userId]
        );
    },

    // Revoga uma chave API
    async revokeApiKey(userId, keyId) {
        return await pool.query(
            'UPDATE api_keys SET revoked = true WHERE user_id = $1 AND id = $2',
            [userId, keyId]
        );
    },

    // Obtém estatísticas de uso
    async getUserStats(userId) {
        const result = await pool.query(`
            SELECT 
                COUNT(g.id) as total_generations,
                COUNT(DISTINCT DATE(g.created_at)) as active_days,
                MAX(g.created_at) as last_generation
            FROM generations g
            WHERE g.user_id = $1
        `, [userId]);
        return result.rows[0];
    },

    // Fecha o pool de conexões
    async close() {
        await pool.end();
    }
};

// Exporta o módulo
export default db; 