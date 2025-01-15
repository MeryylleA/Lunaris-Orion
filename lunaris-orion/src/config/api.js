const { createLogger, format, transports } = require('winston');
const fetch = require('node-fetch');

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.File({ filename: 'logs/api.log' })
    ]
});

class ApiService {
    constructor() {
        this.baseUrl = process.env.GENERATION_API_URL;
        this.adminKey = process.env.GENERATION_API_ADMIN_KEY;
    }

    // Sincroniza uma nova API key com o servidor de geração
    async syncApiKey(userId, apiKey, isPremium) {
        try {
            const response = await fetch(`${this.baseUrl}/admin/keys/sync`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.adminKey}`
                },
                body: JSON.stringify({
                    user_id: userId,
                    api_key: apiKey,
                    is_premium: isPremium,
                    daily_limit: isPremium ? -1 : 50, // -1 = ilimitado
                    created_at: new Date().toISOString()
                })
            });

            if (!response.ok) {
                throw new Error(`Erro ao sincronizar key: ${response.statusText}`);
            }

            const data = await response.json();
            logger.info('API key sincronizada com sucesso:', {
                userId,
                keyPrefix: apiKey.substring(0, 4)
            });

            return data;
        } catch (error) {
            logger.error('Erro ao sincronizar API key:', error);
            throw error;
        }
    }

    // Revoga uma API key no servidor de geração
    async revokeApiKey(apiKey) {
        try {
            const response = await fetch(`${this.baseUrl}/admin/keys/revoke`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.adminKey}`
                },
                body: JSON.stringify({
                    api_key: apiKey
                })
            });

            if (!response.ok) {
                throw new Error(`Erro ao revogar key: ${response.statusText}`);
            }

            logger.info('API key revogada com sucesso:', {
                keyPrefix: apiKey.substring(0, 4)
            });

            return true;
        } catch (error) {
            logger.error('Erro ao revogar API key:', error);
            throw error;
        }
    }

    // Atualiza o status premium de uma API key
    async updateKeyStatus(apiKey, isPremium) {
        try {
            const response = await fetch(`${this.baseUrl}/admin/keys/update`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.adminKey}`
                },
                body: JSON.stringify({
                    api_key: apiKey,
                    is_premium: isPremium,
                    daily_limit: isPremium ? -1 : 50
                })
            });

            if (!response.ok) {
                throw new Error(`Erro ao atualizar key: ${response.statusText}`);
            }

            logger.info('Status da API key atualizado:', {
                keyPrefix: apiKey.substring(0, 4),
                isPremium
            });

            return true;
        } catch (error) {
            logger.error('Erro ao atualizar status da API key:', error);
            throw error;
        }
    }

    // Verifica o status de uma API key
    async checkKeyStatus(apiKey) {
        try {
            const response = await fetch(`${this.baseUrl}/admin/keys/status`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${this.adminKey}`,
                    'X-Api-Key': apiKey
                }
            });

            if (!response.ok) {
                throw new Error(`Erro ao verificar key: ${response.statusText}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            logger.error('Erro ao verificar status da API key:', error);
            throw error;
        }
    }
}

module.exports = new ApiService(); 