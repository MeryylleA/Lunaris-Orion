const { createLogger, format, transports } = require('winston');
const fetch = require('node-fetch');

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.File({ filename: 'logs/api.log' }),
        new transports.Console({
            format: format.combine(
                format.colorize(),
                format.simple()
            )
        })
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
            logger.debug('Iniciando sincronização de API key', {
                userId,
                keyPrefix: apiKey.substring(0, 4),
                isPremium
            });

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
                const errorText = await response.text();
                throw new Error(`Erro ao sincronizar key: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            logger.info('API key sincronizada com sucesso:', {
                userId,
                keyPrefix: apiKey.substring(0, 4),
                isPremium
            });

            return data;
        } catch (error) {
            logger.error('Erro ao sincronizar API key:', {
                error: error.message,
                userId,
                keyPrefix: apiKey.substring(0, 4)
            });
            throw error;
        }
    }

    // Revoga uma API key no servidor de geração
    async revokeApiKey(apiKey) {
        try {
            logger.debug('Iniciando revogação de API key', {
                keyPrefix: apiKey.substring(0, 4)
            });

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
                const errorText = await response.text();
                throw new Error(`Erro ao revogar key: ${response.status} - ${errorText}`);
            }

            logger.info('API key revogada com sucesso:', {
                keyPrefix: apiKey.substring(0, 4)
            });

            return true;
        } catch (error) {
            logger.error('Erro ao revogar API key:', {
                error: error.message,
                keyPrefix: apiKey.substring(0, 4)
            });
            throw error;
        }
    }

    // Atualiza o status premium de uma API key
    async updateKeyStatus(apiKey, isPremium) {
        try {
            logger.debug('Iniciando atualização de status da API key', {
                keyPrefix: apiKey.substring(0, 4),
                isPremium
            });

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
                const errorText = await response.text();
                throw new Error(`Erro ao atualizar key: ${response.status} - ${errorText}`);
            }

            logger.info('Status da API key atualizado:', {
                keyPrefix: apiKey.substring(0, 4),
                isPremium
            });

            return true;
        } catch (error) {
            logger.error('Erro ao atualizar status da API key:', {
                error: error.message,
                keyPrefix: apiKey.substring(0, 4)
            });
            throw error;
        }
    }

    // Verifica o status de uma API key
    async checkKeyStatus(apiKey) {
        try {
            logger.debug('Verificando status da API key', {
                keyPrefix: apiKey.substring(0, 4)
            });

            const response = await fetch(`${this.baseUrl}/admin/keys/status`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${this.adminKey}`,
                    'X-Api-Key': apiKey
                }
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Erro ao verificar key: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            logger.info('Status da API key verificado:', {
                keyPrefix: apiKey.substring(0, 4),
                status: data.status
            });

            return data;
        } catch (error) {
            logger.error('Erro ao verificar status da API key:', {
                error: error.message,
                keyPrefix: apiKey.substring(0, 4)
            });
            throw error;
        }
    }

    // Verifica a saúde do serviço de geração
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${this.adminKey}`
                }
            });

            return {
                status: response.ok ? 'healthy' : 'unhealthy',
                statusCode: response.status
            };
        } catch (error) {
            logger.error('Erro ao verificar saúde do serviço:', error);
            return {
                status: 'unhealthy',
                error: error.message
            };
        }
    }
}

module.exports = new ApiService();