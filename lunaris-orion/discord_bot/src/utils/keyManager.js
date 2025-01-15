import { createLogger, format, transports } from 'winston';
import KeyGenerator from './keyGenerator.js';
import database from './database.js';

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

class KeyManager {
    constructor() {
        this.db = database;
    }
    
    /**
     * Gera uma nova chave API.
     * @param {string} userId - ID do usuário
     * @param {Object} options - Opções da chave
     * @returns {Promise<Object>} - Dados da chave gerada
     */
    async generateKey(userId, options = {}) {
        try {
            // Verifica limites do usuário
            const userData = await this.db.getUserData(userId);
            if (!userData) {
                throw new Error('Usuário não encontrado');
            }
            
            // Verifica limite de chaves ativas
            const activeKeys = await this.db.listApiKeys(userData.id);
            const activeCount = activeKeys.filter(k => k.active).length;
            
            // Plano gratuito: 1 chave, Premium: ilimitado
            const maxKeys = userData.subscription?.status === 'active' ? 999 : 1;
            
            if (activeCount >= maxKeys) {
                throw new Error(`Limite de ${maxKeys} chaves ativas atingido`);
            }
            
            // Gera nova chave
            const key = KeyGenerator.generateSecureKey();
            const keyHash = KeyGenerator.hashKey(key);
            
            // Define limites baseados no plano
            // Plano gratuito: 50/dia, Premium: ilimitado
            const usageLimit = userData.subscription?.status === 'active' ? -1 : 50;
            
            // Calcula data de expiração
            const expiresAt = new Date();
            expiresAt.setDate(expiresAt.getDate() + 30); // 30 dias
            
            // Salva no banco
            const keyData = await this.db.saveApiKey(userData.id, {
                keyHash,
                expiresAt,
                usageLimit,
                ...options
            });
            
            return {
                key,
                expiresAt: keyData.expires_at,
                usageLimit: keyData.usage_limit,
                dailyLimit: userData.subscription?.status === 'active' ? null : 50
            };
            
        } catch (error) {
            logger.error('Erro ao gerar chave:', error);
            throw error;
        }
    }
    
    /**
     * Revoga uma chave API.
     * @param {string} userId - ID do usuário
     * @param {string} keyId - ID da chave
     * @returns {Promise<boolean>} - True se a chave foi revogada
     */
    async revokeKey(userId, keyId) {
        try {
            const userData = await this.db.getUserData(userId);
            if (!userData) {
                throw new Error('Usuário não encontrado');
            }
            
            const result = await this.db.revokeApiKey(userData.id, keyId);
            return !!result;
            
        } catch (error) {
            logger.error('Erro ao revogar chave:', error);
            throw error;
        }
    }
    
    /**
     * Lista todas as chaves do usuário.
     * @param {string} userId - ID do usuário
     * @returns {Promise<Array>} - Lista de chaves
     */
    async listKeys(userId) {
        try {
            const userData = await this.db.getUserData(userId);
            if (!userData) {
                throw new Error('Usuário não encontrado');
            }
            
            return await this.db.listApiKeys(userData.id);
            
        } catch (error) {
            logger.error('Erro ao listar chaves:', error);
            throw error;
        }
    }
    
    /**
     * Verifica se uma chave é válida e pode ser usada.
     * @param {string} key - Chave a ser verificada
     * @returns {Promise<Object>} - Status da chave e limites
     */
    async verifyKey(key) {
        try {
            // Primeiro valida o formato
            if (!KeyGenerator.validateKey(key)) {
                return {
                    valid: false,
                    reason: 'formato_invalido'
                };
            }
            
            // Busca chave no banco
            const keyData = await this.db.getKeyByHash(KeyGenerator.hashKey(key));
            if (!keyData) {
                return {
                    valid: false,
                    reason: 'chave_nao_encontrada'
                };
            }
            
            // Verifica se está ativa
            if (!keyData.active) {
                return {
                    valid: false,
                    reason: 'chave_inativa'
                };
            }
            
            // Verifica expiração
            if (new Date() > new Date(keyData.expires_at)) {
                return {
                    valid: false,
                    reason: 'chave_expirada'
                };
            }
            
            // Verifica plano do usuário
            const userData = await this.db.getUserData(keyData.user_id);
            const isPremium = userData.subscription?.status === 'active';
            
            // Se for premium, tem uso ilimitado
            if (isPremium) {
                return {
                    valid: true,
                    unlimited: true
                };
            }
            
            // Plano gratuito: verifica limite diário
            const today = new Date().toISOString().split('T')[0];
            const generationsToday = await this.db.getGenerationsCount(keyData.user_id, today);
            
            if (generationsToday >= 50) {
                return {
                    valid: false,
                    reason: 'limite_diario_excedido',
                    generationsToday,
                    dailyLimit: 50
                };
            }
            
            return {
                valid: true,
                unlimited: false,
                generationsToday,
                dailyLimit: 50,
                remainingToday: 50 - generationsToday
            };
            
        } catch (error) {
            logger.error('Erro ao verificar chave:', error);
            return {
                valid: false,
                reason: 'erro_interno'
            };
        }
    }
}

export default new KeyManager(); 