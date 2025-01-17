import crypto from 'crypto';
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

class KeyGenerator {
    /**
     * Gera uma chave segura para a API.
     * @param {number} length - Comprimento da chave (default: 48 caracteres)
     * @returns {string} - Chave no formato LUN-XXXX-XXXX-XXXX
     */
    static generateSecureKey(length = 48) {
        try {
            // Gera bytes aleatórios
            const randomBytes = crypto.randomBytes(32);
            
            // Adiciona timestamp para unicidade
            const timestamp = Buffer.alloc(8);
            timestamp.writeBigInt64BE(BigInt(Date.now()));
            
            // Combina e gera hash
            const combined = Buffer.concat([randomBytes, timestamp]);
            const hash = crypto.createHash('sha256').update(combined).digest();
            
            // Codifica em base64 e remove caracteres especiais
            const encoded = hash.toString('base64url');
            const cleanKey = encoded.replace(/[^a-zA-Z0-9]/g, '');
            
            // Trunca para o tamanho desejado
            const truncated = cleanKey.slice(0, length);
            
            // Formata como LUN-XXXX-XXXX-XXXX
            const parts = [
                'LUN',
                truncated.slice(0, 4),
                truncated.slice(4, 8),
                truncated.slice(8, 12)
            ];
            
            return parts.join('-');
            
        } catch (error) {
            logger.error('Erro ao gerar chave:', error);
            throw error;
        }
    }
    
    /**
     * Valida o formato da chave.
     * @param {string} key - Chave a ser validada
     * @returns {boolean} - True se a chave tem formato válido
     */
    static validateKey(key) {
        if (!key) return false;
        
        try {
            // Verifica formato
            const parts = key.split('-');
            if (parts.length !== 4) return false;
            
            // Verifica prefixo
            if (parts[0] !== 'LUN') return false;
            
            // Verifica comprimento das partes
            if (!parts.slice(1).every(part => part.length === 4)) return false;
            
            // Verifica caracteres válidos
            const validChars = /^[0-9A-Za-z]+$/;
            if (!parts.slice(1).every(part => validChars.test(part))) return false;
            
            return true;
            
        } catch (error) {
            logger.error('Erro ao validar chave:', error);
            return false;
        }
    }
    
    /**
     * Gera hash da chave para armazenamento seguro.
     * @param {string} key - Chave a ser hasheada
     * @returns {string} - Hash da chave
     */
    static hashKey(key) {
        try {
            // Usa SHA-256 com salt
            const salt = crypto.randomBytes(16).toString('hex');
            const hash = crypto.createHash('sha256')
                .update(key + salt)
                .digest('hex');
            
            return `${salt}$${hash}`;
            
        } catch (error) {
            logger.error('Erro ao gerar hash da chave:', error);
            throw error;
        }
    }
    
    /**
     * Verifica se uma chave corresponde ao hash armazenado.
     * @param {string} key - Chave a ser verificada
     * @param {string} hashedKey - Hash armazenado
     * @returns {boolean} - True se a chave corresponde ao hash
     */
    static verifyKey(key, hashedKey) {
        try {
            const [salt, hash] = hashedKey.split('$');
            
            const checkHash = crypto.createHash('sha256')
                .update(key + salt)
                .digest('hex');
            
            return crypto.timingSafeEqual(
                Buffer.from(checkHash, 'hex'),
                Buffer.from(hash, 'hex')
            );
            
        } catch (error) {
            logger.error('Erro ao verificar chave:', error);
            return false;
        }
    }
}

export default KeyGenerator; 