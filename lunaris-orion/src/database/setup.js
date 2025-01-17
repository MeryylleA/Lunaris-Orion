const { Pool } = require('pg');
const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const os = require('os');
const { createLogger, format, transports } = require('winston');

// Configuração do logger
const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.Console(),
        new transports.File({ filename: 'logs/database.log' })
    ]
});

const execAsync = promisify(exec);

class DatabaseSetup {
    constructor() {
        this.dbName = 'lunaris_orion';
        this.dbUser = 'lunaris_app';
        this.dbPassword = this.generateSecurePassword();
        this.dbPort = 5432;
    }

    generateSecurePassword(length = 32) {
        const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*';
        return Array.from(crypto.randomBytes(length))
            .map(byte => charset[byte % charset.length])
            .join('');
    }

    async isPostgresInstalled() {
        try {
            await execAsync('psql --version');
            return true;
        } catch {
            return false;
        }
    }

    async installPostgres() {
        logger.info('Instalando PostgreSQL...');

        const platform = os.platform();
        try {
            if (platform === 'linux') {
                // Instala no Ubuntu/Debian
                await execAsync('sudo apt-get update');
                await execAsync('sudo apt-get install -y postgresql postgresql-contrib libpq-dev');
                
                // Inicia o serviço
                await execAsync('sudo systemctl start postgresql');
                await execAsync('sudo systemctl enable postgresql');
            } else if (platform === 'win32') {
                logger.error('Por favor, instale o PostgreSQL manualmente no Windows: https://www.postgresql.org/download/windows/');
                throw new Error('Instalação manual necessária no Windows');
            } else {
                throw new Error(`Sistema operacional não suportado: ${platform}`);
            }
        } catch (error) {
            logger.error('Erro ao instalar PostgreSQL:', error);
            throw error;
        }
    }

    async createDbAndUser() {
        logger.info('Configurando banco de dados e usuário...');

        const commands = [
            `CREATE USER ${this.dbUser} WITH PASSWORD '${this.dbPassword}';`,
            `CREATE DATABASE ${this.dbName} OWNER ${this.dbUser};`,
            `GRANT ALL PRIVILEGES ON DATABASE ${this.dbName} TO ${this.dbUser};`
        ];

        try {
            for (const cmd of commands) {
                await execAsync(`sudo -u postgres psql -c "${cmd}"`);
            }
            logger.info('Banco de dados e usuário criados com sucesso!');
        } catch (error) {
            logger.error('Erro ao criar banco de dados:', error);
            throw error;
        }
    }

    async optimizePostgres() {
        logger.info('Otimizando configurações do PostgreSQL...');

        // Detecta memória total do sistema
        const totalMem = os.totalmem();
        const totalMemGb = totalMem / (1024 * 1024 * 1024);

        const optimizations = [
            `shared_buffers = '${Math.floor(totalMemGb * 0.25)}GB'`,
            `effective_cache_size = '${Math.floor(totalMemGb * 0.5)}GB'`,
            `work_mem = '64MB'`,
            `maintenance_work_mem = '256MB'`,
            `max_connections = 100`,
            `wal_buffers = '16MB'`,
            `checkpoint_completion_target = 0.9`,
            `random_page_cost = 1.1`,
            `effective_io_concurrency = 200`,
            `synchronous_commit = off`
        ];

        try {
            const confFile = '/etc/postgresql/*/main/postgresql.conf';
            for (const opt of optimizations) {
                const [param] = opt.split('=');
                await execAsync(`sudo sed -i 's/^#?${param}.*/${opt}/' ${confFile}`);
            }

            await execAsync('sudo systemctl restart postgresql');
            logger.info('Configurações otimizadas com sucesso!');
        } catch (error) {
            logger.error('Erro ao otimizar PostgreSQL:', error);
            throw error;
        }
    }

    async setupDatabase() {
        try {
            // Verifica se PostgreSQL está instalado
            if (!await this.isPostgresInstalled()) {
                await this.installPostgres();
                // Aguarda serviço iniciar
                await new Promise(resolve => setTimeout(resolve, 5000));
            }

            // Cria banco e usuário
            await this.createDbAndUser();

            // Otimiza configurações
            await this.optimizePostgres();

            // Conecta ao banco
            const pool = new Pool({
                database: this.dbName,
                user: this.dbUser,
                password: this.dbPassword,
                host: 'localhost',
                port: this.dbPort
            });

            // Executa migrações
            const migrationsPath = path.join(__dirname, 'migrations');
            const migrations = await fs.readdir(migrationsPath);
            
            for (const migration of migrations.sort()) {
                if (migration.endsWith('.sql')) {
                    logger.info(`Executando migração: ${migration}`);
                    const sql = await fs.readFile(path.join(migrationsPath, migration), 'utf8');
                    await pool.query(sql);
                    logger.info(`Migração ${migration} concluída`);
                }
            }

            // Salva credenciais
            const envContent = `
# Database Configuration
DB_NAME=${this.dbName}
DB_USER=${this.dbUser}
DB_PASSWORD=${this.dbPassword}
DB_HOST=localhost
DB_PORT=${this.dbPort}
`;

            await fs.appendFile('.env', envContent);
            logger.info('Configuração do banco de dados concluída com sucesso!');
            logger.info('Credenciais salvas em .env');

            await pool.end();

        } catch (error) {
            logger.error('Erro durante a configuração:', error);
            throw error;
        }
    }
}

// Se executado diretamente
if (require.main === module) {
    const setup = new DatabaseSetup();
    setup.setupDatabase().catch(error => {
        logger.error('Erro na execução do script:', error);
        process.exit(1);
    });
}

module.exports = DatabaseSetup; 