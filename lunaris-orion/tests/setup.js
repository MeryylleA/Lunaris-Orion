const { config } = require('dotenv');
const { Pool } = require('pg');

// Carrega variáveis de ambiente de teste
config({ path: '.env.test' });

// Mock global para o cliente Discord
global.mockDiscordClient = {
    user: {
        id: '123456789',
        tag: 'TestBot#0000'
    },
    commands: new Map(),
    guilds: new Map(),
    users: new Map()
};

// Configuração do pool de teste
global.testPool = new Pool({
    user: process.env.POSTGRES_USER,
    password: process.env.POSTGRES_PASSWORD,
    database: process.env.POSTGRES_TEST_DB,
    host: 'localhost'
});

// Limpa o banco de teste antes de cada teste
beforeEach(async () => {
    await testPool.query(`
        TRUNCATE TABLE users, subscriptions, api_keys, generations, checkout_sessions CASCADE;
    `);
});

// Fecha conexões após todos os testes
afterAll(async () => {
    await testPool.end();
}); 