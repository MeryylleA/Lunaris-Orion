const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

async function setupDatabase() {
    const pool = new Pool({
        host: process.env.DB_HOST,
        port: process.env.DB_PORT,
        database: process.env.DB_NAME,
        user: process.env.DB_USER,
        password: process.env.DB_PASSWORD,
        ssl: process.env.DB_SSL === 'true' ? {
            rejectUnauthorized: false
        } : false
    });

    let client;
    try {
        client = await pool.connect();

        // Lista os arquivos SQL em ordem
        const sqlFiles = [
            '001_initial_schema.sql',
            '002_add_discord_id.sql',
            '003_fix_users_table.sql',
            '004_add_stripe_fields.sql'
        ];

        for (const file of sqlFiles) {
            try {
                const sqlPath = path.join(__dirname, '..', 'sql', file);
                const sqlContent = fs.readFileSync(sqlPath, 'utf8');
                
                console.log(`Executando ${file}...`);
                await client.query(sqlContent);
                console.log(`${file} executado com sucesso!`);
            } catch (error) {
                console.error(`Erro ao executar ${file}:`, error);
                throw error;
            }
        }

        console.log('Database setup completed successfully!');
    } catch (error) {
        console.error('Error setting up database:', error);
    } finally {
        if (client) {
            client.release();
        }
        await pool.end();
    }
}

setupDatabase(); 