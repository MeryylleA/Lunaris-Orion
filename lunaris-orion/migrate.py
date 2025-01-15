import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuração do banco de dados
DB_CONFIG = {
    'host': 'postgresql-1ba07f7c-oea4da127.database.cloud.ovh.net',
    'port': '20184',
    'database': 'defaultdb',  # Usando o banco defaultdb existente
    'user': 'avnadmin',
    'password': 'RUF1g6KplPYxk0q7rse8',
    'sslmode': 'require'
}

def execute_migration(conn, migration_file):
    """Executa um arquivo de migração específico"""
    try:
        with open(os.path.join('src/database/migrations', migration_file), 'r', encoding='utf-8') as f:
            sql = f.read()
            
        with conn.cursor() as cur:
            logger.info(f"Executando migração: {migration_file}")
            cur.execute(sql)
            logger.info(f"Migração {migration_file} executada com sucesso!")
            
    except Exception as e:
        logger.error(f"Erro ao executar migração {migration_file}: {str(e)}")
        raise

def create_migrations_table(conn):
    """Cria a tabela de controle de migrações se não existir"""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

def get_executed_migrations(conn):
    """Retorna lista de migrações já executadas"""
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations ORDER BY version;")
        return [row[0] for row in cur.fetchall()]

def main():
    try:
        # Conecta ao banco defaultdb
        logger.info("Conectando ao banco de dados 'defaultdb'...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Testa a conexão
        with conn.cursor() as cur:
            cur.execute('SELECT version();')
            version = cur.fetchone()
            logger.info(f"Conectado ao PostgreSQL versão: {version[0]}")
        
        # Cria tabela de controle de migrações
        create_migrations_table(conn)
        
        # Lista migrações já executadas
        executed_migrations = get_executed_migrations(conn)
        logger.info(f"Migrações já executadas: {executed_migrations}")
        
        # Lista todos os arquivos de migração
        migration_files = sorted([
            f for f in os.listdir('src/database/migrations')
            if f.endswith('.sql')
        ])
        
        # Executa migrações pendentes
        for migration_file in migration_files:
            if migration_file not in executed_migrations:
                try:
                    execute_migration(conn, migration_file)
                    
                    # Registra migração executada
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO schema_migrations (version) VALUES (%s)",
                            (migration_file,)
                        )
                    
                except Exception as e:
                    logger.error(f"Falha ao executar migração {migration_file}")
                    raise
        
        logger.info("Todas as migrações foram executadas com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante o processo de migração: {str(e)}")
        sys.exit(1)
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 