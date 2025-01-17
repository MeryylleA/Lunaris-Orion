import psycopg2
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
    'database': 'defaultdb',
    'user': 'avnadmin',
    'password': 'RUF1g6KplPYxk0q7rse8',
    'sslmode': 'require'
}

def list_tables():
    try:
        # Conecta ao banco de dados
        conn = psycopg2.connect(**DB_CONFIG)
        
        with conn.cursor() as cur:
            # Lista todas as tabelas
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            
            tables = cur.fetchall()
            
            print("\nTabelas no banco de dados:")
            print("-" * 30)
            for table in tables:
                # Para cada tabela, conta o número de registros
                cur.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cur.fetchone()[0]
                print(f"- {table[0]} ({count} registros)")
                
                # Mostra a estrutura da tabela
                cur.execute(f"""
                    SELECT column_name, data_type, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_name = '{table[0]}'
                    ORDER BY ordinal_position;
                """)
                
                columns = cur.fetchall()
                for col in columns:
                    col_name, col_type, max_length = col
                    if max_length:
                        print(f"  └─ {col_name}: {col_type}({max_length})")
                    else:
                        print(f"  └─ {col_name}: {col_type}")
                print()
                
    except Exception as e:
        logger.error(f"Erro ao listar tabelas: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    list_tables() 