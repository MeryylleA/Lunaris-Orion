import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def test_connection():
    try:
        # Parâmetros de conexão
        conn_params = {
            'host': 'postgresql-1ba07f7c-oea4da127.database.cloud.ovh.net',
            'port': '20184',
            'database': 'lunaris',
            'user': 'avnadmin',
            'password': 'RUF1g6KplPYxk0q7rse8',
            'sslmode': 'require'
        }
        
        # Tenta estabelecer a conexão
        print("Tentando conectar ao banco de dados...")
        conn = psycopg2.connect(**conn_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Testa a conexão executando uma query simples
        with conn.cursor() as cur:
            cur.execute('SELECT version();')
            version = cur.fetchone()
            print(f"Conexão estabelecida com sucesso!\nVersão do PostgreSQL: {version[0]}")
            
            # Lista as tabelas existentes
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cur.fetchall()
            print("\nTabelas existentes:")
            for table in tables:
                print(f"- {table[0]}")
        
        return True
        
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_connection() 