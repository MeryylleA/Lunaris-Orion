-- Adiciona a coluna discord_id se ela não existir
DO $$ 
BEGIN 
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'users' 
        AND column_name = 'discord_id'
    ) THEN
        ALTER TABLE users ADD COLUMN discord_id VARCHAR(255);
        ALTER TABLE users ADD CONSTRAINT users_discord_id_unique UNIQUE (discord_id);
    END IF;
END $$; 