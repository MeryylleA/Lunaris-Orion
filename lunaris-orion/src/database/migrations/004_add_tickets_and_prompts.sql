-- Tabela de tickets
CREATE TABLE IF NOT EXISTS tickets (
    id SERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    feedback VARCHAR(50)
);

-- Tabela de prompts compartilhados
CREATE TABLE IF NOT EXISTS shared_prompts (
    id SERIAL PRIMARY KEY,
    generation_id UUID REFERENCES generations(id),
    user_id BIGINT REFERENCES users(id),
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de curtidas em prompts
CREATE TABLE IF NOT EXISTS prompt_likes (
    id SERIAL PRIMARY KEY,
    prompt_id INTEGER REFERENCES shared_prompts(id),
    user_id BIGINT REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(prompt_id, user_id)
);

-- Índices para melhor performance
CREATE INDEX IF NOT EXISTS idx_tickets_user_id ON tickets(user_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_shared_prompts_user_id ON shared_prompts(user_id);
CREATE INDEX IF NOT EXISTS idx_shared_prompts_category ON shared_prompts(category);
CREATE INDEX IF NOT EXISTS idx_prompt_likes_prompt_id ON prompt_likes(prompt_id);
CREATE INDEX IF NOT EXISTS idx_prompt_likes_user_id ON prompt_likes(user_id); 