const fs = require('fs');
const path = require('path');

// Mapeamento de arquivos
const filesToMigrate = [
    {
        source: '/home/meryy/Desktop/lunaris-orion/src/bot.js',
        target: '/home/meryy/Desktop/lunaris-orion/discord_bot/src/bot.cjs'
    },
    {
        source: '/home/meryy/Desktop/lunaris-orion/src/webhooks/stripe.js',
        target: '/home/meryy/Desktop/lunaris-orion/discord_bot/src/webhooks/stripe.cjs'
    },
    {
        source: '/home/meryy/Desktop/lunaris-orion/src/config/api.js',
        target: '/home/meryy/Desktop/lunaris-orion/discord_bot/src/config/api.cjs'
    },
    {
        source: '/home/meryy/Desktop/lunaris-orion/src/config/stripe.js',
        target: '/home/meryy/Desktop/lunaris-orion/discord_bot/src/config/stripe.cjs'
    }
];

// Função para converter o arquivo
function convertFile(sourceFile, targetFile) {
    console.log(`\nProcessando: ${path.basename(sourceFile)}`);
    console.log(`Lendo arquivo: ${sourceFile}`);
    
    const content = fs.readFileSync(sourceFile, 'utf8');
    
    // Converte imports para CommonJS
    let newContent = content
        // Converte import padrão
        .replace(
            /import\s+(\w+)\s+from\s+['"]([^'"]+)['"]/g,
            'const $1 = require(\'$2\')'
        )
        // Converte import com desestruturação
        .replace(
            /import\s+{([^}]+)}\s+from\s+['"]([^'"]+)['"]/g,
            'const {$1} = require(\'$2\')'
        )
        // Converte export default
        .replace(
            /export\s+default\s+/g,
            'module.exports = '
        )
        // Converte export const
        .replace(
            /export\s+const\s+(\w+)/g,
            'const $1 = '
        );
    
    // Ajusta caminhos relativos
    newContent = newContent.replace(
        /require\(['"]\.\.?\/([^'"]+)['"]\)/g,
        (match, p1) => {
            // Ajusta o caminho relativo baseado na nova estrutura
            return `require('./${p1}')`;
        }
    );
    
    // Garante que o diretório de destino existe
    const targetDir = path.dirname(targetFile);
    if (!fs.existsSync(targetDir)) {
        console.log(`Criando diretório: ${targetDir}`);
        fs.mkdirSync(targetDir, { recursive: true });
    }
    
    console.log(`Salvando arquivo: ${targetFile}`);
    fs.writeFileSync(targetFile, newContent);
    console.log(`Convertido: ${path.basename(sourceFile)} -> ${path.basename(targetFile)}`);
}

// Processa cada arquivo
filesToMigrate.forEach(file => {
    try {
        convertFile(file.source, file.target);
    } catch (error) {
        console.error(`Erro ao processar ${file.source}:`, error);
    }
});

console.log('\nMigração do sistema concluída!');
