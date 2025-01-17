const fs = require('fs');
const path = require('path');

// Diretórios
const sourceDir = '/home/meryy/Desktop/lunaris-orion/src/commands';
const targetDir = '/home/meryy/Desktop/lunaris-orion/discord_bot/src/commands';

// Função para converter o arquivo
function convertFile(sourceFile, targetFile) {
    console.log(`Lendo arquivo: ${sourceFile}`);
    const content = fs.readFileSync(sourceFile, 'utf8');
    
    // Converte require('discord.js') para CommonJS
    let newContent = content.replace(
        /import\s+{([^}]+)}\s+from\s+['"]discord\.js['"]/g,
        'const {$1} = require(\'discord.js\')'
    );
    
    // Converte outros imports para CommonJS
    newContent = newContent.replace(
        /import\s+{([^}]+)}\s+from\s+['"]([^'"]+)['"]/g,
        'const {$1} = require(\'$2\')'
    );
    
    // Converte export default para module.exports
    newContent = newContent.replace(
        /export\s+default\s+{/g,
        'module.exports = {'
    );
    
    // Adiciona extensão .cjs ao arquivo
    const targetFileCjs = targetFile.replace('.js', '.cjs');
    
    console.log(`Salvando arquivo: ${targetFileCjs}`);
    // Salva o arquivo convertido
    fs.writeFileSync(targetFileCjs, newContent);
    console.log(`Convertido: ${path.basename(sourceFile)} -> ${path.basename(targetFileCjs)}`);
}

// Cria o diretório de destino se não existir
if (!fs.existsSync(targetDir)) {
    console.log(`Criando diretório: ${targetDir}`);
    fs.mkdirSync(targetDir, { recursive: true });
}

// Lista e converte os arquivos
console.log(`Lendo diretório: ${sourceDir}`);
fs.readdirSync(sourceDir).forEach(file => {
    if (file.endsWith('.js')) {
        const sourceFile = path.join(sourceDir, file);
        const targetFile = path.join(targetDir, file);
        convertFile(sourceFile, targetFile);
    }
});

console.log('Migração concluída!');
