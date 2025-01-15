const { EmbedBuilder, ActionRowBuilder } = require('discord.js');
const keyCommand = require('../../src/commands/key');

describe('Comando Key', () => {
    let mockInteraction;
    let mockClient;

    beforeEach(() => {
        // Mock da interação
        mockInteraction = {
            user: {
                id: '123456789',
                tag: 'TestUser#0000',
                send: jest.fn()
            },
            reply: jest.fn(),
            followUp: jest.fn(),
            options: {
                getSubcommand: jest.fn()
            }
        };

        // Mock do cliente
        mockClient = {
            pool: global.testPool,
            getUserApiKey: jest.fn()
        };
    });

    describe('Subcomando info', () => {
        beforeEach(() => {
            mockInteraction.options.getSubcommand.mockReturnValue('info');
        });

        test('deve mostrar informações quando não há chave', async () => {
            // Mock: sem chave
            mockClient.getUserApiKey.mockResolvedValue(null);

            // Executa comando
            await keyCommand.execute(mockInteraction, mockClient);

            // Verifica resposta
            expect(mockInteraction.reply).toHaveBeenCalled();
            const replyCall = mockInteraction.reply.mock.calls[0][0];
            expect(replyCall.embeds[0]).toBeInstanceOf(EmbedBuilder);
            expect(replyCall.embeds[0].data.fields).toEqual(
                expect.arrayContaining([
                    expect.objectContaining({
                        name: 'Status',
                        value: '❌ Não gerada'
                    })
                ])
            );
        });

        test('deve mostrar informações da chave ativa', async () => {
            // Mock: chave existente
            mockClient.getUserApiKey.mockResolvedValue({
                created_at: new Date(),
                key_hash: 'abc123'
            });

            // Executa comando
            await keyCommand.execute(mockInteraction, mockClient);

            // Verifica resposta
            expect(mockInteraction.reply).toHaveBeenCalled();
            const replyCall = mockInteraction.reply.mock.calls[0][0];
            expect(replyCall.embeds[0].data.fields).toEqual(
                expect.arrayContaining([
                    expect.objectContaining({
                        name: 'Status',
                        value: '✅ Ativa'
                    })
                ])
            );
            expect(replyCall.components[0]).toBeInstanceOf(ActionRowBuilder);
        });
    });

    describe('Subcomando gerar', () => {
        beforeEach(() => {
            mockInteraction.options.getSubcommand.mockReturnValue('gerar');
        });

        test('não deve gerar nova chave se já existe uma', async () => {
            // Mock: chave existente
            mockClient.getUserApiKey.mockResolvedValue({
                key_hash: 'abc123'
            });

            // Executa comando
            await keyCommand.execute(mockInteraction, mockClient);

            // Verifica resposta
            expect(mockInteraction.reply).toHaveBeenCalledWith({
                content: 'Você já possui uma chave API ativa. Use `/key revogar` primeiro se deseja gerar uma nova.',
                ephemeral: true
            });
        });

        test('deve gerar e enviar nova chave', async () => {
            // Mock: sem chave existente
            mockClient.getUserApiKey.mockResolvedValue(null);
            mockClient.pool.query = jest.fn().mockResolvedValue({ rows: [] });

            // Executa comando
            await keyCommand.execute(mockInteraction, mockClient);

            // Verifica envio da chave
            expect(mockInteraction.user.send).toHaveBeenCalled();
            expect(mockInteraction.reply).toHaveBeenCalledWith({
                content: 'Sua chave API foi gerada e enviada por mensagem privada.',
                ephemeral: true
            });
        });

        test('deve lidar com erro ao enviar DM', async () => {
            // Mock: sem chave existente
            mockClient.getUserApiKey.mockResolvedValue(null);
            mockClient.pool.query = jest.fn().mockResolvedValue({ rows: [] });
            mockInteraction.user.send.mockRejectedValue(new Error('DM bloqueada'));

            // Executa comando
            await keyCommand.execute(mockInteraction, mockClient);

            // Verifica resposta de erro
            expect(mockInteraction.reply).toHaveBeenCalledWith({
                content: expect.stringContaining('Não foi possível enviar sua chave por mensagem privada'),
                ephemeral: true
            });
        });
    });

    describe('Subcomando revogar', () => {
        beforeEach(() => {
            mockInteraction.options.getSubcommand.mockReturnValue('revogar');
        });

        test('deve revogar chave existente', async () => {
            // Mock: chave existente
            mockClient.getUserApiKey.mockResolvedValue({
                key_hash: 'abc123'
            });
            mockClient.pool.query = jest.fn().mockResolvedValue({ rows: [] });

            // Executa comando
            await keyCommand.execute(mockInteraction, mockClient);

            // Verifica revogação
            expect(mockClient.pool.query).toHaveBeenCalled();
            expect(mockInteraction.reply).toHaveBeenCalledWith({
                content: 'Sua chave API foi revogada com sucesso.',
                ephemeral: true
            });
        });

        test('não deve permitir revogar se não há chave', async () => {
            // Mock: sem chave
            mockClient.getUserApiKey.mockResolvedValue(null);

            // Executa comando
            await keyCommand.execute(mockInteraction, mockClient);

            // Verifica resposta
            expect(mockInteraction.reply).toHaveBeenCalledWith({
                content: 'Você não possui uma chave API ativa para revogar.',
                ephemeral: true
            });
        });
    });

    test('deve tratar erros adequadamente', async () => {
        // Simula erro
        mockClient.getUserApiKey.mockRejectedValue(new Error('Erro de teste'));

        // Executa comando
        await keyCommand.execute(mockInteraction, mockClient);

        // Verifica tratamento de erro
        expect(mockInteraction.reply).toHaveBeenCalledWith({
            content: 'Erro ao processar comando. Tente novamente mais tarde.',
            ephemeral: true
        });
    });
}); 