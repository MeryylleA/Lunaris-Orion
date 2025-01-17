const { EmbedBuilder } = require('discord.js');
const userCommand = require('../../src/commands/user');

describe('Comando User', () => {
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
            getSubscriptionStatus: jest.fn(),
            getUserApiKey: jest.fn()
        };
    });

    test('deve criar novo usuário se não existir', async () => {
        // Configura mocks
        mockClient.pool.query = jest.fn()
            .mockResolvedValueOnce({ rows: [] }) // Primeira chamada: usuário não existe
            .mockResolvedValueOnce({ rows: [{ id: 1 }] }); // Segunda chamada: criação do usuário

        // Executa comando
        await userCommand.execute(mockInteraction, mockClient);

        // Verifica se o usuário foi criado
        expect(mockClient.pool.query).toHaveBeenCalledTimes(2);
        expect(mockInteraction.reply).toHaveBeenCalled();
    });

    test('deve mostrar informações do usuário existente', async () => {
        // Mock de dados do usuário
        const mockUserData = {
            rows: [{
                id: 1,
                discord_id: '123456789',
                status: 'active',
                tier: 'premium',
                expires_at: new Date(),
                key_hash: 'abc123'
            }]
        };

        // Configura mocks
        mockClient.pool.query = jest.fn()
            .mockResolvedValueOnce(mockUserData)
            .mockResolvedValueOnce({ rows: [{ count: '25' }] });

        // Executa comando
        await userCommand.execute(mockInteraction, mockClient);

        // Verifica se as informações foram exibidas
        expect(mockInteraction.reply).toHaveBeenCalled();
        const replyCall = mockInteraction.reply.mock.calls[0][0];
        expect(replyCall.embeds[0]).toBeInstanceOf(EmbedBuilder);
        expect(replyCall.ephemeral).toBe(true);
    });

    test('deve mostrar uso diário para usuários gratuitos', async () => {
        // Mock de dados do usuário gratuito
        const mockUserData = {
            rows: [{
                id: 1,
                discord_id: '123456789',
                status: null,
                tier: null
            }]
        };

        // Configura mocks
        mockClient.pool.query = jest.fn()
            .mockResolvedValueOnce(mockUserData)
            .mockResolvedValueOnce({ rows: [{ count: '25' }] });

        // Executa comando
        await userCommand.execute(mockInteraction, mockClient);

        // Verifica se o uso diário foi exibido
        expect(mockInteraction.reply).toHaveBeenCalled();
        const replyCall = mockInteraction.reply.mock.calls[0][0];
        expect(replyCall.embeds[0].data.fields).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    name: 'Gerações Hoje',
                    value: '25/50'
                })
            ])
        );
    });

    test('deve tratar erros adequadamente', async () => {
        // Simula erro
        mockClient.pool.query = jest.fn().mockRejectedValue(new Error('Erro de teste'));

        // Executa comando
        await userCommand.execute(mockInteraction, mockClient);

        // Verifica tratamento de erro
        expect(mockInteraction.reply).toHaveBeenCalledWith({
            content: 'Erro ao buscar informações. Tente novamente mais tarde.',
            ephemeral: true
        });
    });
}); 