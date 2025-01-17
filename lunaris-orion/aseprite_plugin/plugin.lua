-- Lunaris Orion Plugin
-- Versão 2.0.0

if not app.isUIAvailable then
    return
end

local dlg = nil

-- Constantes
local DAILY_LIMIT = 50

-- Verifica se é um novo dia
local function is_new_day(last_reset)
    if not last_reset then return true end
    local current_date = os.date("*t")
    local last_date = os.date("*t", last_reset)
    return current_date.year ~= last_date.year or
           current_date.month ~= last_date.month or
           current_date.day ~= last_date.day
end

function init(plugin)
    -- Inicializa as preferências se não existirem
    if plugin.preferences.api_key == nil then
        plugin.preferences.api_key = ""
    end
    if plugin.preferences.resolution == nil then
        plugin.preferences.resolution = "32x32"
    end
    if plugin.preferences.style == nil then
        plugin.preferences.style = "pixel"
    end
    if plugin.preferences.daily_generations == nil then
        plugin.preferences.daily_generations = 0
    end
    if plugin.preferences.last_reset == nil then
        plugin.preferences.last_reset = os.time()
    end

    -- Reseta contador se for um novo dia
    if is_new_day(plugin.preferences.last_reset) then
        plugin.preferences.daily_generations = 0
        plugin.preferences.last_reset = os.time()
    end

    plugin:newCommand{
        id="LunarisOrion",
        title="Lunaris Orion",
        group="file_scripts",
        onclick=function()
            -- Fecha diálogo existente
            if dlg then
                dlg:close()
            end

            -- Cria diálogo
            dlg = Dialog {
                title="Lunaris Orion",
                onclose=function()
                    dlg = nil
                end
            }

            -- Seção de Configurações
            dlg:separator{ text="Configurações" }
            
            dlg:entry{
                id="api_key",
                label="Chave API:",
                text=plugin.preferences.api_key,
                onchange=function()
                    plugin.preferences.api_key = dlg.data.api_key
                end
            }
            
            dlg:combobox{
                id="resolution",
                label="Resolução:",
                option=plugin.preferences.resolution,
                options={"16x16", "32x32", "64x64", "128x128"},
                onchange=function()
                    plugin.preferences.resolution = dlg.data.resolution
                end
            }
            
            dlg:combobox{
                id="style",
                label="Estilo:",
                option=plugin.preferences.style,
                options={"pixel", "retro", "moderno", "minimalista"},
                onchange=function()
                    plugin.preferences.style = dlg.data.style
                end
            }

            -- Seção de Status
            dlg:separator{ text="Status" }
            
            if plugin.preferences.api_key == "" then
                dlg:label{
                    id="status",
                    text="Use o comando /key no Discord para gerar sua chave API"
                }
            else
                dlg:label{
                    id="status",
                    text=string.format(
                        "Gerações hoje: %d/%d", 
                        plugin.preferences.daily_generations,
                        DAILY_LIMIT
                    )
                }
                
                dlg:button{
                    id="premium",
                    text="🌟 Assinar Premium",
                    onclick=function()
                        app.alert{
                            title="Plano Premium",
                            text={
                                "Para assinar o plano premium:",
                                "1. Use o comando /premium no Discord",
                                "2. Escolha o método de pagamento",
                                "3. Após o pagamento, use /key para",
                                "   gerar sua nova chave premium",
                                "",
                                "Benefícios:",
                                "✓ Gerações ilimitadas",
                                "✓ Acesso a recursos beta",
                                "✓ Prioridade na fila",
                                "✓ Suporte premium"
                            }
                        }
                    end
                }
            end

            -- Seção de Geração
            dlg:separator{ text="Geração de Pixel Art" }
            
            dlg:entry{
                id="prompt",
                label="Prompt:",
                text="",
                focus=true
            }
            
            dlg:button{
                id="generate",
                text="Gerar",
                onclick=function()
                    -- Verifica chave API
                    if plugin.preferences.api_key == "" then
                        app.alert{
                            title="Chave API Necessária",
                            text={
                                "Para gerar pixel art, você precisa de uma chave API.",
                                "",
                                "Como obter:",
                                "1. Entre no nosso servidor Discord",
                                "2. Use o comando /key",
                                "3. Cole a chave aqui no plugin"
                            }
                        }
                        return
                    end

                    -- Verifica limite diário
                    if plugin.preferences.daily_generations >= DAILY_LIMIT then
                        app.alert{
                            title="Limite Diário Atingido",
                            text={
                                string.format("Você atingiu o limite de %d gerações hoje.", DAILY_LIMIT),
                                "",
                                "Para continuar gerando:",
                                "1. Use o comando /premium no Discord",
                                "2. Após assinar, use /key para",
                                "   gerar sua nova chave premium"
                            }
                        }
                        return
                    end

                    local prompt = dlg.data.prompt
                    if prompt and prompt ~= "" then
                        -- Incrementa contador
                        plugin.preferences.daily_generations = plugin.preferences.daily_generations + 1
                        
                        -- Atualiza status
                        dlg:modify{
                            id="status",
                            text=string.format(
                                "Gerações hoje: %d/%d", 
                                plugin.preferences.daily_generations,
                                DAILY_LIMIT
                            )
                        }

                        -- Mostra mensagem de geração
                        app.alert(string.format(
                            "Gerando: %s\nResolução: %s\nEstilo: %s\nGerações restantes hoje: %d",
                            prompt,
                            plugin.preferences.resolution,
                            plugin.preferences.style,
                            DAILY_LIMIT - plugin.preferences.daily_generations
                        ))
                    else
                        app.alert("Digite um prompt")
                    end
                end
            }

            -- Mostra diálogo
            dlg:show{ wait=false }
        end
    }
end

function exit(plugin)
    if dlg then
        dlg:close()
        dlg = nil
    end
end

return {
    init = init,
    exit = exit
}