--[[
    Lunaris Orion - Extensão Aseprite
    Versão: 2.0.0
    Descrição: Extensão para geração e manipulação de pixel art usando IA
]]

local plugin = {
    name = "Lunaris Orion",
    version = "2.0.0",
    author = "Lunaris Team",
    api_url = "http://localhost:8000"
}

-- Configurações e estado
local config = {
    api_key = nil,
    api_url = plugin.api_url,
    default_resolution = "32x32",
    default_style = "retro",
    max_history = 10,
    auto_save = true,
    use_layers = true,
    preserve_palette = true,
    key_validated = false,
    subscription_status = nil,
    daily_generations = 0,
    daily_limit = 50  -- Limite do plano gratuito
}

-- Cache para histórico de gerações
local generation_history = {}

-- Utilitários
local function save_config()
    app.preferences.plugin_data[plugin.name] = {
        api_key = config.api_key,
        api_url = config.api_url,
        default_resolution = config.default_resolution,
        default_style = config.default_style,
        auto_save = config.auto_save,
        use_layers = config.use_layers,
        preserve_palette = config.preserve_palette,
        daily_generations = config.daily_generations
    }
end

local function load_config()
    local data = app.preferences.plugin_data[plugin.name] or {}
    config.api_key = data.api_key
    config.api_url = data.api_url or config.api_url
    config.default_resolution = data.default_resolution or config.default_resolution
    config.default_style = data.default_style or config.default_style
    config.auto_save = data.auto_save ~= false
    config.use_layers = data.use_layers ~= false
    config.preserve_palette = data.preserve_palette ~= false
    config.daily_generations = data.daily_generations or 0
end

-- Funções de API
local function validate_key(key)
    local curl = require("curl")
    local json = require("json")
    
    local response = curl.request{
        url = config.api_url .. "/validate_key",
        method = "POST",
        headers = {
            ["Content-Type"] = "application/json"
        },
        data = json.encode({
            key = key
        })
    }
    
    if response.status == 200 then
        local result = json.decode(response.body)
        config.subscription_status = result.subscription_status
        config.daily_limit = result.subscription_status == "premium" and math.huge or 50
        return result.valid, result.expires_at, result.subscription_status
    end
    
    return false, nil, nil
end

local function check_subscription_status()
    if not config.api_key then return false end
    
    local curl = require("curl")
    local json = require("json")
    
    local response = curl.request{
        url = config.api_url .. "/subscription/status",
        method = "GET",
        headers = {
            ["Authorization"] = "Bearer " .. config.api_key
        }
    }
    
    if response.status == 200 then
        local result = json.decode(response.body)
        config.subscription_status = result.status
        config.daily_limit = result.status == "premium" and math.huge or 50
        return true
    end
    
    return false
end

local function make_request(method, endpoint, data)
    local curl = require("curl")
    local json = require("json")
    
    -- Verifica se tem chave válida
    if not config.key_validated then
        app.alert("Chave de API não validada. Configure sua chave nas configurações.")
        return nil, "Chave não validada"
    end
    
    -- Verifica limite diário para plano gratuito
    if config.subscription_status ~= "premium" and config.daily_generations >= config.daily_limit then
        app.alert("Limite diário de gerações atingido. Faça upgrade para o plano premium para gerações ilimitadas!")
        return nil, "Limite diário excedido"
    end
    
    local headers = {
        ["Content-Type"] = "application/json",
        ["Authorization"] = "Bearer " .. (config.api_key or "")
    }
    
    local url = config.api_url .. endpoint
    local response = curl.request{
        url = url,
        method = method,
        headers = headers,
        data = data and json.encode(data) or nil
    }
    
    if response.status == 200 then
        if method == "POST" and endpoint == "/generate" then
            config.daily_generations = config.daily_generations + 1
            save_config()
        end
        return json.decode(response.body)
    elseif response.status == 401 then
        config.key_validated = false
        app.alert("Chave de API inválida ou expirada. Por favor, reconfigure.")
        return nil, "Chave inválida"
    elseif response.status == 429 then
        app.alert("Limite de requisições atingido. Aguarde alguns minutos.")
        return nil, "Limite de requisições"
    else
        return nil, response.status .. ": " .. response.body
    end
end

-- Funções de manipulação de imagem
local function extract_palette(sprite)
    local palette = {}
    if sprite.palettes and #sprite.palettes > 0 then
        local current_palette = sprite.palettes[1]
        for i = 0, #current_palette - 1 do
            local color = current_palette:getColor(i)
            table.insert(palette, {
                r = color.red,
                g = color.green,
                b = color.blue,
                a = color.alpha
            })
        end
    end
    return palette
end

local function apply_palette(sprite, palette)
    if not palette or #palette == 0 then return end
    
    local new_palette = Palette(#palette)
    for i, color in ipairs(palette) do
        new_palette:setColor(i-1, Color(color.r, color.g, color.b, color.a))
    end
    sprite:setPalette(new_palette)
end

local function create_layer_from_response(sprite, response)
    local layer = sprite:newLayer()
    layer.name = "Lunaris Generated"
    
    if response.seed then
        layer.name = layer.name .. " (Seed: " .. response.seed .. ")"
    end
    
    local cel = sprite:newCel(layer, app.activeFrame)
    
    -- Cria uma nova imagem a partir dos dados recebidos
    local image = Image(response.image)
    cel.image = image
    
    -- Aplica paleta se necessário
    if config.preserve_palette then
        apply_palette(sprite, extract_palette(sprite))
    end
    
    return layer
end

-- Funções principais
local function generate_pixel_art(prompt, options)
    -- Verifica status da assinatura
    if not check_subscription_status() then
        app.alert("Erro ao verificar status da assinatura")
        return nil
    end
    
    -- Prepara dados
    local data = {
        prompt = prompt,
        resolution = options.resolution or config.default_resolution,
        style = options.style or config.default_style,
        negative_prompt = options.negative_prompt,
        seed = options.seed,
        palette_size = options.palette_size
    }
    
    -- Extrai paleta atual se necessário
    if config.preserve_palette then
        data.palette = extract_palette(app.activeSprite)
    end
    
    -- Faz requisição
    local response, error = make_request("POST", "/generate", data)
    if not response then
        app.alert("Erro na geração: " .. (error or "Erro desconhecido"))
        return nil
    end
    
    -- Adiciona à história
    table.insert(generation_history, 1, {
        prompt = prompt,
        options = options,
        response = response,
        timestamp = os.time()
    })
    
    -- Mantém histórico limitado
    while #generation_history > config.max_history do
        table.remove(generation_history)
    end
    
    return response
end

-- Diálogos
local function show_key_dialog(force)
    if not force and config.key_validated then
        return true
    end
    
    local dlg = Dialog("Configurar Chave de API")
    
    dlg:entry{
        id = "key",
        label = "Chave de API:",
        text = config.api_key or "",
        focus = true
    }
    
    dlg:label{
        id = "info",
        label = "",
        text = "Obtenha sua chave usando o comando /key no Discord"
    }
    
    dlg:button{
        id = "validate",
        text = "Validar",
        focus = true
    }
    dlg:button{
        id = "cancel",
        text = "Cancelar"
    }
    
    dlg:show()
    
    local data = dlg.data
    if data.validate then
        local key = data.key:gsub("%s+", "")  -- Remove espaços
        
        if key == "" then
            app.alert("Por favor, insira uma chave válida")
            return false
        end
        
        -- Tenta validar a chave
        local valid, expires_at, subscription = validate_key(key)
        if valid then
            config.api_key = key
            config.key_validated = true
            config.subscription_status = subscription
            save_config()
            
            local msg = "Chave validada com sucesso!\n"
            if subscription == "premium" then
                msg = msg .. "Plano: Premium (Gerações ilimitadas)"
            else
                msg = msg .. string.format("Plano: Gratuito (%d/%d gerações hoje)", 
                    config.daily_generations, config.daily_limit)
            end
            
            if expires_at then
                msg = msg .. "\nExpira em: " .. expires_at
            end
            
            app.alert(msg)
            return true
        else
            app.alert("Chave inválida")
            return false
        end
    end
    
    return false
end

-- Diálogo de informações
local function show_info_dialog()
    local dlg = Dialog("Informações do Lunaris Orion")
    
    local status_text = "Não configurado"
    if config.key_validated then
        if config.subscription_status == "premium" then
            status_text = "Premium (Gerações ilimitadas)"
        else
            status_text = string.format("Gratuito (%d/%d gerações hoje)", 
                config.daily_generations, config.daily_limit)
        end
    end
    
    dlg:label{
        id = "version",
        text = "Versão: " .. plugin.version
    }
    dlg:label{
        id = "status",
        text = "Status: " .. status_text
    }
    dlg:label{
        id = "generations",
        text = string.format("Gerações hoje: %d", config.daily_generations)
    }
    
    if config.subscription_status ~= "premium" then
        dlg:button{
            id = "upgrade",
            text = "Fazer Upgrade",
            onclick = function()
                app.alert("Use o comando /plano no Discord para fazer upgrade para o plano premium!")
            end
        }
    end
    
    dlg:button{
        id = "close",
        text = "Fechar"
    }
    
    dlg:show()
end

-- Registra comandos
function init(plugin)
    load_config()
    
    plugin:newCommand{
        id = "lunarisGenerate",
        title = "Gerar Pixel Art...",
        group = "sprite_generation",
        onclick = function()
            if not config.key_validated and not show_key_dialog(true) then
                return
            end
            show_generation_dialog()
        end
    }
    
    plugin:newCommand{
        id = "lunarisSettings",
        title = "Configurações do Lunaris...",
        group = "sprite_generation",
        onclick = function()
            show_settings_dialog()
        end
    }
    
    plugin:newCommand{
        id = "lunarisInfo",
        title = "Informações do Lunaris...",
        group = "sprite_generation",
        onclick = function()
            show_info_dialog()
        end
    }
    
    -- Reseta contagem diária à meia-noite
    local function check_daily_reset()
        local current_date = os.date("*t")
        if current_date.hour == 0 and current_date.min == 0 then
            config.daily_generations = 0
            save_config()
        end
    end
    
    app.timer:interval(60000)  -- Verifica a cada minuto
    app.timer:connect(check_daily_reset)
end

return plugin 