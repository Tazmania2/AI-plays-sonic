-- BizHawk Lua Bridge for AI Training - FIXED VERSION
-- Fixed file-based communication bridge with proper input handling

-- Configuration
-- Build communication directory path
local DEFAULT_BASE_DIR = os.getenv("BIZHAWK_COMM_BASE") or (os.getenv("PWD") or ".")

-- Optional instance ID to support multiple parallel emulators
local INSTANCE_ID = tonumber(os.getenv("BIZHAWK_INSTANCE_ID") or "0")

local COMM_DIR = string.format("%s/bizhawk_comm_%d", DEFAULT_BASE_DIR, INSTANCE_ID)
local REQUEST_FILE = COMM_DIR .. "/request.txt"
local RESPONSE_FILE = COMM_DIR .. "/response.txt"
local STATUS_FILE = COMM_DIR .. "/status.txt"

-- Genesis controller state - PERSISTENT STATE
local genesis_state = {
    up = false,
    down = false,
    left = false,
    right = false,
    a = false,
    b = false,
    c = false,
    start = false
}

-- Input mapping for Genesis controller
local INPUT_MAP = {
    ["UP"] = "P1 Up",
    ["DOWN"] = "P1 Down", 
    ["LEFT"] = "P1 Left",
    ["RIGHT"] = "P1 Right",
    ["A"] = "P1 A",
    ["B"] = "P1 B", 
    ["C"] = "P1 C",
    ["START"] = "P1 Start"
}

-- Utility functions
local function log(message)
    print(string.format("[BizHawk Bridge] %s", message))
end

local function create_directory(path)
    local success = os.execute("mkdir \"" .. path .. "\" 2>nul")
    return success
end

local function write_file(path, content)
    local file = io.open(path, "w")
    if file then
        file:write(content)
        file:close()
        return true
    end
    return false
end

local function read_file(path)
    local file = io.open(path, "r")
    if file then
        local content = file:read("*all")
        file:close()
        return content
    end
    return nil
end

local function delete_file(path)
    os.remove(path)
end

-- FIXED: Proper input state management
local function set_input_state(input_name, state)
    if INPUT_MAP[input_name] then
        -- Update our persistent state
        genesis_state[string.lower(input_name)] = state
        log(string.format("Set %s = %s", input_name, tostring(state)))
    else
        log(string.format("Unknown input: %s", input_name))
    end
end

-- FIXED: Apply all current inputs every frame
local function apply_inputs()
    local btn = {}
    for input_name, state in pairs(genesis_state) do
        if state then
            local bizhawk_input = INPUT_MAP[string.upper(input_name)]
            if bizhawk_input then
                btn[bizhawk_input] = true
            end
        end
    end
    
    -- Apply the inputs
    if next(btn) then
        joypad.set(btn)
    end
end

local function get_game_state()
    local state = {}
    
    -- Try to read basic memory values with error handling
    local success, value
    
    -- Player position (basic addresses)
    success, value = pcall(function() return memory.readword(0xFFB000) end)
    state.sonic_x = success and value or 0
    
    success, value = pcall(function() return memory.readword(0xFFB004) end)
    state.sonic_y = success and value or 0
    
    -- Player state
    success, value = pcall(function() return memory.readbyte(0xFFB020) end)
    state.sonic_rings = success and value or 0
    
    success, value = pcall(function() return memory.readbyte(0xFFB02A) end)
    state.sonic_lives = success and value or 0
    
    -- Level info
    success, value = pcall(function() return memory.readbyte(0xFFFE10) end)
    state.level_id = success and value or 0
    
    return state
end

local function encode_response(success, data, error_msg)
    local response = {}
    table.insert(response, "SUCCESS:" .. tostring(success))
    if data then
        table.insert(response, "DATA:" .. tostring(data))
    end
    if error_msg then
        table.insert(response, "ERROR:" .. tostring(error_msg))
    end
    return table.concat(response, "|")
end

local function parse_command(data)
    local parts = {}
    for part in data:gmatch("[^|]+") do
        local key, value = part:match("([^:]+):(.+)")
        if key and value then
            parts[key] = value
        end
    end
    return parts
end

local function handle_command(command_data)
    local command = parse_command(command_data)
    local action = command.ACTION
    
    if action == "GET_STATE" then
        local state = get_game_state()
        local state_str = string.format("x:%d,y:%d,rings:%d,lives:%d,level:%d", 
            state.sonic_x, state.sonic_y, state.sonic_rings, 
            state.sonic_lives, state.level_id)
        return encode_response(true, state_str)
        
    elseif action == "SET_INPUT" then
        local input = command.INPUT
        local state = command.STATE == "true"
        
        if input then
            set_input_state(input, state)
            return encode_response(true, string.format("%s:%s", input, tostring(state)))
        else
            return encode_response(false, nil, "Invalid input command")
        end
        
    elseif action == "SET_INPUTS" then
        local inputs = command.INPUTS
        if inputs then
            for input_part in inputs:gmatch("[^|]+") do
                local input_name, input_state = input_part:match("([^:]+):(.+)")
                if input_name and input_state then
                    set_input_state(input_name, input_state == "true")
                end
            end
            return encode_response(true, inputs)
        else
            return encode_response(false, nil, "Invalid inputs command")
        end
        
    elseif action == "RESET_INPUTS" then
        for input_name, _ in pairs(genesis_state) do
            set_input_state(string.upper(input_name), false)
        end
        return encode_response(true, "All inputs reset")
        
    elseif action == "PING" then
        return encode_response(true, "pong")
        
    else
        return encode_response(false, nil, string.format("Unknown command: %s", action or "nil"))
    end
end

-- Initialize bridge
log("BizHawk file-based bridge starting...")
log("Using communication directory: " .. COMM_DIR)

-- Create communication directory
if create_directory(COMM_DIR) then
    log("Communication directory created successfully")
else
    log("Could not create communication directory")
end

-- Create status file
if write_file(STATUS_FILE, "READY") then
    log("Status file created: READY")
else
    log("ERROR: Could not create status file")
end

log("Request file: " .. REQUEST_FILE)
log("Response file: " .. RESPONSE_FILE)
log("Status file: " .. STATUS_FILE)

log("File-based bridge ready. Waiting for requests...")

-- Main loop
while true do
    -- Apply current inputs every frame
    apply_inputs()
    
    -- Check for request file
    local request_content = read_file(REQUEST_FILE)
    
    if request_content and request_content ~= "" then
        -- Trim whitespace
        request_content = request_content:gsub("^%s+",""):gsub("%s+$","")

        -- Handle raw memory read command: "read <hex_addr> <size>"
        local addr_hex, size_str = request_content:match("^read%s+(%x+)%s+(%d+)$")
        if addr_hex and size_str then
            local addr = tonumber(addr_hex, 16)
            local size = tonumber(size_str)

            -- Read bytes from memory safely
            local bytes = {}
            for i = 0, size - 1 do
                local b = memory.read_u8(addr + i, "System Bus") or 0
                table.insert(bytes, string.char(b))
            end

            -- Concatenate into binary string
            local data_str = table.concat(bytes)

            -- Write binary response
            local file = io.open(RESPONSE_FILE, "wb")
            if file then
                file:write(data_str)
                file:close()
                log(string.format("Responded to memory read 0x%X (%d bytes)", addr, size))
            else
                log("ERROR: Could not write memory response file")
            end

            -- Clear request file
            delete_file(REQUEST_FILE)

            -- Continue loop without advancing again (we will advance at loop end)
            goto continue_main_loop
        end

        log("Received request: " .. request_content)
        
        -- Process command
        local response = handle_command(request_content)
        
        -- Write response
        if write_file(RESPONSE_FILE, response) then
            log("Sent response: " .. response)
        else
            log("ERROR: Could not write response file")
        end
        
        -- Clear request file
        delete_file(REQUEST_FILE)
    end

    ::continue_main_loop::
    -- Small delay to prevent excessive CPU usage
    emu.frameadvance()
end
