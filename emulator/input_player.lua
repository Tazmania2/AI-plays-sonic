-- BizHawk Input Player for AI Training
-- File-based communication system for AI input control

-- Configuration
local DEFAULT_BASE_DIR = os.getenv("BIZHAWK_COMM_BASE") or (os.getenv("PWD") or ".")
local INSTANCE_ID = tonumber(os.getenv("BIZHAWK_INSTANCE_ID") or "0")

local COMM_DIR = string.format("%s/bizhawk_comm_%d", DEFAULT_BASE_DIR, INSTANCE_ID)
local INPUT_FILE = COMM_DIR .. "/ai_inputs.txt"
local LOG_FILE = COMM_DIR .. "/game_log.txt"
local STATUS_FILE = COMM_DIR .. "/status.txt"
local COMPLETION_FILE = COMM_DIR .. "/execution_complete.txt"

-- Genesis controller state
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
    ["START"] = "P1 Start",
    ["NOOP"] = nil,
    ["RESET"] = "RESET"
}

-- Game state tracking
local current_frame = 0
local last_log_frame = 0
local log_interval = 5  -- Log every 5 frames

-- Utility functions
local function log(message)
    print(string.format("[InputPlayer-%d] %s", INSTANCE_ID, message))
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

local function set_input_state(input_name, state)
    if input_name == "RESET" then
        -- Handle reset command
        log("Executing reset command")
        emu.softreset()
        return true
    end
    
    if INPUT_MAP[input_name] then
        local success, err = pcall(function()
            local btn = {}
            btn[INPUT_MAP[input_name]] = state
            joypad.set(btn)
        end)
        
        if success then
            genesis_state[string.lower(input_name)] = state
            log(string.format("Set %s = %s", input_name, tostring(state)))
            return true
        else
            log(string.format("Failed to set input %s: %s", input_name, tostring(err)))
            return false
        end
    else
        log(string.format("Unknown input: %s", input_name))
        return false
    end
end

local function reset_all_inputs()
    for input_name, _ in pairs(genesis_state) do
        set_input_state(string.upper(input_name), false)
    end
    log("All inputs reset")
end

local function get_game_state()
    local state = {}
    
    -- Try to read memory values with error handling
    local success, value
    
    -- Player position (Sonic 1 addresses)
    success, value = pcall(function() return memory.readword(0xFFD030) end)
    state.x = success and value or 0
    
    success, value = pcall(function() return memory.readword(0xFFD038) end)
    state.y = success and value or 0
    
    -- Player state
    success, value = pcall(function() return memory.readword(0xFFE002) end)
    state.rings = success and value or 0
    
    success, value = pcall(function() return memory.readword(0xFFE004) end)
    state.lives = success and value or 0
    
    success, value = pcall(function() return memory.readdword(0xFFE000) end)
    state.score = success and value or 0
    
    -- Level info
    success, value = pcall(function() return memory.readword(0xFFE012) end)
    state.zone = success and value or 0
    
    success, value = pcall(function() return memory.readword(0xFFE014) end)
    state.act = success and value or 0
    
    success, value = pcall(function() return memory.readword(0xFFE010) end)
    state.timer = success and value or 0
    
    -- Player status
    success, value = pcall(function() return memory.readword(0xFFE00E) end)
    state.invincibility = success and value or 0
    
    success, value = pcall(function() return memory.readword(0xFFD044) end)
    state.status = success and value or 0
    
    -- Current frame
    state.frame = current_frame
    
    return state
end

local function log_game_state()
    if current_frame - last_log_frame >= log_interval then
        local state = get_game_state()
        local state_json = string.format(
            '{"frame":%d,"x":%d,"y":%d,"rings":%d,"lives":%d,"score":%d,"zone":%d,"act":%d,"timer":%d,"invincibility":%d,"status":%d}',
            state.frame, state.x, state.y, state.rings, state.lives, state.score,
            state.zone, state.act, state.timer, state.invincibility, state.status
        )
        
        -- Append to log file
        local file = io.open(LOG_FILE, "a")
        if file then
            file:write(state_json .. "\n")
            file:close()
        end
        
        last_log_frame = current_frame
    end
end

local function read_input_sequence()
    local content = read_file(INPUT_FILE)
    if not content or content == "" then
        return {}
    end
    
    local inputs = {}
    for line in content:gmatch("[^\r\n]+") do
        local frame_str, action = line:match("^(%d+):(.+)$")
        if frame_str and action then
            local frame = tonumber(frame_str)
            table.insert(inputs, {frame = frame, action = action})
        end
    end
    
    return inputs
end

local function execute_input_sequence(inputs)
    if #inputs == 0 then
        return
    end
    
    log(string.format("Executing %d inputs", #inputs))
    
    -- Reset all inputs first
    reset_all_inputs()
    
    local current_input_index = 1
    local max_frames = 300  -- Maximum frames to execute
    
    for frame = 0, max_frames do
        current_frame = frame
        
        -- Check if we have inputs for this frame
        while current_input_index <= #inputs and inputs[current_input_index].frame <= frame do
            local input = inputs[current_input_index]
            set_input_state(input.action, true)
            current_input_index = current_input_index + 1
        end
        
        -- Log game state periodically
        log_game_state()
        
        -- Advance frame
        emu.frameadvance()
        
        -- Check if we've processed all inputs
        if current_input_index > #inputs then
            break
        end
    end
    
    -- Reset all inputs at the end
    reset_all_inputs()
    
    -- Mark execution as complete
    write_file(COMPLETION_FILE, "COMPLETE")
    
    log("Input sequence execution completed")
end

-- Initialize
log("Input Player starting...")
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

log("Input file: " .. INPUT_FILE)
log("Log file: " .. LOG_FILE)
log("Status file: " .. STATUS_FILE)
log("Completion file: " .. COMPLETION_FILE)

log("Input Player ready. Waiting for input files...")

-- Main loop
while true do
    -- Check for input file
    local inputs = read_input_sequence()
    
    if #inputs > 0 then
        log(string.format("Found %d inputs to execute", #inputs))
        
        -- Clear input file
        delete_file(INPUT_FILE)
        
        -- Execute input sequence
        execute_input_sequence(inputs)
    else
        -- No inputs, just log game state and advance
        log_game_state()
        emu.frameadvance()
    end
end 