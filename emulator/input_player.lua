-- Input Player for AI Training
-- Reads inputs from file and executes them while logging game state

-- Configuration
local INPUT_FILE = "ai_inputs.txt"
local LOG_FILE = "game_log.txt"
local FRAME_DELAY = 1  -- Frames to wait between inputs
local LOG_INTERVAL = 10  -- Log every N frames

-- Game state tracking
local last_state = {}
local frame_count = 0
local input_count = 0

-- Memory addresses for Sonic (Genesis)
local MEMORY_ADDRESSES = {
    -- Player position
    SONIC_X = 0xFFB000,
    SONIC_Y = 0xFFB004,
    
    -- Player state
    SONIC_STATE = 0xFFB023,
    SONIC_INVINCIBLE = 0xFFB02E,
    SONIC_RINGS = 0xFFB020,
    SONIC_LIVES = 0xFFB02A,
    
    -- Level info
    LEVEL_ID = 0xFFFE10,
    ACT_ID = 0xFFFE11,
    SCORE = 0xFFFE26,
    
    -- Game state
    GAME_STATE = 0xFFF600,
    TIME_MINUTES = 0xFFFE22,
    TIME_SECONDS = 0xFFFE23,
    TIME_FRAMES = 0xFFFE24,
    
    -- Camera position
    CAMERA_X = 0xFFEE00,
    CAMERA_Y = 0xFFEE04
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
    print(string.format("[InputPlayer] %s", message))
end

local function read_memory_byte(address)
    return memory.read_u8(address, "System Bus")
end

local function read_memory_word(address)
    return memory.read_u16_le(address, "System Bus")
end

local function read_memory_dword(address)
    return memory.read_u32_le(address, "System Bus")
end

local function get_game_state()
    local state = {}
    
    -- Read player position
    state.sonic_x = read_memory_word(MEMORY_ADDRESSES.SONIC_X)
    state.sonic_y = read_memory_word(MEMORY_ADDRESSES.SONIC_Y)
    
    -- Read player state
    state.sonic_state = read_memory_byte(MEMORY_ADDRESSES.SONIC_STATE)
    state.sonic_invincible = read_memory_byte(MEMORY_ADDRESSES.SONIC_INVINCIBLE)
    state.sonic_rings = read_memory_byte(MEMORY_ADDRESSES.SONIC_RINGS)
    state.sonic_lives = read_memory_byte(MEMORY_ADDRESSES.SONIC_LIVES)
    
    -- Read level info
    state.level_id = read_memory_byte(MEMORY_ADDRESSES.LEVEL_ID)
    state.act_id = read_memory_byte(MEMORY_ADDRESSES.ACT_ID)
    state.score = read_memory_dword(MEMORY_ADDRESSES.SCORE)
    
    -- Read game state
    state.game_state = read_memory_byte(MEMORY_ADDRESSES.GAME_STATE)
    state.time_minutes = read_memory_byte(MEMORY_ADDRESSES.TIME_MINUTES)
    state.time_seconds = read_memory_byte(MEMORY_ADDRESSES.TIME_SECONDS)
    state.time_frames = read_memory_byte(MEMORY_ADDRESSES.TIME_FRAMES)
    
    -- Read camera position
    state.camera_x = read_memory_word(MEMORY_ADDRESSES.CAMERA_X)
    state.camera_y = read_memory_word(MEMORY_ADDRESSES.CAMERA_Y)
    
    return state
end

local function set_input_state(input_name, state)
    if INPUT_MAP[input_name] then
        joypad.set(INPUT_MAP[input_name], state)
        log(string.format("Set %s = %s", input_name, tostring(state)))
    else
        log(string.format("Unknown input: %s", input_name))
    end
end

local function reset_all_inputs()
    for input_name, _ in pairs(INPUT_MAP) do
        set_input_state(input_name, false)
    end
    log("All inputs reset")
end

local function log_game_state(frame, input_info)
    local state = get_game_state()
    
    -- Create log entry
    local log_entry = string.format(
        "FRAME:%d|INPUT:%s|X:%d|Y:%d|RINGS:%d|LIVES:%d|LEVEL:%d|ACT:%d|SCORE:%d|TIME:%02d:%02d:%02d",
        frame,
        input_info or "NONE",
        state.sonic_x,
        state.sonic_y,
        state.sonic_rings,
        state.sonic_lives,
        state.level_id,
        state.act_id,
        state.score,
        state.time_minutes,
        state.time_seconds,
        state.time_frames
    )
    
    -- Write to log file
    local file = io.open(LOG_FILE, "a")
    if file then
        file:write(log_entry .. "\n")
        file:close()
    end
    
    -- Also print to console for debugging
    log(log_entry)
end

local function read_input_file()
    local file = io.open(INPUT_FILE, "r")
    if not file then
        return nil
    end
    
    local inputs = {}
    for line in file:lines() do
        line = line:gsub("%s+", "")  -- Remove whitespace
        if line ~= "" then
            table.insert(inputs, line)
        end
    end
    file:close()
    
    return inputs
end

local function clear_log_file()
    local file = io.open(LOG_FILE, "w")
    if file then
        file:write("")  -- Clear file
        file:close()
    end
end

-- Main execution
log("Input Player starting...")
log("Input file: " .. INPUT_FILE)
log("Log file: " .. LOG_FILE)
log("Current working directory: " .. (os.getenv("PWD") or "unknown"))

-- Clear previous log
clear_log_file()

-- Write header to log
local header_file = io.open(LOG_FILE, "w")
if header_file then
    header_file:write("FRAME|INPUT|X|Y|RINGS|LIVES|LEVEL|ACT|SCORE|TIME\n")
    header_file:close()
end

-- Main loop
while true do
    frame_count = frame_count + 1
    
    -- Check for input file
    local inputs = read_input_file()
    
    if inputs and #inputs > 0 then
        log("Found " .. #inputs .. " inputs to execute")
        
        -- Execute each input
        for i, input_line in ipairs(inputs) do
            input_count = input_count + 1
            
            -- Parse input (format: "FRAME:ACTION" or just "ACTION")
            local frame, action = input_line:match("(%d+):(.+)")
            if not frame then
                action = input_line
                frame = frame_count
            else
                frame = tonumber(frame)
            end
            
            -- Wait until we reach the target frame
            while frame_count < frame do
                emu.frameadvance()
                frame_count = frame_count + 1
                
                -- Log periodically
                if frame_count % LOG_INTERVAL == 0 then
                    log_game_state(frame_count, "WAIT")
                end
            end
            
            -- Execute the action
            if action == "RESET" then
                reset_all_inputs()
                log_game_state(frame_count, "RESET")
            elseif action == "NONE" or action == "NOOP" then
                log_game_state(frame_count, "NONE")
            else
                -- Parse multiple inputs (e.g., "LEFT+A")
                for input_name in action:gmatch("[^+]+") do
                    set_input_state(input_name, true)
                end
                
                log_game_state(frame_count, action)
                
                -- Hold inputs for a few frames
                for hold_frame = 1, FRAME_DELAY do
                    emu.frameadvance()
                    frame_count = frame_count + 1
                end
                
                -- Reset all inputs
                reset_all_inputs()
            end
        end
        
        -- Delete input file after execution
        os.remove(INPUT_FILE)
        log("Input execution complete, file deleted")
        
        -- Write completion marker
        local completion_file = io.open("execution_complete.txt", "w")
        if completion_file then
            completion_file:write(string.format("COMPLETE|FRAMES:%d|INPUTS:%d", frame_count, input_count))
            completion_file:close()
        end
        
        log("Execution complete. Waiting for new input file...")
        
    else
        -- No input file, just advance frame and log periodically
        if frame_count % LOG_INTERVAL == 0 then
            log_game_state(frame_count, "IDLE")
        end
    end
    
    emu.frameadvance()
end 