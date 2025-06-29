-- Debug Input Player for AI Training
-- Simplified version with detailed logging

-- Configuration
local INPUT_FILE = "ai_inputs.txt"
local LOG_FILE = "game_log.txt"
local COMPLETION_FILE = "execution_complete.txt"

-- Game state tracking
local frame_count = 0
local input_count = 0
local inputs_processed = false

-- Memory addresses for Sonic (Genesis)
local MEMORY_ADDRESSES = {
    SONIC_X = 0xFFB000,
    SONIC_Y = 0xFFB004,
    SONIC_RINGS = 0xFFB020,
    SONIC_LIVES = 0xFFB02A,
    LEVEL_ID = 0xFFFE10,
    ACT_ID = 0xFFFE11,
    SCORE = 0xFFFE26,
    TIME_MINUTES = 0xFFFE22,
    TIME_SECONDS = 0xFFFE23,
    TIME_FRAMES = 0xFFFE24
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
    print(string.format("[DEBUG] %s", message))
end

local function read_memory_byte(address)
    return memory.read_u8(address, "System Bus")
end

local function read_memory_word(address)
    return memory.read_u16_le(address, "System Bus")
end

local function get_game_state()
    local state = {}
    state.sonic_x = read_memory_word(MEMORY_ADDRESSES.SONIC_X)
    state.sonic_y = read_memory_word(MEMORY_ADDRESSES.SONIC_Y)
    state.sonic_rings = read_memory_byte(MEMORY_ADDRESSES.SONIC_RINGS)
    state.sonic_lives = read_memory_byte(MEMORY_ADDRESSES.SONIC_LIVES)
    state.level_id = read_memory_byte(MEMORY_ADDRESSES.LEVEL_ID)
    state.act_id = read_memory_byte(MEMORY_ADDRESSES.ACT_ID)
    state.score = read_memory_word(MEMORY_ADDRESSES.SCORE)
    state.time_minutes = read_memory_byte(MEMORY_ADDRESSES.TIME_MINUTES)
    state.time_seconds = read_memory_byte(MEMORY_ADDRESSES.TIME_SECONDS)
    state.time_frames = read_memory_byte(MEMORY_ADDRESSES.TIME_FRAMES)
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
end

local function log_game_state(frame, input_info)
    local state = get_game_state()
    
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
    
    -- Try to write to log file
    local file = io.open(LOG_FILE, "a")
    if file then
        file:write(log_entry .. "\n")
        file:close()
        log("Wrote to log file: " .. log_entry)
    else
        log("Failed to write to log file!")
    end
    
    -- Also print to console
    log(log_entry)
end

local function read_input_file()
    log("Attempting to read input file: " .. INPUT_FILE)
    
    local file = io.open(INPUT_FILE, "r")
    if not file then
        log("Input file not found!")
        return nil
    end
    
    local inputs = {}
    for line in file:lines() do
        line = line:gsub("%s+", "")  -- Remove whitespace
        if line ~= "" then
            table.insert(inputs, line)
            log("Read input: " .. line)
        end
    end
    file:close()
    
    log("Total inputs read: " .. #inputs)
    return inputs
end

local function write_completion_file()
    log("Writing completion file...")
    local file = io.open(COMPLETION_FILE, "w")
    if file then
        file:write(string.format("COMPLETE|FRAMES:%d|INPUTS:%d", frame_count, input_count))
        file:close()
        log("Completion file written successfully!")
    else
        log("Failed to write completion file!")
    end
end

-- Main execution
log("Debug Input Player starting...")
log("Input file: " .. INPUT_FILE)
log("Log file: " .. LOG_FILE)
log("Completion file: " .. COMPLETION_FILE)

-- Clear previous log
local clear_file = io.open(LOG_FILE, "w")
if clear_file then
    clear_file:write("FRAME|INPUT|X|Y|RINGS|LIVES|LEVEL|ACT|SCORE|TIME\n")
    clear_file:close()
    log("Log file cleared and header written")
else
    log("Failed to clear log file!")
end

-- Main loop
while true do
    frame_count = frame_count + 1
    
    -- Check for input file if we haven't processed inputs yet
    if not inputs_processed then
        local inputs = read_input_file()
        
        if inputs and #inputs > 0 then
            log("Processing " .. #inputs .. " inputs...")
            
            -- Execute each input
            for i, input_line in ipairs(inputs) do
                input_count = input_count + 1
                log("Processing input " .. i .. ": " .. input_line)
                
                -- Parse input (format: "FRAME:ACTION")
                local frame, action = input_line:match("(%d+):(.+)")
                if not frame then
                    log("Invalid input format: " .. input_line)
                    goto continue
                end
                
                frame = tonumber(frame)
                log("Target frame: " .. frame .. ", Action: " .. action)
                
                -- Wait until we reach the target frame
                while frame_count < frame do
                    emu.frameadvance()
                    frame_count = frame_count + 1
                    
                    -- Log every 10 frames
                    if frame_count % 10 == 0 then
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
                    
                    -- Hold inputs for 2 frames
                    for hold_frame = 1, 2 do
                        emu.frameadvance()
                        frame_count = frame_count + 1
                    end
                    
                    -- Reset all inputs
                    reset_all_inputs()
                end
                
                ::continue::
            end
            
            -- Delete input file after execution
            if os.remove(INPUT_FILE) then
                log("Input file deleted successfully")
            else
                log("Failed to delete input file")
            end
            
            -- Write completion marker
            write_completion_file()
            
            inputs_processed = true
            log("Input execution complete!")
            
        else
            -- No input file, just advance frame and log periodically
            if frame_count % 10 == 0 then
                log_game_state(frame_count, "IDLE")
            end
        end
    else
        -- Inputs already processed, just advance frame
        if frame_count % 10 == 0 then
            log_game_state(frame_count, "IDLE")
        end
    end
    
    emu.frameadvance()
end 