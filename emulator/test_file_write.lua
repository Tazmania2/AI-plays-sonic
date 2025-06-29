-- Simple file write test for BizHawk
print("Testing file writing...")

-- Try to write a test file
local test_file = io.open("test_write.txt", "w")
if test_file then
    test_file:write("Test successful at frame " .. emu.framecount() .. "\n")
    test_file:close()
    print("File write successful!")
else
    print("File write failed!")
end

-- Try to read the input file
local input_file = io.open("ai_inputs.txt", "r")
if input_file then
    print("Input file found!")
    local content = input_file:read("*all")
    print("Content: " .. content)
    input_file:close()
else
    print("Input file not found!")
end

-- List current directory contents
print("Current directory contents:")
local handle = io.popen("dir")
if handle then
    local result = handle:read("*all")
    print(result)
    handle:close()
end

print("Test complete!") 