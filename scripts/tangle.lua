-- Lua filter for pandoc to tangle org-mode code blocks
-- Extracts code blocks with :tangle attribute and writes them to files

local tangle_files = {}
local errors = {}

function CodeBlock(block)
  -- Look for tangle attribute in block attributes
  local tangle_path = block.attributes and block.attributes['tangle']

  if tangle_path then
    -- Initialize file entry if first time seeing this path
    if not tangle_files[tangle_path] then
      tangle_files[tangle_path] = ""
    end

    -- Append code block content with newline separator
    tangle_files[tangle_path] = tangle_files[tangle_path] .. block.text .. "\n"
  end

  return block
end

function Pandoc(doc)
  -- Write tangled files after document is processed
  for filepath, content in pairs(tangle_files) do
    local file, err = io.open(filepath, "w")
    if not file then
      table.insert(errors, "Failed to open " .. filepath .. ": " .. err)
    else
      file:write(content)
      file:close()
      print("Tangled: " .. filepath)
    end
  end

  -- Exit with error if any files failed to write
  if #errors > 0 then
    for _, err in ipairs(errors) do
      io.stderr:write("ERROR: " .. err .. "\n")
    end
    os.exit(1)
  end
end
