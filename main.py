import json
with open("data/dev.json") as f:
    dev = json.load(f)

# Take first 1200 examples
subset = dev[:1200]

# Save to new file
with open("1200devTestSet.json", "w") as f:
    json.dump(subset, f, indent=4)

print("Saved 1200 entries to 1200devTestSet.json")
