# 🧹 Sonic AI Codebase Cleanup Plan

## 🎯 **Overview**
This document outlines a comprehensive cleanup of the Sonic AI codebase to remove unnecessary code, unused files, repetitive tests, and improve code quality following best practices.

## 📊 **Current Issues Identified**

### **1. Duplicate/Redundant Files**
- `test_simplified_approach.py` vs `test_simplified_integration.py` (similar functionality)
- `MARIO_AI_COMPARISON.md` vs `SIMPLIFIED_APPROACH_SUMMARY.md` (redundant documentation)
- `QUICK_START_SIMPLIFIED.md` (could be merged into main README)
- Multiple debug/test files with overlapping functionality

### **2. Unused/Empty Directories**
- `bizhawk_comm_0/` (empty directory)
- `test_ai_training/` (minimal content)
- `__pycache__/` directories (should be in .gitignore)

### **3. Debug/Test Files to Clean**
- `debug_bizhawk.py` (debugging script)
- `debug_input_system.py` (debugging script)
- `quick_input_test.py` (test script)
- `test_file_based_ai.py` (test script)
- `test_file_bridge.py` (test script)
- `check_windows.py` (utility script)

### **4. Configuration Files**
- Multiple RetroArch config files (keep only essential ones)
- Multiple training configs (consolidate)

### **5. Logging Issues**
- Excessive print statements instead of proper logging
- Inconsistent logging levels
- No structured logging

### **6. Code Quality Issues**
- Long functions that need refactoring
- Inconsistent error handling
- Missing type hints in some areas
- Hardcoded values that should be configurable

## 🚀 **Cleanup Actions**

### **Phase 1: Remove Unnecessary Files**
1. Delete empty directories
2. Remove redundant test files
3. Consolidate documentation
4. Remove debug scripts (or move to debug/ directory)

### **Phase 2: Consolidate Configuration**
1. Merge training configurations
2. Keep only essential RetroArch configs
3. Create unified config structure

### **Phase 3: Improve Code Quality**
1. Replace print statements with proper logging
2. Add missing type hints
3. Refactor long functions
4. Improve error handling
5. Add docstrings where missing

### **Phase 4: Organize Structure**
1. Create proper directory structure
2. Move files to appropriate locations
3. Update imports and paths
4. Create unified entry points

## 📁 **Proposed New Structure**

```
sonic-ai/
├── README.md                    # Main documentation
├── requirements.txt             # Dependencies
├── setup.py                     # Installation
├── .gitignore                   # Git ignore rules
├── LICENSE                      # License
├── configs/                     # Configuration files
│   ├── training_config.yaml     # Main training config
│   └── simplified_config.yaml   # Simplified training config
├── src/                         # Main source code
│   ├── environment/             # Environment code
│   ├── agents/                  # Agent implementations
│   ├── utils/                   # Utility functions
│   └── visualization/           # Visualization tools
├── scripts/                     # Executable scripts
│   ├── train.py                 # Main training script
│   ├── train_simplified.py      # Simplified training
│   ├── play.py                  # Play trained models
│   └── cleanup.py               # System cleanup
├── tests/                       # Test files
│   ├── test_environment.py      # Environment tests
│   ├── test_agents.py           # Agent tests
│   └── test_integration.py      # Integration tests
├── docs/                        # Documentation
│   ├── TRAINING_GUIDE.md        # Training guide
│   ├── SETUP_GUIDE.md           # Setup guide
│   └── API_REFERENCE.md         # API documentation
├── debug/                       # Debug scripts (optional)
├── logs/                        # Training logs
├── models/                      # Trained models
├── data/                        # Data files
└── roms/                        # Game ROMs
```

## 🎯 **Specific Files to Remove/Consolidate**

### **Files to Delete:**
- `bizhawk_comm_0/` (empty directory)
- `test_simplified_approach.py` (redundant with test_simplified_integration.py)
- `debug_bizhawk.py` (move to debug/ if needed)
- `debug_input_system.py` (move to debug/ if needed)
- `quick_input_test.py` (consolidate into tests/)
- `test_file_based_ai.py` (consolidate into tests/)
- `test_file_bridge.py` (consolidate into tests/)
- `check_windows.py` (move to scripts/ if needed)
- `MARIO_AI_COMPARISON.md` (merge into docs/)
- `SIMPLIFIED_APPROACH_SUMMARY.md` (merge into docs/)
- `QUICK_START_SIMPLIFIED.md` (merge into main README)
- Multiple RetroArch config files (keep only essential)

### **Files to Consolidate:**
- Merge training configurations
- Consolidate test files
- Merge documentation files

### **Files to Improve:**
- Replace print statements with logging
- Add type hints
- Improve error handling
- Add docstrings

## 📋 **Implementation Steps**

1. **Backup current state**
2. **Remove unnecessary files**
3. **Consolidate configurations**
4. **Improve code quality**
5. **Update documentation**
6. **Test everything works**
7. **Update .gitignore**

## ✅ **Success Criteria**

- [ ] No duplicate functionality
- [ ] Clean directory structure
- [ ] Proper logging throughout
- [ ] Consistent code style
- [ ] Complete documentation
- [ ] All tests pass
- [ ] No unused imports
- [ ] Proper error handling
- [ ] Type hints where appropriate
- [ ] Clear entry points

## 🚨 **Risk Mitigation**

- Keep backups of all files before deletion
- Test thoroughly after each change
- Maintain git history for important changes
- Document all changes made
- Ensure all functionality is preserved
