# Stage 2A Smart Filtering Guide

## Problem
Original subgraph retrieval returned **too much data**:
- 369 nodes, 1654 edges
- 1.0 MB JSON file
- 131 TailwindUtility nodes
- 1329 styling edges
- Deep JSX trees with overwhelming detail

## Solution: Smart Filtering

### Three Filtering Levels

#### 1. **Minimal** (Recommended for Test Generation)
Focus only on test-relevant entities:
```bash
python subgraph_retrieval.py --input-json stage1_output.json \
  --include-only Component Route State EventHandler Hook Prop
```

**Result**: 82 nodes, 94 edges, 98 KB
- ✅ All components and routes
- ✅ State management
- ✅ Event handlers
- ✅ Props and hooks
- ❌ No JSX elements (too granular)
- ❌ No styling noise

**Use when**: Generating integration tests, E2E tests, or understanding component architecture

---

#### 2. **Smart Filtered** (Default)
Balanced view with aggregated styling:
```bash
python subgraph_retrieval.py --input-json stage1_output.json \
  --depth 2 --jsx-depth 1
```

**Result**: 235 nodes, 247 edges, ~250 KB
- ✅ Components, routes, state, handlers
- ✅ Top-level JSX elements (depth=1)
- ✅ Styling aggregated into summary
- ❌ No individual Tailwind/inline style nodes

**Use when**: Need some UI structure but want to avoid styling noise

---

#### 3. **Full** (Legacy)
Everything including styling details:
```bash
python subgraph_retrieval.py --input-json stage1_output.json \
  --depth 2 --jsx-depth 0 --no-aggregate-styling
```

**Result**: 369 nodes, 1654 edges, 1.0 MB
- ✅ Complete graph
- ⚠️ Heavy with styling nodes
- ⚠️ Deep JSX trees

**Use when**: Debugging or need complete styling information

---

## Configuration Options

### Environment Variables
```bash
# Depth controls
export STAGE2_MAX_DEPTH=2              # Graph traversal depth
export STAGE2_MAX_JSX_DEPTH=1          # JSX tree depth (0=unlimited)

# Filtering
export STAGE2_AGGREGATE_STYLING=true   # Aggregate Tailwind/inline styles
export STAGE2_INCLUDE_FILE_CONTEXT=false  # Include file nodes
```

### CLI Arguments
```bash
--depth N                # Max graph traversal depth (default: 2)
--jsx-depth N           # Max JSX tree depth (default: 1, 0=unlimited)
--include-only TYPE...  # Whitelist: only include these node types
--exclude TYPE...       # Blacklist: exclude these node types
--no-aggregate-styling  # Keep individual styling nodes
--include-file-context  # Add file containment relationships
```

---

## Node Type Reference

### Test-Relevant (Minimal)
- `Component` - React components
- `Route` - Application routes
- `State` - Component state (useState, etc.)
- `EventHandler` - User interaction handlers
- `Hook` - React hooks usage
- `Prop` - Component props

### UI Structure
- `JSXElement` - DOM/component elements
- `File` - Source files (optional)

### Styling (Usually Aggregated)
- `TailwindUtility` - Individual Tailwind classes
- `InlineStyle` - Inline style objects

---

## Recommendations by Use Case

### 🧪 Test Case Generation (Stage 3)
```bash
--include-only Component Route State EventHandler Hook Prop --depth 2
```
**Why**: Tests care about behavior (state, handlers), not presentation (styling, DOM structure)

### 📊 Component Architecture Analysis
```bash
--depth 2 --jsx-depth 0 --exclude TailwindUtility InlineStyle
```
**Why**: See full component relationships without styling noise

### 🎨 UI/Styling Analysis
```bash
--depth 2 --jsx-depth 2 --no-aggregate-styling
```
**Why**: Need detailed styling information

### 🔍 Debugging/Exploration
```bash
--depth 3 --jsx-depth 0 --no-aggregate-styling --include-file-context
```
**Why**: Maximum detail for investigation

---

## Performance Impact

| Level | Nodes | Edges | File Size | Processing Time | Memory |
|-------|-------|-------|-----------|----------------|---------|
| Minimal | 82 | 94 | 98 KB | ~0.5s | Low |
| Smart Filtered | 235 | 247 | ~250 KB | ~1s | Medium |
| Full | 369 | 1654 | 1.0 MB | ~2s | High |

---

## Migration from v1

**Old (v1)**:
```python
subgraph = retrieve_subgraph(structured_input, depth=2)
# Returns 369 nodes with styling noise
```

**New (v2 - Minimal)**:
```python
config = Stage2Config()
config.include_only_node_types = ["Component", "Route", "State", "EventHandler", "Hook", "Prop"]
subgraph = retrieve_subgraph(structured_input, depth=2, config=config)
# Returns 82 nodes, focused on test-relevant entities
```

**New (v2 - Smart Default)**:
```python
# Just use defaults - automatically filters styling
subgraph = retrieve_subgraph(structured_input, depth=2)
# Returns 235 nodes with aggregated styling
```

