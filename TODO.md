# TODO - KazusaAIChatbot

## Current Tasks

### High Priority
- [ ] Test the enhanced affinity system with 21 levels (every 50 points)
- [ ] Verify the aggressive memory storage system is working correctly
- [ ] Test the new timestamp awareness in supervisor prompts
- [ ] Validate step logging provides clear visibility into the pipeline

### Medium Priority
- [ ] Add unit tests for the refined affinity system
- [ ] Create integration tests for memory storage aggressiveness
- [ ] Add logging level configuration for different environments
- [ ] Document the new affinity levels and their behavioral guidelines

### Low Priority
- [ ] Consider adding affinity change tracking over time
- [ ] Add memory cleanup/retention policies
- [ ] Create performance metrics for agent dispatch timing

## Recent Changes

### ✅ Completed
- **Enhanced Affinity System**: Expanded from 5 to 21 levels (every 50 points) with nuanced behavioral instructions
- **Aggressive Memory Storage**: Updated memory check system to be more proactive about storing facts, especially from web searches
- **Timestamp Awareness**: Added current system time to all supervisor LLM prompts
- **Step Logging**: Added clear logging for each supervisor stage (Step 0-6)

### 🔧 In Progress
- Testing and validation of recent changes

## 🎯 Architecture Implementation Gap Analysis

### Current vs. Vision - What We Have vs. What We Need

#### **1. The Intent Layer (Processed Input)**
**✅ Already Done:**
- Supervisor gets `assembler_output` from the Assembler node
- Includes `channel_topic` and `user_topic` classification
- Has access to `message_text` and basic context

**❌ Missing/Needs Implementation:**
- [ ] **Classified Intent**: Only basic topic classification, not intent types (technical fix vs emotional venting)
- [ ] **Entities Extracted**: No named entity recognition (names, dates, IDs)
- [ ] **Language & Tone**: No sentiment analysis or language detection

**Implementation Needed:**
```python
# Need to add to assembler_output:
{
    "classified_intent": "technical_fix|emotional_vent|question|casual_chat",
    "extracted_entities": {"names": [], "dates": [], "ids": []},
    "language_detected": "en|es|...",
    "sentiment": "positive|negative|neutral",
    "urgency": "high|medium|low"
}
```

#### **2. The Identity Layer (Persona & Relationship)**
**✅ Already Done:**
- Supervisor has access to `personality` object with persona constraints
- Uses `persona_name` and `bot_id` for identity
- Has `affinity` scoring (0-1000) with refined 21-level system
- Basic `user_memory` for long-term preferences

**❌ Missing/Needs Implementation:**
- [ ] **User Profile**: Limited structured user preferences (communication style, knowledge level, etc.)
- [ ] **Enhanced Persona Constraints**: Could be more detailed than current basic personality object

**Implementation Needed:**
```python
# Need to expand user profile in memory:
{
    "communication_preferences": {
        "response_length": "short|detailed",
        "formality_level": "casual|formal",
        "preferred_tone": "technical|friendly"
    },
    "technical_background": {
        "python_level": "beginner|intermediate|expert",
        "domain_expertise": ["web_dev", "data_science", ...]
    }
}
```

#### **3. The Ground Truth Layer (RAG & Memory)**
**✅ Already Done:**
- **Short-term Memory**: Access to `conversation_history` (limited by CONVERSATION_HISTORY_LIMIT)
- **Long-term Memory**: `user_memory` and `user_facts` from database
- **Knowledge System**: Has `knowledge_agent` for detailed document retrieval

**❌ Missing/Needs Implementation:**
- [ ] **Retrieved Documents**: No automatic RAG retrieval in supervisor context
- [ ] **Structured Memory**: Memory format is still simple descriptions, not rich structured data
- [ ] **Knowledge Integration**: Supervisor doesn't automatically fetch relevant knowledge

**Implementation Needed:**
```python
# Need to add automatic retrieval:
{
    "retrieved_documents": [
        {
            "content": "relevant snippet",
            "source": "conversation|knowledge_base|web_search",
            "relevance_score": 0.95,
            "timestamp": "2025-01-05T..."
        }
    ]
}
```

#### **4. The "Agent Catalog" (Capability Map)**
**✅ Already Done:**
- Has `_build_agent_catalog()` that creates structured agent descriptions
- Includes agent names and capability descriptions
- Supervisor uses this for planning decisions

**❌ Missing/Needs Implementation:**
- [ ] **Input Requirements**: No structured input requirements for each agent
- [ ] **Capability Mapping**: Could be more detailed about when to use each agent

**Implementation Needed:**
```python
# Need to improve agent catalog structure:
{
    "web_search_agent": {
        "capability": "Real-time information retrieval",
        "use_cases": ["current events", "technical documentation", "news"],
        "input_requirements": {
            "query": "string (required)",
            "time_range": "optional (day|week|month)",
            "language": "optional"
        }
    }
}
```

## 📋 Implementation Roadmap

### **Phase 1** (Immediate - Intent Layer Enhancement)
- [ ] Enhance assembler to classify intent and extract entities
- [ ] Add sentiment and language detection to assembler
- [ ] Update supervisor prompts to use enhanced intent data
- [ ] Add intent classification: technical_fix, emotional_vent, question, casual_chat
- [ ] Implement named entity recognition for names, dates, IDs
- [ ] Add urgency detection for message prioritization

### **Phase 2** (Short-term - Memory & RAG Enhancement)
- [ ] Implement structured user profiles with communication preferences
- [ ] Add technical background tracking (skill levels, domain expertise)
- [ ] Implement automatic RAG retrieval for relevant documents
- [ ] Enhance memory format with metadata and categorization
- [ ] Add knowledge fetching to relevance_agent (as noted in TODO)
- [ ] Implement memory importance scoring and retrieval priority

### **Phase 3** (Medium-term - Agent Catalog & Supervisor Enhancement)
- [ ] Improve agent catalog with detailed capability mapping
- [ ] Add input requirements and use case examples for each agent
- [ ] Better agent selection logic in supervisor
- [ ] Move memory responsibilities from memory_writer to supervisor
- [ ] Rewrite relevance_agent to work with enhanced memory system

### **Phase 4** (Long-term - Advanced Features)
- [ ] Add affinity change tracking over time
- [ ] Implement memory cleanup and retention policies
- [ ] Add performance metrics and monitoring
- [ ] Create comprehensive test coverage for new features

## 🚀 Major Architecture Tasks

### Supervisor-Driven Memory Management
- [ ] Move memory responsibilities from memory_writer to supervisor
- [ ] Supervisor should fetch user-related memory proactively
- [ ] Keep memory structure consistent under supervisor control
- [ ] Remove redundant memory logic from memory_writer

### Enhanced User Memory Fields
- [ ] Add user preferences, interests, habits tracking
- [ ] Track user communication style and patterns
- [ ] Record user's technical knowledge level
- [ ] Store user's emotional state patterns
- [ ] Track user's preferred interaction frequency/times

### Rich Memory Storage Format
- [ ] Store full conversation context instead of summaries
- [ ] Include metadata: timestamps, topics, sentiment analysis
- [ ] Categorize memories: factual, emotional, preference, contextual
- [ ] Add memory importance scoring and retrieval priority
- [ ] Include source attribution and cross-references

## Testing Requirements

### New Test Coverage Needed
- [ ] Test intent classification accuracy
- [ ] Validate entity recognition functionality
- [ ] Test enhanced user profile tracking
- [ ] Validate RAG retrieval effectiveness
- [ ] Test structured memory storage and retrieval
- [ ] Validate enhanced agent catalog functionality

### Integration Tests
- [ ] End-to-end tests for enhanced intent processing
- [ ] Test supervisor with rich memory context
- [ ] Validate agent selection with enhanced catalog
- [ ] Test knowledge fetching and integration

## Documentation Updates

### User Documentation
- [ ] Document enhanced intent classification system
- [ ] Create user guide for structured profiles
- [ ] Document new memory capabilities

### Developer Documentation
- [ ] Document enhanced assembler processing
- [ ] Update API documentation for new features
- [ ] Create architecture decision records for changes

---

**Last Updated**: 2025-01-05  
**Maintainers**: KazusaAIChatbot Development Team

## 🎯 Vision Summary

**Current State**: Solid foundation with basic intent processing, affinity system, and agent catalog
**Target State**: Comprehensive layered architecture with rich intent classification, detailed user profiles, automatic RAG, and enhanced agent capabilities
**Gap Analysis**: 40% of vision implemented, 60% needs development across all four layers

**Key Insight**: We have the right components but need to enhance each layer with the detailed structured data and processing capabilities outlined in the vision.
