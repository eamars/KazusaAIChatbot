# Debugging Scripts Guide

## SKILL: Memory Debugging Operations

### insert_memory
**Purpose**: Debug memory storage by inserting test memory entries with automatic embedding generation.

**Usage**:
```bash
python -m src.scripts.insert_memory <memory_name> <content>
```

**Arguments**:
- `memory_name`: Name/identifier for the memory (required)
- `content`: The memory content/text (required)

**Debugging Features**:
- Test memory insertion functionality
- Verify embedding generation works
- Validate database connections
- Monitor logging output for errors
- Supports test data creation for search debugging

**Debugging Examples**:
```bash
# Insert test memory for debugging search
python -m src.scripts.insert_memory "Test Memory" "This is a test memory entry for debugging purposes."

# Insert sample data for vector search testing
python -m src.scripts.insert_memory "Python Debug" "Python debugging test data for vector search validation."
```

### search_memory
**Purpose**: Debug memory search functionality by testing both keyword and vector search methods.

**Usage**:
```bash
python -m src.scripts.search_memory <query> [--method keyword|vector] [--limit N]
```

**Arguments**:
- `query`: Search query string (required)
- `--method`: Search method - `keyword` (regex) or `vector` (semantic) (default: vector)
- `--limit`: Maximum number of results (default: 5)

**Debugging Features**:
- Test keyword regex search patterns
- Validate vector search embeddings
- Check similarity scores
- Verify search result formatting
- Debug search performance issues

**Debugging Examples**:
```bash
# Test vector search with known memory
python -m src.scripts.search_memory "Test Memory" --method vector --limit 3

# Debug keyword search with regex
python -m src.scripts.search_memory "Python.*Debug" --method keyword --limit 5

# Test search with different limits
python -m src.scripts.search_memory "debug" --method vector --limit 10
```

---

## SKILL: Conversation History Debugging

### search_conversation
**Purpose**: Debug conversation history search functionality with filtering and search method testing.

**Usage**:
```bash
python -m src.scripts.search_conversation <query> [--method keyword|vector] [--channel ID] [--user ID] [--limit N]
```

**Arguments**:
- `query`: Search query string (required)
- `--method`: Search method - `keyword` (regex) or `vector` (semantic) (default: vector)
- `--channel`: Filter by channel ID (optional)
- `--user`: Filter by user ID (optional)
- `--limit`: Maximum number of results (default: 5)

**Debugging Features**:
- Test conversation search functionality
- Validate filtering by user/channel
- Debug embedding vs keyword search differences
- Test search performance with different queries
- Verify conversation history formatting

**Debugging Examples**:
```bash
# Debug conversation search with specific user
python -m src.scripts.search_conversation "debug" --user 320899931776745483 --method keyword

# Test vector search in specific channel
python -m src.scripts.search_conversation "testing" --channel 123456789 --method vector

# Debug search result limits
python -m src.scripts.search_conversation "error" --limit 10
```

---

## SKILL: Embedding System Debugging

### create_conversation_history_embedding
**Purpose**: Debug conversation history embedding creation and vector index functionality.

**Usage**:
```bash
python -m src.scripts.create_conversation_history_embedding [--channel ID] [--limit N]
```

**Debugging Features**:
- Test embedding generation for conversations
- Validate vector index creation
- Debug embedding API connections
- Monitor embedding processing progress
- Test channel-specific embedding creation

**When to Use for Debugging**:
- Vector search returns no results
- After migrating conversation data
- When search functionality fails
- To test embedding API connectivity

### create_user_facts_embedding
**Purpose**: Debug user facts embedding creation and search functionality.

**Usage**:
```bash
python -m src.scripts.create_user_facts_embedding [--user ID] [--limit N]
```

**Debugging Features**:
- Test user facts embedding generation
- Validate user preference search
- Debug user memory indexing
- Test user-specific embedding processing

**When to Use for Debugging**:
- User fact search isn't working
- After user data migration
- When user preference search fails
- To test user embedding API calls

---

## SKILL: Interactive Supervisor Debugging

### debug_supervisor_speech
**Purpose**: Interactive debugging tool for testing supervisor planning and speech generation in isolation.

**Usage**:
```bash
python -m src.scripts.debug_supervisor_speech
```

**Environment Variables**:
- `DEBUG_PERSONALITY_PATH`: Path to personality file (optional)
- `DEBUG_USER_ID`: Default user ID (default: "debug_user")
- `DEBUG_USER_NAME`: Default user name (default: "Debugger")
- `DEBUG_CHANNEL_ID`: Default channel ID (default: "debug_channel")
- `DEBUG_BOT_ID`: Default bot ID (default: "debug_bot")
- `DEBUG_CHANNEL_TOPIC`: Default channel topic (default: "Debug session")
- `DEBUG_USER_TOPIC`: Default user topic (default: "Manual prompt debugging")

**Debugging Features**:
- Interactive prompt for testing different scenarios
- Real-time supervisor planning output
- Agent execution debugging
- Speech generation analysis
- Personality behavior validation
- Multi-agent coordination testing

**Debugging Use Cases**:
- Debug supervisor planning logic
- Test agent coordination issues
- Analyze speech generation problems
- Validate personality behavior changes
- Test new agent implementations
- Debug multi-agent workflows

**Debugging Session Example**:
```
Supervisor + Speech debugger
Press Enter to use the shown default. Type quit to exit.

user_id [debug_user]: test_user_123
user_name [Debugger]: TestUser
channel_id [debug_channel]: test_channel
bot_id [debug_bot]: kazusa_bot

message_text: Do you remember the Python programming guide?

===== supervisor_plan =====
{
  "agents": ["memory_agent"],
  "instructions": {
    "memory_agent": {
      "command": "Recall stored memory about Python programming guide",
      "expected_response": "Return relevant Python programming information"
    }
  }
}

===== agent_results =====
[{"agent": "memory_agent", "status": "success", "summary": "..."}]

===== speech_brief =====
{"response_brief": {"topics_to_cover": ["Python programming"], ...}}

===== response =====
I remember that Python is a high-level programming language...
```

---

## WORKFLOW: Debugging Memory Search Issues

### Step 1: Verify Vector Index
```bash
python -c "import asyncio; from kazusa_ai_chatbot.db import enable_memory_vector_index; asyncio.run(enable_memory_vector_index())"
```

### Step 2: Insert Test Data
```bash
python -m src.scripts.insert_memory "Debug Test" "Test memory for debugging search functionality."
```

### Step 3: Test Search Methods
```bash
python -m src.scripts.search_memory "Debug Test" --method vector
python -m src.scripts.search_memory "Debug.*Test" --method keyword
```

### Step 4: Analyze Results
- Check if vector search returns results
- Verify similarity scores are reasonable
- Test with different query variations

---

## WORKFLOW: Debugging Agent Coordination

### Step 1: Start Interactive Debug
```bash
python -m src.scripts.debug_supervisor_speech
```

### Step 2: Test Agent Dispatch
- Try memory recall: "Do you remember [topic]?"
- Test web search: "What's the latest news about [topic]?"
- Check conversation history: "What did we discuss about [topic]?"

### Step 3: Analyze Output
- Review supervisor planning decisions
- Check agent execution results
- Verify speech generation quality
- Test different personality configurations

---

## TROUBLESHOOTING: Common Debugging Scenarios

### Vector Search Returns No Results
**Debugging Steps**:
1. Check vector index: `enable_memory_vector_index()`
2. Verify embeddings exist: Test with known data
3. Check embedding API: Monitor logs for API calls
4. Test query relevance: Try different search terms

### Agent Execution Fails
**Debugging Steps**:
1. Use debug supervisor to isolate issues
2. Check agent registration in `AGENT_REGISTRY`
3. Verify personality file format and content
4. Test individual agent with debug supervisor

### Memory Insertion Fails
**Debugging Steps**:
1. Check LLM service connectivity
2. Verify API keys and base URLs
3. Test with simple content first
4. Monitor embedding generation logs

### Supervisor Planning Issues
**Debugging Steps**:
1. Use debug supervisor with verbose logging
2. Test different message types
3. Verify personality configuration
4. Check agent availability in registry

---

## BEST PRACTICES: Debugging with Scripts

### Memory Debugging
- Use descriptive test memory names
- Test both search methods for validation
- Monitor embedding generation in logs
- Create test datasets for consistent debugging

### Agent Debugging
- Use debug supervisor for complex scenarios
- Test individual agents before integration
- Keep personality files well-structured
- Document debugging scenarios for regression testing

### Performance Debugging
- Monitor search result times
- Test with different result limits
- Check database connection usage
- Profile embedding generation performance
- Ensure vector index exists: Run `enable_memory_vector_index()`
- Check embeddings are generated for existing data
- Verify query terms are relevant

### Scripts Fail to Connect to Database
**Solutions**:
- Check MongoDB connection string
- Ensure MongoDB is accessible
- Verify database name is correct

### Memory Insertion Fails
**Solutions**:
- Check LLM service is running for embeddings
- Verify API key and base URL are correct
- Ensure content is not empty

### Debug Supervisor Shows Errors
**Solutions**:
- Check all agents are registered
- Verify personality file exists and is valid JSON
- Ensure MCP services are available for web search

---

## BEST PRACTICES

### Memory Management
- Use descriptive memory names for better searchability
- Test both keyword and vector search methods
- Monitor embedding generation in logs

### Debugging
- Use debug supervisor for complex multi-agent scenarios
- Test individual components before full integration
- Keep personality files well-structured

### Performance
- Limit search results to improve response times
- Use appropriate search methods for different use cases
- Monitor database connection usage

**When to Use**:
- Initial setup of conversation search
- After migrating conversation data
- When search functionality isn't working properly

### create_user_facts_embedding.py

**Purpose**: Create embeddings for user facts to enable semantic search of user preferences.

**Usage**:
```bash
python -m src.scripts.create_user_facts_embedding [--user ID] [--limit N]
```

**Features**:
- Processes user facts and generates embeddings
- Supports user-specific processing
- Configurable processing limits
- Progress logging

**When to Use**:
- Initial setup of user fact search
- After migrating user data
- When user preference search isn't working

---

## Debug Scripts

### debug_supervisor_speech.py

**Purpose**: Interactive debugging tool for testing supervisor planning and speech generation in isolation.

**Usage**:
```bash
python -m src.scripts.debug_supervisor_speech
```

**Environment Variables**:
- `DEBUG_PERSONALITY_PATH`: Path to personality file (optional)
- `DEBUG_USER_ID`: Default user ID (default: "debug_user")
- `DEBUG_USER_NAME`: Default user name (default: "Debugger")
- `DEBUG_CHANNEL_ID`: Default channel ID (default: "debug_channel")
- `DEBUG_BOT_ID`: Default bot ID (default: "debug_bot")
- `DEBUG_CHANNEL_TOPIC`: Default channel topic (default: "Debug session")
- `DEBUG_USER_TOPIC`: Default user topic (default: "Manual prompt debugging")

**Features**:
- Interactive prompt for testing different scenarios
- Loads real conversation history and user facts
- Shows detailed supervisor planning output
- Displays agent results and speech generation
- Supports all registered agents (conversation_history, memory, web_search)

**Use Cases**:
- Debug supervisor planning logic
- Test agent coordination
- Analyze speech generation output
- Validate personality behavior
- Test new agent implementations

**Example Session**:
```
Supervisor + Speech debugger
Press Enter to use the shown default. Type quit to exit.

user_id [debug_user]: test_user_123
user_name [Debugger]: TestUser
channel_id [debug_channel]: test_channel
guild_id []: test_guild
bot_id [debug_bot]: kazusa_bot
default channel_topic [Debug session]: General Chat
default user_topic [Manual prompt debugging]: Testing memory recall

message_text: Do you remember the Python programming guide?
channel_topic [General Chat]: 
user_topic [Testing memory recall]: 

===== supervisor_plan =====
{
  "agents": ["memory_agent"],
  "instructions": {
    "memory_agent": {
      "command": "Recall stored memory about Python programming guide",
      "expected_response": "Return relevant Python programming information"
    }
  },
  "response_language": "English",
  "topics_to_cover": ["Python programming"],
  "facts_to_cover": ["If relevant memory exists, include key Python features"],
  "emotion_directive": "Helpful"
}

===== agent_results =====
[
  {
    "agent": "memory_agent",
    "status": "success",
    "summary": "Found Python programming memory with high relevance",
    "tool_history": [...]
  }
]

===== speech_brief =====
{
  "response_brief": {
    "response_language": "English",
    "topics_to_cover": ["Answer using Python programming details"],
    "facts_to_cover": ["Python is a high-level language with simple syntax"],
    "emotion_directive": "Helpful"
  }
}

===== response =====
I remember that Python is a high-level, interpreted programming language known for its simple, readable syntax. It supports multiple programming paradigms and is widely used for web development, data science, and machine learning.
```

---

## Environment Configuration

### Virtual Environment Setup

The project uses a virtual environment at `venv/` (not `.venv/`):

**Windows**:
```powershell
# Activate
venv\Scripts\activate

# Run scripts with virtual environment Python
& "venv\Scripts\python.exe" -m src.scripts.insert_memory "Test" "Content"
```

**Unix/Linux**:
```bash
# Activate
source venv/bin/activate

# Run scripts
python -m src.scripts.insert_memory "Test" "Content"
```

### Database Configuration

All scripts require MongoDB connection. Ensure your environment has:
- `MONGODB_URI`: MongoDB connection string
- `MONGODB_DB_NAME`: Database name
- `LLM_API_KEY`: API key for embeddings (for memory scripts)
- `LLM_BASE_URL`: Base URL for LLM service
- `LLM_MODEL`: Model name for embeddings

---

## Common Workflows

### Setting Up Memory Search

1. **Create vector index** (if not exists):
```bash
python -c "import asyncio; from kazusa_ai_chatbot.db import enable_memory_vector_index; asyncio.run(enable_memory_vector_index())"
```

2. **Insert sample memories**:
```bash
python -m src.scripts.insert_memory "Python Guide" "Python is a high-level programming language..."
python -m src.scripts.insert_memory "LangGraph" "LangGraph is a framework for building LLM applications..."
```

3. **Test search functionality**:
```bash
python -m src.scripts.search_memory "Python" --method vector
python -m src.scripts.search_memory "programming" --method keyword
```

### Debugging Agent Behavior

1. **Start the debug supervisor**:
```bash
python -m src.scripts.debug_supervisor_speech
```

2. **Test different scenarios**:
   - Memory recall: "Do you remember [topic]?"
   - Web search: "What's the latest news about [topic]?"
   - Conversation history: "What did we discuss earlier about [topic]?"

3. **Analyze the output**:
   - Check supervisor planning logic
   - Verify agent execution
   - Review speech generation

### Managing Embeddings

1. **Create conversation embeddings**:
```bash
python -m src.scripts.create_conversation_history_embedding
```

2. **Create user fact embeddings**:
```bash
python -m src.scripts.create_user_facts_embedding
```

3. **Verify search works**:
```bash
python -m src.scripts.search_conversation "topic" --method vector
python -m src.scripts.search_user_facts "user_id"
```

---

## Troubleshooting

### Common Issues

1. **Vector search returns no results**:
   - Ensure vector index exists: Run `enable_memory_vector_index()`
   - Check embeddings are generated for existing data
   - Verify query terms are relevant

2. **Scripts fail to connect to database**:
   - Check MongoDB connection string
   - Ensure MongoDB is accessible
   - Verify database name is correct

3. **Memory insertion fails**:
   - Check LLM service is running for embeddings
   - Verify API key and base URL are correct
   - Ensure content is not empty

4. **Debug supervisor shows errors**:
   - Check all agents are registered
   - Verify personality file exists and is valid JSON
   - Ensure MCP services are available for web search

### Getting Help

1. **Check logs**: All scripts provide detailed logging output
2. **Use debug mode**: The debug supervisor script provides comprehensive debugging information
3. **Verify environment**: Ensure all required environment variables are set
4. **Test components individually**: Use individual scripts to isolate issues

---

## Best Practices

1. **Use descriptive memory names**: Make memories easily searchable
2. **Test search methods**: Try both keyword and vector search for best results
3. **Monitor embedding generation**: Check logs for embedding API calls
4. **Use debug supervisor for complex scenarios**: Test multi-agent interactions
5. **Keep scripts updated**: Ensure scripts match current database schema
6. **Document custom memories**: Use clear, structured content for better recall
