# ============================================================
# HYBRID CSV INTELLIGENCE SYSTEM - LANGCHAIN COMPATIBLE
# Using LangChain Agents + Custom Tools + Groq LLM
# ============================================================

# STEP 1: Install Required Libraries
# ============================================================


# STEP 2: Import Libraries
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.tools import tool

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Dict, Any
import json
import warnings
warnings.filterwarnings('ignore')

# STEP 3: Global Variables
# ============================================================
df = None
collection = None
embedding_model = None
llm = None
chroma_client = None
conversation_history = []
DEBUG_MODE = True

# STEP 4: Initialize System
# ============================================================

def initialize_system(api_key: str):
    """Initialize LangChain with Groq LLM"""
    global embedding_model, llm, chroma_client
    
    print("🔧 Initializing LangChain system...")
    
    # Initialize Groq LLM
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=2000
    )
    print("✓ Groq LLM (Llama 3.3 70B) initialized")
    
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Embedding model loaded")
    
    # Initialize ChromaDB
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    print("✓ Vector database ready")
    
    print("✅ LangChain system initialized!\n")

# STEP 5: Vector Database Functions (RAG)
# ============================================================

def create_vector_db(dataframe: pd.DataFrame):
    """Create vector database from CSV"""
    global collection
    
    try:
        chroma_client.delete_collection(name="csv_data")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="csv_data",
        metadata={"hnsw:space": "cosine"}
    )
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in dataframe.iterrows():
        doc_text = " | ".join([f"{col}: {row[col]}" for col in dataframe.columns])
        documents.append(doc_text)
        metadatas.append({"row_index": idx})
        ids.append(f"row_{idx}")
    
    print(f"📊 Creating vector embeddings for {len(documents)} rows...")
    embeddings = embedding_model.encode(documents, show_progress_bar=False).tolist()
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✓ Vector database created\n")

def semantic_search(query: str, top_k: int = 5) -> str:
    """Perform semantic search and return results"""
    if collection is None:
        return "No data indexed yet"
    
    if DEBUG_MODE:
        print(f"\n🔍 [RAG] Searching for: {query}")
    
    query_embedding = embedding_model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    output = []
    for i in range(len(results['ids'][0])):
        row_idx = results['metadatas'][0][i]['row_index']
        doc = results['documents'][0][i]
        output.append(f"Row {row_idx}: {doc}")
    
    result = "\n".join(output)
    
    if DEBUG_MODE:
        print(f"✓ Found {len(output)} relevant rows\n")
    
    return result

# STEP 6: Custom Tool Functions
# ============================================================

def get_csv_info() -> str:
    """Get CSV dataset information"""
    if df is None:
        return "No CSV loaded"
    
    info = f"""Dataset Information:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Data Types: {df.dtypes.to_dict()}

First 3 rows:
{df.head(3).to_string()}

Basic Statistics:
{df.describe().to_string()}
"""
    return info

def execute_pandas_query(query: str) -> str:
    """Execute pandas operations using LangChain agent"""
    if df is None:
        return "No CSV loaded"
    
    if DEBUG_MODE:
        print(f"\n🐍 [Pandas Agent] Processing: {query}")
    
    try:
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=DEBUG_MODE,
            agent_type="zero-shot-react-description",
            allow_dangerous_code=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        result = pandas_agent.invoke(query)
        
        if DEBUG_MODE:
            print(f"✓ Pandas operation completed\n")
        
        if isinstance(result, dict):
            return str(result.get('output', result))
        return str(result)
        
    except Exception as e:
        return f"Error in pandas operation: {str(e)}"

def create_visualization(query: str) -> str:
    """Create visualizations from data"""
    if df is None:
        return "No CSV loaded"
    
    if DEBUG_MODE:
        print(f"\n📊 [Visualization] Creating: {query}")
    
    try:
        prompt = f"""Given this query: "{query}"

Available columns: {df.columns.tolist()}
Sample data: {df.head(2).to_dict()}

Generate Python code using plotly express to create the visualization.
Use variable 'df' for the dataframe and 'fig' for the plotly figure.
Code must use: import plotly.express as px
Then: fig = px.[chart_type]()
Then: fig.show()

Only return the Python code, nothing else."""

        code_response = llm.invoke(prompt)
        code = code_response.content
        
        # Clean the code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        code = code.strip()
        
        if DEBUG_MODE:
            print(f"Generated code:\n{code}\n")
        
        # Execute the visualization code
        exec_globals = {'df': df, 'px': px, 'plt': plt, 'np': np}
        exec(code, exec_globals)
        
        return f"Visualization created successfully!"
        
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

def statistical_analysis(query: str) -> str:
    """Perform statistical analysis"""
    if df is None:
        return "No CSV loaded"
    
    if DEBUG_MODE:
        print(f"\n📈 [Statistical Analysis] Analyzing: {query}")
    
    try:
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=DEBUG_MODE,
            agent_type="zero-shot-react-description",
            allow_dangerous_code=True,
            prefix="You are a data scientist. Perform statistical analysis on the dataframe."
        )
        
        result = pandas_agent.invoke(f"Perform statistical analysis: {query}")
        
        if isinstance(result, dict):
            return str(result.get('output', result))
        return str(result)
        
    except Exception as e:
        return f"Error in statistical analysis: {str(e)}"

# STEP 7: Smart Query Router (Agent-like Logic)
# ============================================================

def route_query(user_query: str) -> Dict[str, Any]:
    """Route query to appropriate tool"""
    
    if DEBUG_MODE:
        print("\n" + "🎯 " + "="*60)
        print("QUERY ROUTING")
        print("="*60)
        print(f"User Query: {user_query}")
        print("="*60)
    
    # Build decision prompt
    decision_prompt = f"""You are a query router. Analyze the user's question and decide which tool to use.

User Query: "{user_query}"

CSV Info:
- Columns: {df.columns.tolist() if df is not None else 'N/A'}
- Shape: {df.shape if df is not None else 'N/A'}

Available Tools:
1. csv_info - Dataset information (shape, columns, types, sample data)
2. semantic_search - Find specific rows using natural language search
3. pandas_query - Calculations, counting, filtering, aggregations, grouping
4. visualization - Create charts and graphs
5. statistical_analysis - Correlations, distributions, statistical tests

Respond ONLY with a JSON object:
{{
    "tool": "tool_name",
    "reasoning": "why this tool",
    "search_query": "refined query for the tool"
}}

Guidelines:
- Use csv_info for: "shape", "columns", "info", "dataset information"
- Use semantic_search for: "find", "show me", "rows with", "passengers named"
- Use pandas_query for: "count", "how many", "average", "sum", "filter", "group by"
- Use visualization for: "chart", "plot", "graph", "visualize"
- Use statistical_analysis for: "correlation", "distribution", "outliers", "statistics"

Respond ONLY with valid JSON, no markdown."""

    try:
        response = llm.invoke(decision_prompt)
        decision_text = response.content
        
        # Extract JSON
        import re
        json_match = re.search(r'\{.*\}', decision_text, re.DOTALL)
        if json_match:
            decision_text = json_match.group()
        
        decision = json.loads(decision_text)
        
        if DEBUG_MODE:
            print(f"\n✅ ROUTING DECISION")
            print("="*60)
            print(f"Tool: {decision.get('tool', 'N/A')}")
            print(f"Reasoning: {decision.get('reasoning', 'N/A')}")
            print("="*60 + "\n")
        
        return decision
        
    except Exception as e:
        if DEBUG_MODE:
            print(f"\n⚠️ Routing failed, defaulting to pandas_query: {str(e)}\n")
        return {"tool": "pandas_query", "search_query": user_query}

def execute_tool(tool_name: str, query: str) -> str:
    """Execute the selected tool"""
    
    tool_map = {
        "csv_info": get_csv_info,
        "semantic_search": semantic_search,
        "pandas_query": execute_pandas_query,
        "visualization": create_visualization,
        "statistical_analysis": statistical_analysis
    }
    
    tool_func = tool_map.get(tool_name)
    
    if tool_func:
        if tool_name == "csv_info":
            return tool_func()
        else:
            return tool_func(query)
    else:
        return f"Unknown tool: {tool_name}"

def process_query(user_query: str) -> str:
    """Main query processing function"""
    global conversation_history
    
    conversation_history.append({"role": "user", "content": user_query})
    
    # Route query to appropriate tool
    decision = route_query(user_query)
    tool_name = decision.get("tool", "pandas_query")
    search_query = decision.get("search_query", user_query)
    
    # Execute tool
    tool_result = execute_tool(tool_name, search_query)
    
    # Generate natural language response
    if DEBUG_MODE:
        print("💬 Generating final answer...")
    
    history_context = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in conversation_history[-5:]
    ])
    
    final_prompt = f"""Based on the tool output, provide a clear, concise answer to the user's question.

User Question: {user_query}
Tool Used: {tool_name}
Tool Output:
{tool_result}

Conversation History:
{history_context}

Provide a natural language answer. Be specific and clear."""

    try:
        response = llm.invoke(final_prompt)
        answer = response.content
    except Exception as e:
        answer = f"Tool Result: {tool_result}"
    
    conversation_history.append({"role": "assistant", "content": answer})
    
    return answer

# STEP 8: CSV Management
# ============================================================

def load_csv_file(file_path: str = None):
    """Load CSV and create vector database"""
    global df
    
    if file_path is None:
        print("📁 Please provide CSV file path:")
        file_path = input("Enter CSV file path: ").strip()
        if not file_path:
            print("❌ No file path provided")
            return None
    
    print(f"\n📂 Loading: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Columns: {', '.join(df.columns)}\n")
    
    create_vector_db(df)
    
    print("✅ CSV loaded and ready!\n")
    return df

def show_csv_info():
    """Display CSV information"""
    if df is None:
        print("❌ No CSV loaded.\n")
        return
    
    print("\n" + "="*70)
    print("📊 CSV INFORMATION")
    print("="*70)
    print(get_csv_info())
    print("="*70 + "\n")

# STEP 9: CLI Interface
# ============================================================

def print_welcome():
    """Print welcome message"""
    print("\n" + "="*70)
    print("🤖 LANGCHAIN CSV INTELLIGENCE SYSTEM")
    print("="*70)
    print("LangChain Tools + Groq (Llama 3.3 70B) + RAG")
    print("="*70)
    print("\n📋 COMMANDS:")
    print("  load     - Upload CSV file")
    print("  info     - Show CSV info")
    print("  tools    - List available tools")
    print("  debug    - Toggle debug mode")
    print("  clear    - Clear conversation")
    print("  quit     - Exit")
    print("\n💬 Ask questions about your data!")
    print("="*70 + "\n")

def show_tools():
    """Show available tools"""
    print("\n" + "="*70)
    print("🛠️  AVAILABLE TOOLS")
    print("="*70)
    print("\n1. 📊 csv_info - Dataset information")
    print("2. 🔍 semantic_search - Find rows by natural language")
    print("3. 🐍 pandas_query - Calculations & aggregations")
    print("4. 📈 visualization - Create charts")
    print("5. 📉 statistical_analysis - Statistical insights")
    print("\n" + "="*70 + "\n")

def start_chatbot():
    """Start the CLI chatbot"""
    print_welcome()
    
    if df is None:
        print("⚠️  No CSV loaded.\n")
        load_response = input("Load CSV now? (yes/no): ").strip().lower()
        if load_response in ['yes', 'y']:
            load_csv_file()
    
    print("🟢 System ready!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            elif user_input.lower() == 'load':
                load_csv_file()
                continue
            
            elif user_input.lower() == 'info':
                show_csv_info()
                continue
            
            elif user_input.lower() == 'tools':
                show_tools()
                continue
            
            elif user_input.lower() in ['clear', 'reset']:
                conversation_history.clear()
                print("✓ Conversation cleared\n")
                continue
            
            elif user_input.lower() == 'debug':
                global DEBUG_MODE
                DEBUG_MODE = not DEBUG_MODE
                print(f"\n🔧 Debug mode: {'ON' if DEBUG_MODE else 'OFF'}\n")
                continue
            
            elif user_input.lower() in ['help', '?']:
                print_welcome()
                continue
            
            if df is None:
                print("❌ Please load a CSV first\n")
                continue
            
            print("\n🤔 Processing...\n")
            
            try:
                answer = process_query(user_input)
                print(f"\n🤖 Assistant: {answer}\n")
                print("-" * 70 + "\n")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted.\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

# STEP 10: Main
# ============================================================

def main():
    print("\n" + "="*70)
    print("🚀 LANGCHAIN CSV INTELLIGENCE SYSTEM")
    print("="*70 + "\n")
    
    print("🔑 Get FREE API key: https://console.groq.com/keys")
    api_key = input("Enter Groq API Key: ").strip()
    
    if not api_key:
        print("\n❌ API key required!\n")
        return
    
    print()
    
    try:
        initialize_system(api_key)
        start_chatbot()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")

main()