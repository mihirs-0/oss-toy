import streamlit as st
import pandas as pd
import ollama
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Page config
st.set_page_config(
    page_title="Supply Chain Intelligence Demo",
    page_icon="🚚",
    layout="wide"
)

# Initialize session state
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = None
if 'data_modified' not in st.session_state:
    st.session_state.data_modified = False

@st.cache_data
def load_supply_chain_data():
    """Load mock supply chain data"""
    
    # Items data
    items_data = {
        'item_id': ['ITM001', 'ITM002', 'ITM003', 'ITM004', 'ITM005', 'ITM006', 'ITM007', 'ITM008'],
        'item_name': ['Laptop Computer', 'Wireless Mouse', 'USB Cable', 'Monitor Stand', 
                     'Keyboard', 'Webcam', 'Headphones', 'Docking Station'],
        'category': ['Electronics', 'Accessories', 'Cables', 'Furniture', 
                    'Accessories', 'Electronics', 'Electronics', 'Electronics'],
        'unit_cost': [899.99, 29.99, 12.99, 45.99, 79.99, 89.99, 129.99, 199.99],
        'supplier': ['TechCorp', 'AccessoryPlus', 'CableCo', 'OfficeFurn', 
                    'AccessoryPlus', 'TechCorp', 'AudioTech', 'TechCorp']
    }
    
    # Warehouses data
    warehouses_data = {
        'warehouse_id': ['WH001', 'WH002', 'WH003'],
        'location': ['New York', 'Los Angeles', 'Chicago'],
        'capacity': [10000, 8000, 12000],
        'current_utilization': [0.75, 0.82, 0.68]
    }
    
    # Inventory data
    inventory_data = {
        'warehouse_id': ['WH001', 'WH001', 'WH001', 'WH002', 'WH002', 'WH002', 'WH003', 'WH003'],
        'item_id': ['ITM001', 'ITM002', 'ITM003', 'ITM004', 'ITM005', 'ITM006', 'ITM007', 'ITM008'],
        'current_stock': [45, 120, 200, 30, 85, 25, 60, 15],
        'reorder_point': [50, 100, 150, 40, 80, 30, 50, 20],
        'max_stock': [200, 500, 800, 150, 300, 100, 200, 80],
        'last_restock_date': ['2024-01-15', '2024-01-20', '2024-01-18', '2024-01-22', 
                             '2024-01-19', '2024-01-16', '2024-01-21', '2024-01-17']
    }
    
    # Deliveries data
    deliveries_data = {
        'delivery_id': ['DEL001', 'DEL002', 'DEL003', 'DEL004', 'DEL005'],
        'item_id': ['ITM001', 'ITM002', 'ITM005', 'ITM007', 'ITM008'],
        'warehouse_id': ['WH001', 'WH002', 'WH003', 'WH001', 'WH002'],
        'quantity': [25, 50, 30, 20, 10],
        'delivery_date': ['2024-01-25', '2024-01-26', '2024-01-27', '2024-01-28', '2024-01-29'],
        'status': ['Completed', 'In Transit', 'Scheduled', 'Completed', 'In Transit'],
        'supplier': ['TechCorp', 'AccessoryPlus', 'AccessoryPlus', 'AudioTech', 'TechCorp']
    }
    
    return {
        'items': pd.DataFrame(items_data),
        'warehouses': pd.DataFrame(warehouses_data),
        'inventory': pd.DataFrame(inventory_data),
        'deliveries': pd.DataFrame(deliveries_data)
    }

def get_supply_chain_ontology():
    """Define the supply chain ontology for context"""
    return """
    SUPPLY CHAIN ONTOLOGY:
    
    ENTITIES:
    - Items: Physical products with ID, name, category, cost, and supplier
    - Warehouses: Storage facilities with location, capacity, and utilization
    - Inventory: Current stock levels per item per warehouse with reorder points
    - Deliveries: Shipments with status, quantities, and dates
    - Suppliers: Companies that provide items
    
    RELATIONSHIPS:
    - Items are stored in Warehouses (via Inventory)
    - Items are supplied by Suppliers
    - Deliveries transport Items to Warehouses
    - Inventory tracks stock levels and reorder points
    - Warehouses have capacity constraints
    
    KEY METRICS:
    - Stock levels vs reorder points (for restocking decisions)
    - Warehouse utilization (for capacity planning)
    - Delivery status (for supply chain visibility)
    - Cost analysis (for financial optimization)
    
    BUSINESS RULES:
    - Restock when current_stock <= reorder_point
    - Monitor warehouse capacity utilization
    - Track delivery delays and supplier performance
    - Optimize inventory costs and stock levels
    """

@st.cache_resource
def load_model():
    """Load the OpenAI GPT-OSS-20B model via Ollama"""
    try:
        model_name = "gpt-oss:20b"  # Ollama model name
        
        st.info(f"🔄 Loading {model_name} via Ollama...")
        st.info("🎉 Using OpenAI's GPT-OSS-20B running locally with Ollama!")
        
        with st.spinner("Connecting to Ollama and verifying model..."):
            # Check if Ollama is running and model is available
            try:
                # Test connection to Ollama
                models = ollama.list()
                available_models = [model['name'] for model in models['models']]
                
                if model_name not in available_models:
                    st.error(f"❌ Model {model_name} not found in Ollama")
                    st.info("💡 Make sure you've pulled the model with: `ollama pull gpt-oss:20b`")
                    return None, None
                
                # Test the model with a simple prompt (quick test)
                st.info("🧪 Testing GPT-OSS model responsiveness...")
                test_response = ollama.generate(
                    model=model_name,
                    prompt="Hi",
                    options={
                        'num_predict': 5,
                        'temperature': 0.1
                    }
                )
                
                st.info("✅ Ollama connection successful")
                st.info("✅ GPT-OSS-20B model verified and ready")
                
                st.success(f"✅ Successfully connected to OpenAI GPT-OSS-20B via Ollama")
                st.info("🔬 Model ready for advanced supply chain analysis!")
                
                return model_name, None  # Return model name instead of loaded model object
                
            except Exception as ollama_error:
                st.error(f"❌ Ollama connection failed: {str(ollama_error)}")
                st.info("💡 Make sure Ollama is running: `brew services start ollama`")
                st.info("💡 And that you've pulled the model: `ollama pull gpt-oss:20b`")
                return None, None
            
    except Exception as e:
        st.error(f"❌ Error setting up Ollama GPT-OSS model: {str(e)}")
        st.warning("🔄 Falling back to rule-based analysis")
        
        # Show detailed error information
        error_str = str(e).lower()
        if "connection" in error_str or "timeout" in error_str:
            st.info("💡 Ollama service may not be running. Try: `brew services start ollama`")
        elif "not found" in error_str:
            st.info("💡 Model not found. Try: `ollama pull gpt-oss:20b`")
        else:
            st.info("💡 Unexpected error. The app will continue with rule-based analysis.")
            
        return None, None

def extract_relevant_data(query, data):
    """Extract relevant data based on query keywords"""
    query_lower = query.lower()
    relevant_data = {}
    
    # Keywords for different data types
    inventory_keywords = ['stock', 'inventory', 'reorder', 'restock', 'shortage', 'low']
    delivery_keywords = ['delivery', 'shipment', 'transit', 'supplier', 'schedule']
    warehouse_keywords = ['warehouse', 'capacity', 'utilization', 'location']
    item_keywords = ['item', 'product', 'laptop', 'mouse', 'cable', 'monitor', 'keyboard']
    
    # Check which data to include
    if any(keyword in query_lower for keyword in inventory_keywords):
        relevant_data['inventory'] = data['inventory']
        relevant_data['items'] = data['items']
    
    if any(keyword in query_lower for keyword in delivery_keywords):
        relevant_data['deliveries'] = data['deliveries']
        relevant_data['items'] = data['items']
    
    if any(keyword in query_lower for keyword in warehouse_keywords):
        relevant_data['warehouses'] = data['warehouses']
    
    if any(keyword in query_lower for keyword in item_keywords):
        relevant_data['items'] = data['items']
    
    # If no specific keywords, include all data
    if not relevant_data:
        relevant_data = data
    
    return relevant_data

def generate_prompt(query, relevant_data, ontology):
    """Generate an enhanced prompt for the OpenAI GPT-OSS-20B model using Harmony format"""
    
    # Create structured data summary
    data_summary = ""
    insights = []
    
    for data_type, df in relevant_data.items():
        data_summary += f"\n=== {data_type.upper()} DATA ===\n"
        
        if data_type == 'inventory':
            # Add inventory-specific insights
            low_stock = df[df['current_stock'] <= df['reorder_point']]
            if not low_stock.empty:
                insights.append(f"⚠️ {len(low_stock)} items below reorder point")
            
            total_value = (df['current_stock'] * df.merge(relevant_data.get('items', df), on='item_id', how='left')['unit_cost'].fillna(0)).sum()
            insights.append(f"💰 Total inventory value: ${total_value:,.2f}")
            
        elif data_type == 'deliveries':
            # Add delivery-specific insights
            pending = df[df['status'] != 'Completed']
            if not pending.empty:
                insights.append(f"🚛 {len(pending)} pending deliveries")
            
        elif data_type == 'warehouses':
            # Add warehouse-specific insights
            high_util = df[df['current_utilization'] > 0.8]
            if not high_util.empty:
                insights.append(f"🏭 {len(high_util)} warehouses at >80% capacity")
        
        # Add formatted data
        data_summary += df.to_string(index=False, max_rows=20)
        data_summary += "\n"
    
    # Create the enhanced prompt using OpenAI's Harmony format
    # Note: This is a simplified version - the full Harmony format is more complex
    prompt = f"""You are an expert supply chain analyst with deep knowledge of inventory management, logistics, and operational optimization.

SUPPLY CHAIN CONTEXT:
{ontology}

CURRENT DATA SNAPSHOT:
{data_summary}

KEY INSIGHTS DETECTED:
{' | '.join(insights) if insights else 'No critical issues detected'}

USER QUERY: {query}

Please provide a comprehensive supply chain analysis that includes:

1. **Direct Answer**: Address the specific question asked
2. **Key Findings**: Highlight critical issues, trends, or opportunities 
3. **Actionable Recommendations**: Specific steps to improve operations
4. **Risk Assessment**: Potential problems and mitigation strategies
5. **Performance Metrics**: Relevant KPIs and benchmarks

Format your response with clear sections and bullet points for easy reading."""
    
    return prompt

def generate_response(query, data, model_name, tokenizer):
    """Generate response using the OpenAI GPT-OSS-20B model via Ollama"""
    if model_name is None:
        st.info("🤖 Using rule-based analysis (no AI model loaded)")
        return generate_fallback_response(query, data)
    
    try:
        st.info("🧠 AI Model Analysis in Progress...")
        
        with st.spinner("🔍 Analyzing supply chain data with OpenAI GPT-OSS-20B via Ollama..."):
            ontology = get_supply_chain_ontology()
            relevant_data = extract_relevant_data(query, data)
            prompt = generate_prompt(query, relevant_data, ontology)
            
            # Show some debug info
            st.info(f"📊 Processing {len(relevant_data)} data sources")
            st.info("⚡ Using GPT-OSS-20B local inference")
            
            # Create a progress container for streaming
            progress_container = st.empty()
            response_container = st.empty()
            
            st.info("🎯 Generating AI insights with GPT-OSS...")
            
            # Use streaming for better user experience
            full_response = ""
            try:
                stream = ollama.generate(
                    model=model_name,
                    prompt=prompt,
                    options={
                        'num_predict': 400,      # Max tokens to generate
                        'temperature': 0.3,      # Lower temperature for focused responses
                        'top_p': 0.9,           # Nucleus sampling
                        'repeat_penalty': 1.2,   # Reduce repetition
                        'stop': ['\n\n\n', '---', 'END'],  # Stop sequences
                        'num_ctx': 4096,        # Context window
                    },
                    stream=True
                )
                
                # Stream the response
                for chunk in stream:
                    if chunk.get('response'):
                        full_response += chunk['response']
                        # Update the display every few tokens for better UX
                        if len(full_response) % 10 == 0:
                            response_container.markdown(f"**Generating...** {full_response}")
                    
                    if chunk.get('done', False):
                        break
                
                progress_container.empty()
                response_container.empty()
                
            except Exception as stream_error:
                st.warning(f"⚠️ Streaming failed: {stream_error}")
                st.info("🔄 Trying non-streaming approach...")
                
                # Fallback to non-streaming with timeout
                response = ollama.generate(
                    model=model_name,
                    prompt=prompt,
                    options={
                        'num_predict': 300,      # Shorter for faster response
                        'temperature': 0.3,      
                        'top_p': 0.9,           
                        'repeat_penalty': 1.2,   
                        'stop': ['\n\n\n', '---', 'END']
                    }
                )
                full_response = response['response']
        
        # Clean up the response
        ai_response = full_response.strip()
        
        # If response is too short or empty, fall back to rule-based
        if len(ai_response.split()) < 5:
            st.warning("⚠️ GPT-OSS response was too short, using enhanced rule-based analysis")
            return generate_fallback_response(query, data)
        
        st.success("✅ GPT-OSS Analysis Complete!")
        return f"## 🤖 AI-Powered Analysis (OpenAI GPT-OSS-20B)\n\n{ai_response}"
    
    except Exception as e:
        st.error(f"❌ Error during GPT-OSS generation: {str(e)}")
        st.info("🔄 Falling back to rule-based analysis...")
        return generate_fallback_response(query, data)

def generate_fallback_response(query, data):
    """Generate a rule-based response when model is not available"""
    query_lower = query.lower()
    
    # Analyze inventory for low stock
    inventory_df = data['inventory'].merge(data['items'], on='item_id')
    low_stock_items = inventory_df[inventory_df['current_stock'] <= inventory_df['reorder_point']]
    
    response = "## Supply Chain Analysis\n\n"
    
    if 'stock' in query_lower or 'inventory' in query_lower or 'reorder' in query_lower:
        if not low_stock_items.empty:
            response += "### 🚨 Low Stock Alert\n"
            response += "The following items need immediate restocking:\n"
            for _, item in low_stock_items.iterrows():
                response += f"- **{item['item_name']}** (ID: {item['item_id']}): {item['current_stock']} units (below reorder point of {item['reorder_point']})\n"
            
            response += "\n### 📋 Recommended Actions:\n"
            for _, item in low_stock_items.iterrows():
                recommended_order = item['max_stock'] - item['current_stock']
                response += f"- Order {recommended_order} units of {item['item_name']} from {item['supplier']}\n"
        else:
            response += "### ✅ Stock Levels\nAll items are currently above their reorder points.\n"
    
    if 'delivery' in query_lower or 'shipment' in query_lower:
        pending_deliveries = data['deliveries'][data['deliveries']['status'] != 'Completed']
        if not pending_deliveries.empty:
            response += "\n### 🚛 Pending Deliveries\n"
            for _, delivery in pending_deliveries.iterrows():
                response += f"- {delivery['delivery_id']}: {delivery['quantity']} units to {delivery['warehouse_id']} - Status: {delivery['status']}\n"
    
    if 'warehouse' in query_lower or 'capacity' in query_lower:
        response += "\n### 🏭 Warehouse Utilization\n"
        for _, warehouse in data['warehouses'].iterrows():
            utilization_pct = warehouse['current_utilization'] * 100
            status = "🔴 High" if utilization_pct > 80 else "🟡 Medium" if utilization_pct > 60 else "🟢 Normal"
            response += f"- **{warehouse['location']}** ({warehouse['warehouse_id']}): {utilization_pct:.1f}% - {status}\n"
    
    return response

def apply_restock_action(item_id, quantity, data):
    """Simulate applying a restock action"""
    # Find the item in inventory
    inventory_idx = data['inventory'][data['inventory']['item_id'] == item_id].index
    if not inventory_idx.empty:
        data['inventory'].loc[inventory_idx[0], 'current_stock'] += quantity
        return True
    return False

# Main App
def main():
    st.title("🚚 Supply Chain Intelligence Demo")
    st.markdown("**Powered by OpenAI GPT-OSS-20B** | Advanced supply chain analytics with OpenAI's latest open-source model")
    
    # Load data
    data = load_supply_chain_data()
    
    # Sidebar for model status
    with st.sidebar:
        st.header("🤖 AI Model Status")
        
        # Model loading section
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🚀 Load OpenAI GPT-OSS-20B", type="primary", use_container_width=True):
                with st.spinner("Loading OpenAI GPT-OSS-20B model..."):
                    model_name, tokenizer = load_model()
                    st.session_state.ollama_model = model_name
                    if model_name is not None:
                        st.balloons()  # Celebrate successful loading!
        
        with col2:
            if st.button("🔄", help="Refresh model status"):
                st.rerun()
        
        # Model status display
        if st.session_state.ollama_model is not None:
            st.success("✅ **OpenAI GPT-OSS-20B Loaded**")
            st.info("🧠 **AI-Powered Analysis Active**")
            
            # Show model capabilities
            with st.expander("🔬 Model Capabilities"):
                st.markdown("""
                - **20B Parameters**: Advanced reasoning
                - **Supply Chain Expert**: Domain-specific insights
                - **Harmony Format**: Latest OpenAI technology
                - **Real-time Analysis**: Interactive responses
                """)
        else:
            st.warning("⚠️ **AI Model Not Loaded**")
            st.info("🔄 **Using Rule-Based Analysis**")
            
            # Show what's missing
            with st.expander("💡 Why Load the AI Model?"):
                st.markdown("""
                **AI Model Benefits:**
                - 🧠 Deep contextual understanding
                - 🎯 Sophisticated recommendations  
                - 🔍 Pattern recognition across data
                - 📊 Strategic business insights
                - 🚀 Cutting-edge OpenAI technology
                
                **vs Rule-Based Analysis:**
                - ⚙️ Basic pattern matching
                - 📋 Template responses
                - 🔢 Simple calculations
                """)
        
        st.divider()
        
        st.header("📊 Quick Stats")
        total_items = len(data['items'])
        total_warehouses = len(data['warehouses'])
        low_stock_count = len(data['inventory'][data['inventory']['current_stock'] <= data['inventory']['reorder_point']])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Items", total_items)
            st.metric("Warehouses", total_warehouses)
        with col2:
            st.metric("Low Stock", low_stock_count, 
                     delta=f"-{low_stock_count}" if low_stock_count > 0 else "0",
                     delta_color="inverse")
        
        # System info
        st.divider()
        st.header("💻 System Info")
        
        # Check if Ollama is running
        try:
            models = ollama.list()
            st.success("🚀 Ollama: Running")
            st.info(f"📦 Models: {len(models['models'])} available")
        except:
            st.error("❌ Ollama: Not running")
            st.info("💡 Start with: `brew services start ollama`")
        
        # Show CPU info (since we're running on CPU)
        import platform
        st.info(f"🖥️ CPU: {platform.processor() or platform.machine()}")
        st.info(f"🐍 Python: {platform.python_version()}")
    
    # Main query interface
    st.header("💬 Ask About Your Supply Chain")
    
    # Sample queries
    with st.expander("💡 Sample Queries"):
        sample_queries = [
            "Which items need restocking and what's the financial impact?",
            "Analyze warehouse capacity utilization and recommend optimization strategies",
            "What deliveries are pending and how might delays affect operations?",
            "Compare inventory levels across categories and identify trends",
            "Which suppliers have performance issues and what are the risks?",
            "Provide a comprehensive supply chain health assessment",
            "What are the top 3 operational risks I should address immediately?",
            "How can I optimize inventory costs while maintaining service levels?"
        ]
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}"):
                st.session_state.user_query = query
    
    # Query input
    query_container = st.container()
    with query_container:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_query = st.text_input(
                "Enter your question:",
                placeholder="e.g., Which items need restocking and what's the financial impact?",
                value=st.session_state.get('user_query', ''),
                help="Ask complex questions - the AI model can provide sophisticated analysis!"
            )
        
        with col2:
            analysis_type = "🤖 AI Analysis" if st.session_state.ollama_model is not None else "⚙️ Rule-Based"
            st.metric("Mode", analysis_type)
    
    # Analysis button with dynamic text
    button_text = "🚀 Analyze with AI" if st.session_state.ollama_model is not None else "📊 Analyze with Rules"
    button_type = "primary" if st.session_state.ollama_model is not None else "secondary"
    
    if st.button(button_text, type=button_type, use_container_width=True) or user_query:
        if user_query:
            # Clear any previous session state for fresh analysis
            if 'analysis_complete' in st.session_state:
                del st.session_state.analysis_complete
            
            # Show analysis type banner
            if st.session_state.ollama_model is not None:
                st.info("🤖 **AI-Powered Analysis** - Using OpenAI GPT-OSS-20B for sophisticated insights")
            else:
                st.info("⚙️ **Rule-Based Analysis** - Using predefined logic (Load AI model for advanced insights)")
            
            # Generate response with progress tracking
            response = generate_response(
                user_query, 
                data, 
                st.session_state.ollama_model, 
                None  # No tokenizer needed for Ollama
            )
            
            st.markdown("### 📋 Analysis Results")
            st.markdown(response)
            
            # Mark analysis as complete
            st.session_state.analysis_complete = True
            
            # Show actionable buttons for restock recommendations
            if 'restock' in response.lower() or 'order' in response.lower():
                st.markdown("### 🎯 Quick Actions")
                
                # Find items that need restocking
                inventory_df = data['inventory'].merge(data['items'], on='item_id')
                low_stock_items = inventory_df[inventory_df['current_stock'] <= inventory_df['reorder_point']]
                
                if not low_stock_items.empty:
                    st.info(f"Found {len(low_stock_items)} items that need restocking")
                    
                    cols = st.columns(min(3, len(low_stock_items)))
                    for idx, (_, item) in enumerate(low_stock_items.iterrows()):
                        with cols[idx % 3]:
                            recommended_qty = item['max_stock'] - item['current_stock']
                            if st.button(f"📦 Restock {item['item_name']}", key=f"restock_{item['item_id']}"):
                                if apply_restock_action(item['item_id'], recommended_qty, data):
                                    st.success(f"✅ Restocked {recommended_qty} units of {item['item_name']}")
                                    st.rerun()
            
            # Show follow-up suggestions
            if st.session_state.ollama_model is not None:
                st.markdown("### 💡 Follow-up Questions")
                follow_ups = [
                    "What are the financial implications of these recommendations?",
                    "How do these issues compare to industry benchmarks?", 
                    "What preventive measures should we implement?",
                    "Show me a risk assessment for the next quarter"
                ]
                
                cols = st.columns(2)
                for idx, follow_up in enumerate(follow_ups):
                    with cols[idx % 2]:
                        if st.button(follow_up, key=f"followup_{idx}"):
                            st.session_state.user_query = follow_up
                            st.rerun()
    
    # Data visualization
    st.header("📊 Supply Chain Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["📦 Inventory Status", "🏭 Warehouse Utilization", "🚛 Delivery Tracking"])
    
    with tab1:
        # Inventory status chart
        inventory_df = data['inventory'].merge(data['items'], on='item_id')
        
        fig = px.bar(
            inventory_df,
            x='item_name',
            y=['current_stock', 'reorder_point'],
            title="Current Stock vs Reorder Points",
            barmode='group'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed inventory table
        st.subheader("Detailed Inventory")
        display_df = inventory_df[['item_name', 'warehouse_id', 'current_stock', 'reorder_point', 'max_stock']].copy()
        display_df['Status'] = display_df.apply(
            lambda row: '🔴 Low Stock' if row['current_stock'] <= row['reorder_point'] else '✅ Normal',
            axis=1
        )
        st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        # Warehouse utilization
        fig = px.bar(
            data['warehouses'],
            x='location',
            y='current_utilization',
            title="Warehouse Utilization by Location",
            color='current_utilization',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(yaxis_title="Utilization Rate")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(data['warehouses'], use_container_width=True)
    
    with tab3:
        # Delivery status
        delivery_counts = data['deliveries']['status'].value_counts()
        fig = px.pie(
            values=delivery_counts.values,
            names=delivery_counts.index,
            title="Delivery Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Delivery Details")
        delivery_display = data['deliveries'].merge(data['items'], on='item_id')
        st.dataframe(delivery_display[['delivery_id', 'item_name', 'warehouse_id', 'quantity', 'delivery_date', 'status']], use_container_width=True)

if __name__ == "__main__":
    main()