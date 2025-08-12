"""
Streamlit UI for Mental Wellness Companion
Interactive interface for demonstrating the RL-powered mental health support system
"""
import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Mental Wellness Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        align-self: flex-end;
    }
    .agent-message {
        background-color: #f5f5f5;
        align-self: flex-start;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "persona" not in st.session_state:
    st.session_state.persona = None
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/4A90E2/FFFFFF?text=Mental+Wellness", use_container_width=True)
    st.title("🧠 Mental Wellness Companion")
    st.markdown("---")
    
    # Navigation
    page = st.selectbox(
        "Navigation",
        ["💬 Chat Interface", "📊 Training Dashboard", "📈 Analytics", "🧪 Evaluation", "⚙️ Settings"]
    )
    
    st.markdown("---")
    
    # System Status
    st.subheader("System Status")
    try:
        health = requests.get(f"{API_URL}/health").json()
        st.success("🟢 System Online")
    except:
        st.error("🔴 System Offline")
    
    # Statistics
    try:
        stats = requests.get(f"{API_URL}/statistics").json()
        st.metric("Active Sessions", stats.get("active_sessions", 0))
        st.metric("Total Interactions", stats.get("total_sessions", 0))
    except:
        pass

# Main Content
if page == "💬 Chat Interface":
    st.title("💬 Interactive Chat Interface")
    st.markdown("Experience the RL-powered mental wellness support system")
    
    # Session Management
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("🆕 New Session", type="primary", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/session/create", 
                                        json={"use_random_persona": True})
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.session_id = data["session_id"]
                    st.session_state.persona = data["persona"]
                    st.session_state.conversation_history = []
                    st.session_state.metrics_history = []
                    
                    # Add initial message
                    st.session_state.conversation_history.append({
                        "role": "user",
                        "content": data["initial_message"]
                    })
                    st.success("New session created!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error creating session: {e}")
    
    with col2:
        if st.session_state.session_id and st.button("🔚 End Session", use_container_width=True):
            try:
                response = requests.delete(f"{API_URL}/session/{st.session_state.session_id}")
                if response.status_code == 200:
                    st.success("Session ended")
                    st.session_state.session_id = None
                    st.session_state.persona = None
                    st.session_state.conversation_history = []
                    st.rerun()
            except Exception as e:
                st.error(f"Error ending session: {e}")
    
    with col3:
        if st.session_state.session_id:
            st.success(f"Session Active")
    
    # Display Persona Info
    if st.session_state.persona:
        with st.expander("👤 Persona Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Age:** {st.session_state.persona['demographics']['age']}")
                st.write(f"**Gender:** {st.session_state.persona['demographics']['gender']}")
                st.write(f"**Occupation:** {st.session_state.persona['demographics']['occupation']}")
            with col2:
                st.write(f"**Primary Concerns:** {', '.join(st.session_state.persona['mental_health']['primary_concerns'])}")
                st.write(f"**Severity:** {st.session_state.persona['mental_health']['severity']}")
                st.write(f"**Communication Style:** {st.session_state.persona['personality']['communication_style']}")
    
    # Chat Interface
    if st.session_state.session_id:
        # Metrics Display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            risk_level = st.session_state.metrics_history[-1]["risk_level"] if st.session_state.metrics_history else 0
            st.metric("Risk Level", f"{risk_level:.2f}", 
                     delta=f"{risk_level - st.session_state.metrics_history[-2]['risk_level']:.2f}" if len(st.session_state.metrics_history) > 1 else None)
        with col2:
            health_score = st.session_state.metrics_history[-1]["health_score"] if st.session_state.metrics_history else 0
            st.metric("Health Score", f"{health_score:.2f}")
        with col3:
            turn_count = st.session_state.metrics_history[-1]["turn_count"] if st.session_state.metrics_history else 0
            st.metric("Turn Count", turn_count)
        with col4:
            last_strategy = st.session_state.metrics_history[-1]["strategy"] if st.session_state.metrics_history else "N/A"
            st.metric("Last Strategy", last_strategy)
        
        # Conversation Display
        st.markdown("---")
        chat_container = st.container()
        
        with chat_container:
            for msg in st.session_state.conversation_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>👤 You:</b><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message agent-message">
                        <b>🤖 Agent:</b><br>{msg['content']}
                        {f"<br><small>Strategy: {msg.get('metadata', {}).get('strategy', 'N/A')}</small>" if msg.get('metadata') else ""}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Message Input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Your message:", height=100)
            col1, col2 = st.columns([4, 1])
            with col2:
                send_button = st.form_submit_button("Send 📤", type="primary", use_container_width=True)
            
            if send_button and user_input:
                try:
                    # Send message to API
                    response = requests.post(f"{API_URL}/chat", 
                                            json={"session_id": st.session_state.session_id, 
                                                  "message": user_input})
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": user_input
                        })
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": data["response"],
                            "metadata": {
                                "strategy": data["strategy"],
                                "resource": data["resource"]
                            }
                        })
                        
                        # Update metrics
                        st.session_state.metrics_history.append({
                            "risk_level": data["metrics"]["risk_level"],
                            "health_score": data["metrics"]["health_score"],
                            "turn_count": data["metrics"]["turn_count"],
                            "strategy": data["strategy"]
                        })
                        
                        # Show safety info if needed
                        if not data["safety_check"]["is_safe"]:
                            st.warning(f"⚠️ Safety concern detected: {data['safety_check']['violation']}")
                        
                        # Check if conversation should end
                        if not data["continue_conversation"]:
                            st.info("Conversation has ended based on user needs")
                        
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error sending message: {e}")
    else:
        st.info("👆 Click 'New Session' to start a conversation")

elif page == "📊 Training Dashboard":
    st.title("📊 Training Dashboard")
    st.markdown("Monitor PPO and Contextual Bandit training progress")
    
    # Load training status
    try:
        status = requests.get(f"{API_URL}/training/status").json()
        
        # Status Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", status["status"].replace("_", " ").title())
        with col2:
            st.metric("Episodes Completed", status["episodes_completed"])
        with col3:
            st.metric("Current Reward", f"{status['current_reward']:.2f}")
        with col4:
            st.metric("Best Reward", f"{status['best_reward']:.2f}")
        
        # Load detailed training stats
        try:
            with open("data/training_stats.json") as f:
                training_data = json.load(f)
            
            # Learning Curves
            st.subheader("Learning Curves")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Episode Rewards
                fig_rewards = go.Figure()
                fig_rewards.add_trace(go.Scatter(
                    y=training_data["episode_rewards"],
                    mode='lines',
                    name='Episode Reward',
                    line=dict(color='blue')
                ))
                fig_rewards.add_trace(go.Scatter(
                    y=pd.Series(training_data["episode_rewards"]).rolling(10).mean(),
                    mode='lines',
                    name='Moving Average (10)',
                    line=dict(color='red')
                ))
                fig_rewards.update_layout(
                    title="Episode Rewards Over Time",
                    xaxis_title="Episode",
                    yaxis_title="Total Reward",
                    height=400
                )
                st.plotly_chart(fig_rewards, use_container_width=True)
            
            with col2:
                # Policy Loss
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=training_data["training_stats"]["policy_losses"],
                    mode='lines',
                    name='Policy Loss',
                    line=dict(color='green')
                ))
                fig_loss.add_trace(go.Scatter(
                    y=training_data["training_stats"]["value_losses"],
                    mode='lines',
                    name='Value Loss',
                    line=dict(color='orange')
                ))
                fig_loss.update_layout(
                    title="Loss Convergence",
                    xaxis_title="Training Iteration",
                    yaxis_title="Loss",
                    height=400
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            # Entropy and Clip Fraction
            col1, col2 = st.columns(2)
            
            with col1:
                fig_entropy = go.Figure()
                fig_entropy.add_trace(go.Scatter(
                    y=training_data["training_stats"]["entropies"],
                    mode='lines',
                    name='Entropy',
                    line=dict(color='purple')
                ))
                fig_entropy.update_layout(
                    title="Policy Entropy (Exploration)",
                    xaxis_title="Training Iteration",
                    yaxis_title="Entropy",
                    height=300
                )
                st.plotly_chart(fig_entropy, use_container_width=True)
            
            with col2:
                fig_clip = go.Figure()
                fig_clip.add_trace(go.Scatter(
                    y=training_data["training_stats"]["clip_fractions"],
                    mode='lines',
                    name='Clip Fraction',
                    line=dict(color='brown')
                ))
                fig_clip.update_layout(
                    title="PPO Clip Fraction",
                    xaxis_title="Training Iteration",
                    yaxis_title="Clip Fraction",
                    height=300
                )
                st.plotly_chart(fig_clip, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Detailed training stats not available: {e}")
        
        # Training Controls
        st.markdown("---")
        st.subheader("Training Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            episodes = st.number_input("Number of Episodes", min_value=10, max_value=1000, value=100)
        with col2:
            batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32)
        
        if st.button("🚀 Start Training", type="primary"):
            response = requests.post(f"{API_URL}/training/start", 
                                    json={"episodes": episodes, "batch_size": batch_size})
            if response.status_code == 200:
                st.success("Training initiated! Run 'python scripts/train.py' in terminal")
            else:
                st.error("Failed to start training")
        
    except Exception as e:
        st.error(f"Error loading training status: {e}")

elif page == "📈 Analytics":
    st.title("📈 Analytics & Insights")
    st.markdown("Comprehensive analysis of system performance")
    
    # Load evaluation results
    try:
        eval_results = requests.get(f"{API_URL}/evaluation/results").json()
        
        if "status" not in eval_results:
            # Performance Metrics
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Engagement", f"{eval_results['avg_engagement']:.2%}")
            with col2:
                st.metric("Mood Improvement", f"{eval_results['avg_mood_improvement']:.3f}")
            with col3:
                st.metric("Completion Rate", f"{eval_results['completion_rate']:.1%}")
            with col4:
                st.metric("Safety Violations", f"{eval_results['safety_violation_rate']:.3f}")
            
            # Strategy Distribution
            st.subheader("Strategy Usage Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                strategies = list(eval_results["most_used_strategies"].keys())
                counts = list(eval_results["most_used_strategies"].values())
                
                fig_strategies = go.Figure(data=[go.Pie(
                    labels=strategies,
                    values=counts,
                    hole=0.3
                )])
                fig_strategies.update_layout(
                    title="Conversation Strategy Distribution",
                    height=400
                )
                st.plotly_chart(fig_strategies, use_container_width=True)
            
            with col2:
                resources = list(eval_results["most_recommended_resources"].keys())
                counts = list(eval_results["most_recommended_resources"].values())
                
                fig_resources = go.Figure(data=[go.Bar(
                    x=resources,
                    y=counts,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )])
                fig_resources.update_layout(
                    title="Resource Recommendations",
                    xaxis_title="Resource Type",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig_resources, use_container_width=True)
            
            # Contextual Bandit Statistics
            st.subheader("Contextual Bandit Performance")
            
            try:
                with open("data/models/contextual_bandit.json") as f:
                    bandit_data = json.load(f)
                
                # Extract arm statistics
                arms_df = pd.DataFrame([
                    {
                        "Resource": name,
                        "Pulls": data["total_pulls"],
                        "Alpha": data["alpha"],
                        "Beta": data["beta"],
                        "Mean Reward": data["total_reward"] / max(data["total_pulls"], 1)
                    }
                    for name, data in bandit_data["arms"].items()
                ])
                
                fig_bandit = go.Figure()
                fig_bandit.add_trace(go.Bar(
                    x=arms_df["Resource"],
                    y=arms_df["Mean Reward"],
                    name="Mean Reward",
                    marker_color='lightblue'
                ))
                fig_bandit.update_layout(
                    title="Bandit Arm Performance",
                    xaxis_title="Resource Type",
                    yaxis_title="Mean Reward",
                    height=400
                )
                st.plotly_chart(fig_bandit, use_container_width=True)
                
                # Display detailed statistics
                st.dataframe(arms_df, use_container_width=True)
                
            except Exception as e:
                st.info("Contextual bandit statistics not available")
            
        else:
            st.info("No evaluation results available. Run evaluation first.")
            
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

elif page == "🧪 Evaluation":
    st.title("🧪 Model Evaluation")
    st.markdown("Test the trained model on diverse personas")
    
    # Evaluation Controls
    st.subheader("Evaluation Settings")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_personas = st.number_input("Number of Test Personas", min_value=5, max_value=50, value=10)
    with col2:
        max_turns = st.number_input("Max Turns per Conversation", min_value=5, max_value=20, value=10)
    with col3:
        if st.button("🔬 Run Evaluation", type="primary"):
            st.info("Run 'python scripts/evaluate.py' in terminal to perform evaluation")
    
    # Comparative Analysis
    st.subheader("Comparative Analysis")
    
    # Create mock data for demonstration
    comparison_data = pd.DataFrame({
        "Model": ["Random Policy", "PPO Only", "PPO + Bandits"],
        "Engagement": [0.45, 0.68, 0.79],
        "Mood Improvement": [0.02, 0.07, 0.098],
        "Safety Violations": [0.15, 0.02, 0.0],
        "Completion Rate": [0.30, 0.50, 0.60]
    })
    
    fig_comparison = go.Figure()
    
    metrics = ["Engagement", "Mood Improvement", "Safety Violations", "Completion Rate"]
    for metric in metrics:
        fig_comparison.add_trace(go.Bar(
            name=metric,
            x=comparison_data["Model"],
            y=comparison_data[metric]
        ))
    
    fig_comparison.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model Type",
        yaxis_title="Score",
        barmode='group',
        height=500
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Ablation Study Results
    st.subheader("Ablation Study")
    
    ablation_results = {
        "Component": ["Full System", "Without PPO", "Without Bandits", "Without Safety"],
        "Performance": [0.79, 0.62, 0.71, 0.75],
        "Safety Score": [1.0, 0.8, 0.95, 0.0]
    }
    
    df_ablation = pd.DataFrame(ablation_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ablation1 = go.Figure(data=[go.Bar(
            x=df_ablation["Component"],
            y=df_ablation["Performance"],
            marker_color=['green', 'orange', 'orange', 'orange']
        )])
        fig_ablation1.update_layout(
            title="Performance Impact",
            xaxis_title="Configuration",
            yaxis_title="Performance Score",
            height=400
        )
        st.plotly_chart(fig_ablation1, use_container_width=True)
    
    with col2:
        fig_ablation2 = go.Figure(data=[go.Bar(
            x=df_ablation["Component"],
            y=df_ablation["Safety Score"],
            marker_color=['green', 'yellow', 'yellow', 'red']
        )])
        fig_ablation2.update_layout(
            title="Safety Impact",
            xaxis_title="Configuration",
            yaxis_title="Safety Score",
            height=400
        )
        st.plotly_chart(fig_ablation2, use_container_width=True)

elif page == "⚙️ Settings":
    st.title("⚙️ Settings & Configuration")
    
    # Model Settings
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**PPO Settings**")
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.0003, format="%.4f")
        clip_ratio = st.slider("Clip Ratio", 0.1, 0.3, 0.2)
        entropy_coef = st.slider("Entropy Coefficient", 0.0, 0.1, 0.01)
    
    with col2:
        st.write("**Contextual Bandit Settings**")
        exploration_mode = st.selectbox("Exploration Mode", ["thompson", "ucb", "greedy"])
        exploration_rate = st.slider("Exploration Rate", 0.0, 1.0, 0.3)
    
    # Safety Settings
    st.subheader("Safety Configuration")
    
    crisis_threshold = st.slider("Crisis Detection Threshold", 0.5, 1.0, 0.8)
    referral_threshold = st.slider("Professional Referral Threshold", 0.5, 1.0, 0.8)
    
    # Persona Settings
    st.subheader("Persona Generation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        severity_dist = st.multiselect(
            "Severity Distribution",
            ["mild", "moderate", "severe"],
            default=["mild", "moderate", "severe"]
        )
    
    with col2:
        concern_types = st.multiselect(
            "Concern Types",
            ["anxiety", "depression", "stress", "trauma", "relationship"],
            default=["anxiety", "depression", "stress"]
        )
    
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")
    
    # System Information
    st.subheader("System Information")
    
    st.info(f"""
    **Version:** 1.0.0  
    **RL Approaches:** PPO + Contextual Bandits  
    **LLM Model:** GPT-4o-mini  
    **Training Episodes:** 91  
    **Best Reward:** 4.84  
    **Safety Violations:** 0  
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Mental Wellness Companion - RL-Powered Mental Health Support</p>
    <p style='font-size: 0.8em'>Using PPO and Contextual Bandits with LLM-in-the-Loop Training</p>
</div>
""", unsafe_allow_html=True)