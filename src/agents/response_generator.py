"""
Response generator that uses RL-selected strategies with LLM enhancement
Maintains RL control while improving response quality
"""
from openai import OpenAI
from config.settings import OPENAI_API_KEY, MODEL_NAME

class ResponseGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def generate_response(
        self,
        strategy: str,  # From RL agent
        tone: str,      # From RL agent
        context: dict,  # Conversation context
        use_llm: bool = False  # Toggle for comparison
    ) -> str:
        """
        Generate response based on RL-selected strategy
        
        This maintains RL control - the RL agent still decides WHAT to do,
        we just use LLM to express it better
        """
        
        if not use_llm:
            # Use template-based approach (current method)
            return self._template_response(strategy, tone, context)
        
        # Use LLM to generate response following RL's decision
        conversation_history = context.get("conversation_history", [])
        last_user_message = conversation_history[-1]["content"] if conversation_history else ""
        
        prompt = f"""
        You are a mental wellness support agent. 
        You must respond using the following strategy and tone:
        
        Strategy: {strategy}
        Tone: {tone}
        
        Strategy definitions:
        - empathetic_listening: Focus on understanding and reflecting feelings
        - cognitive_behavioral: Help identify and challenge thought patterns
        - validation: Acknowledge and validate their experiences
        - problem_solving: Work together to find practical solutions
        - mindfulness: Guide toward present-moment awareness
        - motivational: Encourage and inspire positive action
        - psychoeducation: Provide educational information
        - supportive: Offer general support and encouragement
        
        User's last message: {last_user_message}
        
        Generate a brief, appropriate response (2-3 sentences) that:
        1. Strictly follows the {strategy} strategy
        2. Maintains a {tone} tone
        3. Is therapeutic and helpful
        4. Avoids giving medical advice
        
        Response:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a supportive mental wellness companion."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback to template
            return self._template_response(strategy, tone, context)
    
    def _template_response(self, strategy, tone, context):
        # Your existing template logic
        templates = {
            "empathetic_listening": "I hear how difficult this is for you.",
            # ... etc
        }
        return templates.get(strategy, "I understand.")