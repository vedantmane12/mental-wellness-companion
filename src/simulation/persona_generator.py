"""
Persona Generator for creating diverse user profiles
Generates psychologically consistent personas for training
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

from src.simulation.openai_client import OpenAIClient
from src.utils.logger import logger
from src.utils.helpers import save_json, load_json, generate_id
from config.settings import DATA_DIR


@dataclass
class PersonaDistribution:
    """Distribution parameters for persona generation"""
    age_range: tuple = (18, 65)
    gender_distribution: dict = None
    severity_distribution: dict = None
    concern_distribution: dict = None
    
    def __post_init__(self):
        if self.gender_distribution is None:
            self.gender_distribution = {
                "male": 0.45,
                "female": 0.45,
                "non-binary": 0.10
            }
        
        if self.severity_distribution is None:
            self.severity_distribution = {
                "mild": 0.40,
                "moderate": 0.45,
                "severe": 0.15
            }
        
        if self.concern_distribution is None:
            self.concern_distribution = {
                "anxiety": 0.35,
                "depression": 0.25,
                "stress": 0.20,
                "trauma": 0.10,
                "relationship": 0.10
            }


class PersonaGenerator:
    """Generate diverse personas for mental wellness training"""
    
    def __init__(self, openai_client: Optional[OpenAIClient] = None):
        """
        Initialize persona generator
        
        Args:
            openai_client: Optional OpenAI client instance
        """
        self.client = openai_client or OpenAIClient()
        self.personas_dir = DATA_DIR / "personas"
        self.personas_dir.mkdir(exist_ok=True)
        self.distribution = PersonaDistribution()
        
        # Define persona archetypes for diversity
        self.archetypes = [
            "anxious_professional",
            "stressed_student", 
            "depressed_adult",
            "overwhelmed_parent",
            "isolated_elder",
            "grieving_individual",
            "burnout_healthcare",
            "relationship_struggles",
            "career_transition",
            "chronic_illness"
        ]
        
        logger.info("Initialized PersonaGenerator")
    
    def generate_batch(
        self,
        batch_size: int = 100,
        save: bool = True,
        prefix: str = "training"
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of diverse personas
        
        Args:
            batch_size: Number of personas to generate
            save: Whether to save personas to disk
            prefix: Prefix for saved files (training/validation/testing)
        
        Returns:
            List of generated personas
        """
        logger.info(f"Generating {batch_size} personas with prefix '{prefix}'")
        personas = []
        
        # Ensure diversity by cycling through archetypes
        for i in range(batch_size):
            archetype = self.archetypes[i % len(self.archetypes)]
            
            # Add variation to archetype
            variation = self._get_archetype_variation(archetype)
            
            try:
                # Generate persona using OpenAI
                persona = self.client.generate_persona(
                    persona_id=i,
                    persona_type=variation
                )
                
                # Add metadata
                persona["archetype"] = archetype
                persona["batch_prefix"] = prefix
                persona["generated_at"] = generate_id("timestamp")
                
                # Calculate initial emotional state
                persona["initial_emotional_state"] = self._calculate_emotional_state(persona)
                
                # Validate persona
                if self._validate_persona(persona):
                    personas.append(persona)
                    
                    if save:
                        self._save_persona(persona, prefix)
                else:
                    logger.warning(f"Invalid persona generated for ID {i}, retrying...")
                    # Retry with fallback
                    persona = self._generate_fallback_persona(i, archetype)
                    personas.append(persona)
                    
            except Exception as e:
                logger.error(f"Error generating persona {i}: {e}")
                # Use fallback
                persona = self._generate_fallback_persona(i, archetype)
                personas.append(persona)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{batch_size} personas")
        
        # Save batch metadata
        if save:
            self._save_batch_metadata(personas, prefix)
        
        logger.info(f"Successfully generated {len(personas)} personas")
        return personas
    
    def _get_archetype_variation(self, archetype: str) -> str:
        """Get a variation of the archetype for diversity"""
        variations = {
            "anxious_professional": [
                "anxious young professional with imposter syndrome",
                "anxious mid-career professional with work-life balance issues",
                "anxious senior professional facing retirement anxiety"
            ],
            "stressed_student": [
                "stressed undergraduate with academic pressure",
                "stressed graduate student with research anxiety",
                "stressed medical student with burnout"
            ],
            "depressed_adult": [
                "adult with seasonal depression",
                "adult with chronic depression",
                "adult with postpartum depression"
            ],
            "overwhelmed_parent": [
                "single parent juggling work and childcare",
                "new parent with adjustment difficulties",
                "parent of special needs child"
            ],
            "isolated_elder": [
                "recently retired individual feeling purposeless",
                "elderly person with mobility limitations",
                "widowed elder experiencing loneliness"
            ]
        }
        
        if archetype in variations:
            return random.choice(variations[archetype])
        return archetype
    
    def _calculate_emotional_state(self, persona: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate initial emotional state based on persona profile
        
        Args:
            persona: Persona profile
        
        Returns:
            Emotional state vector
        """
        emotional_state = {
            "anxiety": 0.3,
            "depression": 0.3,
            "stress": 0.3,
            "anger": 0.2,
            "happiness": 0.4
        }
        
        # Adjust based on mental health profile
        mental_health = persona.get("mental_health", {})
        concerns = mental_health.get("primary_concerns", [])
        severity = mental_health.get("severity", "mild")
        
        # Severity multipliers
        severity_multipliers = {
            "mild": 1.2,
            "moderate": 1.5,
            "severe": 2.0
        }
        multiplier = severity_multipliers.get(severity, 1.2)
        
        # Adjust emotional state based on concerns
        for concern in concerns:
            concern_lower = concern.lower()
            if "anxiety" in concern_lower:
                emotional_state["anxiety"] = min(0.9, emotional_state["anxiety"] * multiplier)
            if "depress" in concern_lower:
                emotional_state["depression"] = min(0.9, emotional_state["depression"] * multiplier)
                emotional_state["happiness"] = max(0.1, emotional_state["happiness"] / multiplier)
            if "stress" in concern_lower:
                emotional_state["stress"] = min(0.9, emotional_state["stress"] * multiplier)
            if "anger" in concern_lower or "irritab" in concern_lower:
                emotional_state["anger"] = min(0.9, emotional_state["anger"] * multiplier)
            if "trauma" in concern_lower:
                emotional_state["anxiety"] = min(0.9, emotional_state["anxiety"] * 1.3)
                emotional_state["depression"] = min(0.9, emotional_state["depression"] * 1.2)
        
        # Normalize to ensure values are between 0 and 1
        for key in emotional_state:
            emotional_state[key] = round(float(np.clip(emotional_state[key], 0.0, 1.0)), 3)
        
        return emotional_state
    
    def _validate_persona(self, persona: Dict[str, Any]) -> bool:
        """
        Validate that persona has all required fields
        
        Args:
            persona: Persona to validate
        
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            "id", "demographics", "mental_health", 
            "personality", "goals", "preferences"
        ]
        
        for field in required_fields:
            if field not in persona:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate demographics
        demographics = persona.get("demographics", {})
        if not demographics.get("age") or not demographics.get("gender"):
            logger.warning("Invalid demographics")
            return False
        
        # Validate mental health
        mental_health = persona.get("mental_health", {})
        if not mental_health.get("primary_concerns"):
            logger.warning("No primary concerns specified")
            return False
        
        return True
    
    def _generate_fallback_persona(self, persona_id: int, archetype: str) -> Dict[str, Any]:
        """Generate a fallback persona when API fails"""
        age = random.randint(18, 65)
        gender = random.choice(["male", "female", "non-binary"])
        
        persona = {
            "id": f"persona_{persona_id}",
            "archetype": archetype,
            "demographics": {
                "age": age,
                "gender": gender,
                "occupation": random.choice([
                    "teacher", "engineer", "nurse", "student", 
                    "retail worker", "freelancer", "manager"
                ]),
                "education": random.choice([
                    "high school", "bachelor's degree", 
                    "master's degree", "trade school"
                ]),
                "location": random.choice(["urban", "suburban", "rural"])
            },
            "mental_health": {
                "primary_concerns": random.sample(
                    ["anxiety", "depression", "stress", "insomnia", "burnout"], 
                    k=random.randint(1, 3)
                ),
                "severity": random.choice(["mild", "moderate", "severe"]),
                "duration": random.choice(["3 months", "6 months", "1 year", "2+ years"]),
                "triggers": random.sample(
                    ["work", "relationships", "health", "finances", "family"],
                    k=random.randint(2, 3)
                ),
                "coping_mechanisms": random.sample(
                    ["exercise", "meditation", "journaling", "music", "talking to friends"],
                    k=random.randint(1, 3)
                ),
                "therapy_history": random.choice(["none", "some", "ongoing"]),
                "medication": random.choice(["no", "yes", "considering"])
            },
            "personality": {
                "traits": random.sample(
                    ["introverted", "anxious", "perfectionist", "empathetic", 
                     "analytical", "creative", "organized", "spontaneous"],
                    k=random.randint(3, 5)
                ),
                "communication_style": random.choice(
                    ["direct", "indirect", "emotional", "analytical"]
                ),
                "openness": random.choice(["low", "medium", "high"]),
                "motivation_level": random.choice(["low", "medium", "high"]),
                "tech_comfort": random.choice(["low", "medium", "high"])
            },
            "goals": random.sample(
                ["reduce anxiety", "improve mood", "better sleep", 
                 "stress management", "build confidence", "improve relationships"],
                k=random.randint(2, 3)
            ),
            "barriers": random.sample(
                ["time", "stigma", "cost", "motivation", "skepticism"],
                k=random.randint(1, 3)
            ),
            "preferences": {
                "interaction_style": random.choice(["supportive", "challenging", "balanced"]),
                "resource_types": random.sample(
                    ["articles", "videos", "exercises", "worksheets", "meditation"],
                    k=random.randint(2, 3)
                ),
                "session_frequency": random.choice(["daily", "weekly", "as-needed"])
            },
            "backstory": f"A {age}-year-old {gender} {archetype.replace('_', ' ')} seeking support.",
            "initial_emotional_state": {
                "anxiety": round(random.uniform(0.3, 0.8), 3),
                "depression": round(random.uniform(0.2, 0.7), 3),
                "stress": round(random.uniform(0.3, 0.8), 3),
                "anger": round(random.uniform(0.1, 0.5), 3),
                "happiness": round(random.uniform(0.2, 0.6), 3)
            }
        }
        
        return persona
    
    def _save_persona(self, persona: Dict[str, Any], prefix: str):
        """Save persona to disk"""
        filename = f"{prefix}_persona_{persona['id']}.json"
        filepath = self.personas_dir / prefix / filename
        filepath.parent.mkdir(exist_ok=True)
        save_json(persona, filepath)
    
    def _save_batch_metadata(self, personas: List[Dict[str, Any]], prefix: str):
        """Save metadata about the batch"""
        metadata = {
            "batch_size": len(personas),
            "prefix": prefix,
            "generated_at": generate_id("timestamp"),
            "archetypes": list(set(p.get("archetype", "unknown") for p in personas)),
            "age_range": [
                min(p.get("demographics", {}).get("age", 30) for p in personas),
                max(p.get("demographics", {}).get("age", 30) for p in personas)
            ],
            "severity_distribution": {}
        }
        
        # Calculate severity distribution
        for persona in personas:
            severity = persona.get("mental_health", {}).get("severity", "unknown")
            metadata["severity_distribution"][severity] = metadata["severity_distribution"].get(severity, 0) + 1
        
        filepath = self.personas_dir / prefix / "batch_metadata.json"
        save_json(metadata, filepath)
        logger.info(f"Saved batch metadata to {filepath}")
    
    def load_personas(self, prefix: str = "training") -> List[Dict[str, Any]]:
        """Load personas from disk"""
        personas_path = self.personas_dir / prefix
        if not personas_path.exists():
            logger.warning(f"No personas found at {personas_path}")
            return []
        
        personas = []
        for filepath in personas_path.glob("*.json"):
            if "metadata" not in filepath.name:
                persona = load_json(filepath)
                if persona:
                    personas.append(persona)
        
        logger.info(f"Loaded {len(personas)} personas from {prefix}")
        return personas


# Test function
def test_persona_generator():
    """Test persona generation"""
    generator = PersonaGenerator()
    
    # Generate a small batch for testing
    personas = generator.generate_batch(batch_size=5, prefix="test")
    
    for i, persona in enumerate(personas):
        logger.info(f"Persona {i}: {persona.get('demographics', {}).get('age')}yo "
                   f"{persona.get('demographics', {}).get('occupation')} - "
                   f"{persona.get('mental_health', {}).get('primary_concerns')}")
    
    return personas


if __name__ == "__main__":
    test_persona_generator()