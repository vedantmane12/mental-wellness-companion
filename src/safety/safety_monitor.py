"""
Safety Monitor for ensuring safe and ethical interactions
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.utils.logger import logger
from config.settings import SAFETY_CONFIG


@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    severity: str  # "low", "medium", "high", "critical"
    type: str  # "crisis", "harmful_content", "boundary", "ethical"
    description: str
    timestamp: datetime
    action_taken: str


class SafetyMonitor:
    """
    Monitor and enforce safety constraints in conversations
    """
    
    def __init__(self):
        """Initialize safety monitor"""
        self.violations = []
        self.crisis_keywords = SAFETY_CONFIG["crisis_keywords"]
        self.professional_referral_threshold = SAFETY_CONFIG["professional_referral_threshold"]
        
        # Additional safety patterns
        self.harmful_patterns = [
            r"how to (hurt|harm|kill) (myself|yourself)",
            r"suicide (method|plan|note)",
            r"want to (die|end it|disappear)",
            r"better off (dead|gone|without me)",
            r"no reason to (live|go on|continue)",
        ]
        
        # Boundary violations
        self.boundary_patterns = [
            r"(give|tell) me (drugs|medication)",
            r"diagnose (me|my)",
            r"am I (crazy|insane|psychotic)",
            r"(cure|fix) my",
        ]
        
        # Statistics
        self.stats = {
            "total_checks": 0,
            "violations_detected": 0,
            "crisis_interventions": 0,
            "referrals_made": 0
        }
        
        logger.info("Initialized SafetyMonitor")
    
    def check_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[SafetyViolation]]:
        """
        Check a message for safety violations
        
        Args:
            message: Message to check
            context: Additional context
        
        Returns:
            Tuple of (is_safe, violation_if_any)
        """
        self.stats["total_checks"] += 1
        
        # Check for crisis keywords
        crisis_check = self._check_crisis_keywords(message)
        if crisis_check:
            violation = SafetyViolation(
                severity="critical",
                type="crisis",
                description=f"Crisis keyword detected: {crisis_check}",
                timestamp=datetime.now(),
                action_taken="Triggered crisis intervention"
            )
            self._handle_violation(violation)
            return False, violation
        
        # Check harmful patterns
        harmful_check = self._check_harmful_patterns(message)
        if harmful_check:
            violation = SafetyViolation(
                severity="high",
                type="harmful_content",
                description=f"Harmful pattern detected: {harmful_check}",
                timestamp=datetime.now(),
                action_taken="Redirected to safe response"
            )
            self._handle_violation(violation)
            return False, violation
        
        # Check boundary violations
        boundary_check = self._check_boundary_violations(message)
        if boundary_check:
            violation = SafetyViolation(
                severity="medium",
                type="boundary",
                description=f"Boundary violation: {boundary_check}",
                timestamp=datetime.now(),
                action_taken="Clarified agent limitations"
            )
            self._handle_violation(violation)
            return False, violation
        
        return True, None
    
    def check_response(
        self,
        response: str,
        strategy: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Check agent response for appropriateness
        
        Args:
            response: Agent's response
            strategy: Strategy used
            context: Additional context
        
        Returns:
            Tuple of (is_appropriate, modified_response_if_needed)
        """
        # Check for inappropriate content
        if self._contains_medical_advice(response):
            logger.warning("Response contains medical advice, modifying...")
            response = self._remove_medical_advice(response)
            response += "\n\nPlease consult with a healthcare professional for medical advice."
        
        # Check for overpromising
        if self._contains_overpromises(response):
            logger.warning("Response contains overpromises, modifying...")
            response = self._temper_promises(response)
        
        # Ensure crisis resources if needed
        if context and context.get("risk_level", 0) > self.professional_referral_threshold:
            if "crisis" not in response.lower() and "professional" not in response.lower():
                response += "\n\n" + self._get_crisis_resources()
                self.stats["referrals_made"] += 1
        
        return True, response
    
    def _check_crisis_keywords(self, message: str) -> Optional[str]:
        """Check for crisis keywords"""
        message_lower = message.lower()
        for keyword in self.crisis_keywords:
            if keyword in message_lower:
                return keyword
        return None
    
    def _check_harmful_patterns(self, message: str) -> Optional[str]:
        """Check for harmful patterns"""
        message_lower = message.lower()
        for pattern in self.harmful_patterns:
            if re.search(pattern, message_lower):
                return pattern
        return None
    
    def _check_boundary_violations(self, message: str) -> Optional[str]:
        """Check for boundary violations"""
        message_lower = message.lower()
        for pattern in self.boundary_patterns:
            if re.search(pattern, message_lower):
                return pattern
        return None
    
    def _contains_medical_advice(self, response: str) -> bool:
        """Check if response contains medical advice"""
        medical_terms = [
            "medication", "prescribe", "dosage", "diagnosis",
            "medical condition", "treatment plan"
        ]
        response_lower = response.lower()
        return any(term in response_lower for term in medical_terms)
    
    def _remove_medical_advice(self, response: str) -> str:
        """Remove medical advice from response"""
        # Simple approach - in production would be more sophisticated
        lines = response.split('\n')
        filtered_lines = []
        for line in lines:
            if not any(term in line.lower() for term in ["medication", "prescribe", "dosage"]):
                filtered_lines.append(line)
        return '\n'.join(filtered_lines)
    
    def _contains_overpromises(self, response: str) -> bool:
        """Check if response contains overpromises"""
        overpromise_patterns = [
            "will definitely", "guaranteed to", "will cure",
            "will fix", "promise you", "100%"
        ]
        response_lower = response.lower()
        return any(pattern in response_lower for pattern in overpromise_patterns)
    
    def _temper_promises(self, response: str) -> str:
        """Temper overpromises in response"""
        replacements = {
            "will definitely": "may",
            "guaranteed to": "could potentially",
            "will cure": "might help with",
            "will fix": "could improve",
            "promise you": "hope that",
            "100%": "very likely"
        }
        
        result = response
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result
    
    def _get_crisis_resources(self) -> str:
        """Get crisis resources text"""
        return SAFETY_CONFIG["crisis_response"]
    
    def _handle_violation(self, violation: SafetyViolation):
        """Handle a safety violation"""
        self.violations.append(violation)
        self.stats["violations_detected"] += 1
        
        if violation.severity == "critical":
            self.stats["crisis_interventions"] += 1
            logger.critical(f"SAFETY VIOLATION: {violation.description}")
        else:
            logger.warning(f"Safety violation ({violation.severity}): {violation.description}")
    
    def get_safe_response(self, violation_type: str) -> str:
        """Get a safe response for different violation types"""
        safe_responses = {
            "crisis": SAFETY_CONFIG["crisis_response"],
            "harmful_content": "I'm concerned about what you're sharing. Your safety is important. Let's focus on getting you support. Would you like to talk about what's troubling you?",
            "boundary": "I understand you're looking for help. While I can provide support and information, I'm not able to provide medical diagnoses or prescribe treatments. For medical concerns, please consult with a healthcare professional.",
            "ethical": "I want to be helpful, but I need to stay within appropriate boundaries. Let's focus on how I can best support you within my capabilities.",
            "default": "I'm here to listen and support you. How can I help in a way that's safe and appropriate?"
        }
        
        return safe_responses.get(violation_type, safe_responses["default"])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety monitoring statistics"""
        return {
            "stats": self.stats,
            "recent_violations": [
                {
                    "severity": v.severity,
                    "type": v.type,
                    "timestamp": v.timestamp.isoformat()
                }
                for v in self.violations[-10:]  # Last 10 violations
            ],
            "violation_breakdown": self._get_violation_breakdown()
        }
    
    def _get_violation_breakdown(self) -> Dict[str, int]:
        """Get breakdown of violations by type"""
        breakdown = {}
        for violation in self.violations:
            violation_type = violation.type
            breakdown[violation_type] = breakdown.get(violation_type, 0) + 1
        return breakdown
    
    def export_safety_report(self, filepath: Optional[str] = None) -> str:
        """Export safety report"""
        report = []
        report.append("=== SAFETY MONITORING REPORT ===\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n\n")
        
        report.append("STATISTICS:\n")
        for key, value in self.stats.items():
            report.append(f"  {key}: {value}\n")
        
        report.append("\nVIOLATION BREAKDOWN:\n")
        for vtype, count in self._get_violation_breakdown().items():
            report.append(f"  {vtype}: {count}\n")
        
        if self.violations:
            report.append("\nRECENT VIOLATIONS:\n")
            for v in self.violations[-5:]:
                report.append(f"  [{v.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                            f"{v.severity.upper()} - {v.type}: {v.description}\n")
        
        report_text = ''.join(report)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(report_text)
            logger.info(f"Safety report exported to {filepath}")
        
        return report_text


# Test function
def test_safety_monitor():
    """Test safety monitor"""
    monitor = SafetyMonitor()
    
    # Test various messages
    test_messages = [
        "I'm feeling a bit down today",  # Safe
        "I want to hurt myself",  # Crisis
        "Can you diagnose my depression?",  # Boundary
        "Things are getting better",  # Safe
    ]
    
    for msg in test_messages:
        is_safe, violation = monitor.check_message(msg)
        print(f"Message: '{msg[:30]}...' - Safe: {is_safe}")
        if violation:
            print(f"  Violation: {violation.type} ({violation.severity})")
    
    # Test response checking
    response = "This medication will definitely cure your anxiety."
    is_appropriate, modified = monitor.check_response(response, "supportive")
    print(f"\nOriginal response: {response}")
    print(f"Modified response: {modified}")
    
    # Get statistics
    stats = monitor.get_statistics()
    print(f"\nSafety Statistics: {stats['stats']}")
    
    return monitor


if __name__ == "__main__":
    test_safety_monitor()