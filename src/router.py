#!/usr/bin/env python3
"""
MCP Router - Intelligent Model Routing System
Routes queries to the best model based on query characteristics and context.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that models can handle."""
    REASONING = "reasoning"  # Complex reasoning, analysis, planning
    CODE_GENERATION = "code_generation"  # Writing code
    CODE_EDIT = "code_edit"  # Editing/refactoring code
    SUMMARIZATION = "summarization"  # Summarizing content
    QUESTION_ANSWERING = "qa"  # Answering questions
    TRANSLATION = "translation"  # Language translation
    CREATIVE = "creative"  # Creative writing, brainstorming
    STREAMING = "streaming"  # Real-time streaming responses
    MULTIMODAL = "multimodal"  # Image, audio, video processing
    EMBEDDING = "embedding"  # Generating embeddings


class Complexity(Enum):
    """Query complexity levels."""
    LOW = "low"  # Simple, straightforward tasks
    MEDIUM = "medium"  # Moderate complexity
    HIGH = "high"  # Complex, requires deep reasoning
    VERY_HIGH = "very_high"  # Extremely complex


@dataclass
class ModelCapabilities:
    """Capabilities and characteristics of a model."""
    name: str
    provider: str  # e.g., "openai", "anthropic", "google"
    model_id: str  # e.g., "gpt-4o", "claude-3-5-sonnet"
    
    # Capabilities
    supports_reasoning: bool = True
    supports_code: bool = True
    supports_streaming: bool = False
    supports_multimodal: bool = False
    supports_embeddings: bool = False
    
    # Performance characteristics
    max_tokens: int = 4096
    context_window: int = 128000
    cost_per_1k_tokens_input: float = 0.0
    cost_per_1k_tokens_output: float = 0.0
    avg_latency_ms: int = 500  # Average latency in milliseconds
    
    # Quality scores (0-1)
    reasoning_quality: float = 0.8
    code_quality: float = 0.8
    speed_score: float = 0.8
    
    # Task preferences
    preferred_tasks: List[TaskType] = field(default_factory=list)
    unsuitable_tasks: List[TaskType] = field(default_factory=list)
    
    # API configuration
    api_key_env_var: str = ""
    base_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "supports_reasoning": self.supports_reasoning,
            "supports_code": self.supports_code,
            "supports_streaming": self.supports_streaming,
            "supports_multimodal": self.supports_multimodal,
            "supports_embeddings": self.supports_embeddings,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "cost_per_1k_tokens_input": self.cost_per_1k_tokens_input,
            "cost_per_1k_tokens_output": self.cost_per_1k_tokens_output,
            "avg_latency_ms": self.avg_latency_ms,
            "reasoning_quality": self.reasoning_quality,
            "code_quality": self.code_quality,
            "speed_score": self.speed_score,
            "preferred_tasks": [t.value for t in self.preferred_tasks],
            "unsuitable_tasks": [t.value for t in self.unsuitable_tasks],
        }


@dataclass
class ChatSummary:
    """Summary of chat history for context-aware routing."""
    total_messages: int = 0
    total_tokens_used: int = 0
    dominant_task_type: Optional[TaskType] = None
    avg_complexity: Optional[Complexity] = None
    topics: List[str] = field(default_factory=list)
    files_mentioned: List[str] = field(default_factory=list)
    languages_used: List[str] = field(default_factory=list)
    error_patterns: List[str] = field(default_factory=list)
    success_rate: float = 1.0  # Track if previous tasks succeeded
    context_depth: str = "shallow"  # "shallow", "medium", "deep"
    session_duration_mins: int = 0
    requires_continuity: bool = False  # Does task need context from chat?
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_messages": self.total_messages,
            "total_tokens_used": self.total_tokens_used,
            "dominant_task_type": self.dominant_task_type.value if self.dominant_task_type else None,
            "avg_complexity": self.avg_complexity.value if self.avg_complexity else None,
            "topics": self.topics,
            "files_mentioned": self.files_mentioned,
            "languages_used": self.languages_used,
            "error_patterns": self.error_patterns,
            "success_rate": self.success_rate,
            "context_depth": self.context_depth,
            "session_duration_mins": self.session_duration_mins,
            "requires_continuity": self.requires_continuity,
        }


@dataclass
class QueryContext:
    """Context information about a query."""
    query: str
    task_type: Optional[TaskType] = None
    complexity: Optional[Complexity] = None
    estimated_tokens: int = 0
    requires_streaming: bool = False
    requires_multimodal: bool = False
    requires_embeddings: bool = False
    priority: str = "normal"  # "low", "normal", "high", "urgent"
    cost_sensitivity: str = "normal"  # "low", "normal", "high"
    latency_sensitivity: str = "normal"  # "low", "normal", "high"
    metadata: Dict[str, Any] = field(default_factory=dict)
    chat_summary: Optional[ChatSummary] = None  # Chat history context


@dataclass
class RoutingDecision:
    """Result of routing a query."""
    selected_model: ModelCapabilities
    confidence: float  # 0-1 confidence score
    reasoning: str  # Explanation of why this model was chosen
    alternatives: List[ModelCapabilities] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_latency_ms: int = 0


class QueryAnalyzer:
    """Analyzes queries to determine characteristics."""
    
    def __init__(self):
        self.task_keywords = {
            TaskType.REASONING: [
                "analyze", "explain", "why", "how", "reason", "plan", "design",
                "architecture", "strategy", "think", "consider", "evaluate"
            ],
            TaskType.CODE_GENERATION: [
                "write", "create", "generate", "implement", "build", "code",
                "function", "class", "script", "program"
            ],
            TaskType.CODE_EDIT: [
                "refactor", "fix", "improve", "optimize", "update", "modify",
                "change", "edit", "rewrite"
            ],
            TaskType.SUMMARIZATION: [
                "summarize", "summary", "brief", "overview", "condense"
            ],
            TaskType.QUESTION_ANSWERING: [
                "what", "when", "where", "who", "which", "question", "answer"
            ],
            TaskType.CREATIVE: [
                "creative", "imagine", "brainstorm", "story", "poem", "write creatively"
            ],
            TaskType.STREAMING: [
                "stream", "real-time", "live", "continuous"
            ],
            TaskType.MULTIMODAL: [
                "image", "picture", "photo", "video", "audio", "visual"
            ],
        }
    
    def analyze(self, query: str, metadata: Optional[Dict] = None) -> QueryContext:
        """Analyze a query and return context."""
        query_lower = query.lower()
        
        # Determine task type
        task_type = self._detect_task_type(query_lower)
        
        # Determine complexity
        complexity = self._detect_complexity(query, task_type)
        
        # Estimate tokens (rough approximation: ~4 chars per token)
        estimated_tokens = len(query) // 4
        
        # Check for special requirements
        requires_streaming = any(kw in query_lower for kw in ["stream", "real-time", "live"])
        requires_multimodal = any(kw in query_lower for kw in ["image", "picture", "photo", "video", "audio"])
        requires_embeddings = "embedding" in query_lower or "embed" in query_lower
        
        return QueryContext(
            query=query,
            task_type=task_type,
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            requires_streaming=requires_streaming,
            requires_multimodal=requires_multimodal,
            requires_embeddings=requires_embeddings,
            metadata=metadata or {}
        )
    
    def _detect_task_type(self, query_lower: str) -> TaskType:
        """Detect the task type from query keywords."""
        scores = {task_type: 0 for task_type in TaskType}
        
        for task_type, keywords in self.task_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[task_type] += 1
        
        # Return task type with highest score, default to QUESTION_ANSWERING
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        return TaskType.QUESTION_ANSWERING
    
    def _detect_complexity(self, query: str, task_type: TaskType) -> Complexity:
        """Detect query complexity."""
        query_lower = query.lower()
        length = len(query)
        
        # Simple heuristics
        if length < 50:
            return Complexity.LOW
        elif length < 200:
            return Complexity.MEDIUM
        elif length < 500:
            return Complexity.HIGH
        else:
            return Complexity.VERY_HIGH


class ChatSummaryAnalyzer:
    """Analyzes chat history to extract routing signals."""
    
    def __init__(self):
        self.programming_languages = {
            "python": ["python", ".py", "pip", "pytest", "django", "flask"],
            "javascript": ["javascript", "js", "npm", "node", "react", "vue", "typescript", ".ts"],
            "rust": ["rust", "cargo", ".rs", "rustc"],
            "go": ["golang", "go ", ".go", "go mod"],
            "java": ["java", ".java", "maven", "gradle", "spring"],
            "c++": ["c++", "cpp", ".cpp", ".hpp", "cmake"],
            "ruby": ["ruby", ".rb", "rails", "gem"],
            "swift": ["swift", ".swift", "xcode"],
        }
        
        self.error_keywords = [
            "error", "failed", "exception", "bug", "crash", "not working",
            "doesn't work", "broken", "issue", "problem", "fix"
        ]
        
        self.continuity_keywords = [
            "continue", "previous", "earlier", "before", "last time",
            "we discussed", "as I mentioned", "building on", "following up",
            "same file", "that function", "the code above"
        ]
    
    def analyze(self, chat_history: List[Dict[str, Any]]) -> ChatSummary:
        """
        Analyze chat history and return a summary.
        
        Args:
            chat_history: List of message dicts with 'role', 'content', 'timestamp' keys
            
        Returns:
            ChatSummary with extracted signals
        """
        if not chat_history:
            return ChatSummary()
        
        summary = ChatSummary()
        summary.total_messages = len(chat_history)
        
        all_content = " ".join(msg.get("content", "") for msg in chat_history)
        all_content_lower = all_content.lower()
        
        # Estimate tokens used
        summary.total_tokens_used = len(all_content) // 4
        
        # Detect programming languages
        summary.languages_used = self._detect_languages(all_content_lower)
        
        # Extract files mentioned
        summary.files_mentioned = self._extract_files(all_content)
        
        # Detect error patterns
        summary.error_patterns = self._detect_errors(all_content_lower)
        
        # Determine context depth
        summary.context_depth = self._determine_context_depth(summary.total_tokens_used)
        
        # Check if continuity is required
        summary.requires_continuity = self._check_continuity(all_content_lower)
        
        # Calculate success rate from error mentions
        error_count = len(summary.error_patterns)
        if summary.total_messages > 0:
            summary.success_rate = max(0.5, 1.0 - (error_count * 0.1))
        
        # Detect dominant task type from history
        summary.dominant_task_type = self._detect_dominant_task(all_content_lower)
        
        # Calculate average complexity
        summary.avg_complexity = self._estimate_avg_complexity(summary)
        
        # Extract topics
        summary.topics = self._extract_topics(all_content_lower)
        
        # Calculate session duration if timestamps available
        if chat_history[0].get("timestamp") and chat_history[-1].get("timestamp"):
            try:
                start = chat_history[0]["timestamp"]
                end = chat_history[-1]["timestamp"]
                summary.session_duration_mins = int((end - start) / 60)
            except (TypeError, ValueError):
                pass
        
        return summary
    
    def analyze_from_text(self, summary_text: str) -> ChatSummary:
        """
        Analyze a text summary (like from ledger.md).
        
        Args:
            summary_text: Text content of chat summary/ledger
            
        Returns:
            ChatSummary with extracted signals
        """
        if not summary_text:
            return ChatSummary()
        
        summary = ChatSummary()
        text_lower = summary_text.lower()
        
        # Estimate based on text length
        summary.total_tokens_used = len(summary_text) // 4
        summary.total_messages = summary_text.count("\n") // 2 + 1
        
        # Extract signals
        summary.languages_used = self._detect_languages(text_lower)
        summary.files_mentioned = self._extract_files(summary_text)
        summary.error_patterns = self._detect_errors(text_lower)
        summary.context_depth = self._determine_context_depth(summary.total_tokens_used)
        summary.requires_continuity = self._check_continuity(text_lower)
        summary.dominant_task_type = self._detect_dominant_task(text_lower)
        summary.avg_complexity = self._estimate_avg_complexity(summary)
        summary.topics = self._extract_topics(text_lower)
        
        return summary
    
    def _detect_languages(self, content: str) -> List[str]:
        """Detect programming languages mentioned."""
        detected = []
        for lang, keywords in self.programming_languages.items():
            if any(kw in content for kw in keywords):
                detected.append(lang)
        return detected
    
    def _extract_files(self, content: str) -> List[str]:
        """Extract file paths mentioned."""
        import re
        # Match common file patterns
        patterns = [
            r'[\w\-/]+\.\w{1,10}',  # file.ext
            r'`[\w\-/\.]+`',  # `file.py`
        ]
        files = set()
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                clean = match.strip('`')
                if '.' in clean and len(clean) > 3:
                    files.add(clean)
        return list(files)[:20]  # Limit to 20 files
    
    def _detect_errors(self, content: str) -> List[str]:
        """Detect error patterns in content."""
        found = []
        for kw in self.error_keywords:
            if kw in content:
                found.append(kw)
        return found
    
    def _determine_context_depth(self, tokens: int) -> str:
        """Determine context depth based on token count."""
        if tokens < 2000:
            return "shallow"
        elif tokens < 10000:
            return "medium"
        else:
            return "deep"
    
    def _check_continuity(self, content: str) -> bool:
        """Check if conversation requires continuity."""
        return any(kw in content for kw in self.continuity_keywords)
    
    def _detect_dominant_task(self, content: str) -> Optional[TaskType]:
        """Detect the dominant task type from chat history."""
        scores = {
            TaskType.CODE_GENERATION: 0,
            TaskType.CODE_EDIT: 0,
            TaskType.REASONING: 0,
            TaskType.SUMMARIZATION: 0,
        }
        
        # Code generation signals
        if any(kw in content for kw in ["create", "write", "implement", "build", "new file"]):
            scores[TaskType.CODE_GENERATION] += 2
        
        # Code edit signals
        if any(kw in content for kw in ["refactor", "fix", "update", "modify", "change"]):
            scores[TaskType.CODE_EDIT] += 2
        
        # Reasoning signals
        if any(kw in content for kw in ["explain", "analyze", "design", "architecture"]):
            scores[TaskType.REASONING] += 2
        
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        return None
    
    def _estimate_avg_complexity(self, summary: ChatSummary) -> Complexity:
        """Estimate average complexity from summary signals."""
        score = 0
        
        # More files = higher complexity
        score += len(summary.files_mentioned) * 0.5
        
        # More languages = higher complexity
        score += len(summary.languages_used) * 0.3
        
        # Errors suggest complexity
        score += len(summary.error_patterns) * 0.2
        
        # Deep context = higher complexity
        if summary.context_depth == "deep":
            score += 2
        elif summary.context_depth == "medium":
            score += 1
        
        # Token count
        score += summary.total_tokens_used / 5000
        
        if score < 2:
            return Complexity.LOW
        elif score < 5:
            return Complexity.MEDIUM
        elif score < 10:
            return Complexity.HIGH
        else:
            return Complexity.VERY_HIGH
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics/themes from content."""
        topics = []
        topic_keywords = {
            "authentication": ["auth", "login", "oauth", "jwt", "password"],
            "database": ["database", "sql", "mongo", "postgres", "query"],
            "api": ["api", "rest", "graphql", "endpoint", "request"],
            "frontend": ["frontend", "ui", "react", "css", "component"],
            "backend": ["backend", "server", "express", "django"],
            "testing": ["test", "pytest", "jest", "coverage"],
            "deployment": ["deploy", "docker", "kubernetes", "ci/cd"],
            "performance": ["optimize", "performance", "speed", "cache"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in content for kw in keywords):
                topics.append(topic)
        
        return topics


class MCPRouter:
    """Main router class for intelligent model selection."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the router."""
        self.models: Dict[str, ModelCapabilities] = {}
        self.analyzer = QueryAnalyzer()
        self.routing_history: List[Dict] = []
        
        # Load default models
        self._load_default_models()
        
        # Load custom config if provided
        if config_path:
            self.load_config(config_path)
    
    def _load_default_models(self):
        """Load default model configurations with latest 2025 industry models."""
        
        # ================================================================
        # TIER 1: FLAGSHIP MODELS (Complex Architecture & Bug Hunts)
        # ================================================================
        
        # OpenAI GPT-5.2 - Latest flagship with enhanced reasoning
        self.register_model(ModelCapabilities(
            name="GPT-5.2",
            provider="openai",
            model_id="gpt-5.2",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=32768,
            context_window=256000,
            cost_per_1k_tokens_input=5.00,
            cost_per_1k_tokens_output=15.00,
            avg_latency_ms=1200,
            reasoning_quality=0.99,
            code_quality=0.98,
            speed_score=0.80,
            preferred_tasks=[TaskType.REASONING, TaskType.CODE_GENERATION, TaskType.MULTIMODAL],
            api_key_env_var="OPENAI_API_KEY"
        ))
        
        # Claude 4.5 Opus - Gold standard for complex refactoring
        self.register_model(ModelCapabilities(
            name="Claude 4.5 Opus",
            provider="anthropic",
            model_id="claude-4.5-opus",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=32768,
            context_window=200000,
            cost_per_1k_tokens_input=25.00,
            cost_per_1k_tokens_output=75.00,
            avg_latency_ms=2000,
            reasoning_quality=0.99,
            code_quality=0.99,
            speed_score=0.60,
            preferred_tasks=[TaskType.REASONING, TaskType.CODE_GENERATION, TaskType.CREATIVE],
            api_key_env_var="ANTHROPIC_API_KEY"
        ))
        
        # Claude 4.5 Sonnet - Default for most Cursor users
        self.register_model(ModelCapabilities(
            name="Claude 4.5 Sonnet",
            provider="anthropic",
            model_id="claude-4.5-sonnet",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=16384,
            context_window=200000,
            cost_per_1k_tokens_input=5.00,
            cost_per_1k_tokens_output=25.00,
            avg_latency_ms=800,
            reasoning_quality=0.97,
            code_quality=0.98,
            speed_score=0.88,
            preferred_tasks=[TaskType.CODE_GENERATION, TaskType.CODE_EDIT, TaskType.REASONING],
            api_key_env_var="ANTHROPIC_API_KEY"
        ))
        
        # ================================================================
        # TIER 2: REASONING MODELS (Chain of Thought / Thinking Models)
        # ================================================================
        
        # OpenAI o3 - Advanced reasoning model
        self.register_model(ModelCapabilities(
            name="o3",
            provider="openai",
            model_id="o3",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=False,
            max_tokens=32768,
            context_window=200000,
            cost_per_1k_tokens_input=10.00,
            cost_per_1k_tokens_output=40.00,
            avg_latency_ms=3000,
            reasoning_quality=0.99,
            code_quality=0.95,
            speed_score=0.50,
            preferred_tasks=[TaskType.REASONING],
            api_key_env_var="OPENAI_API_KEY"
        ))
        
        # OpenAI o3-mini (High) - Faster reasoning model
        self.register_model(ModelCapabilities(
            name="o3-mini (High)",
            provider="openai",
            model_id="o3-mini-high",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=False,
            max_tokens=16384,
            context_window=128000,
            cost_per_1k_tokens_input=1.50,
            cost_per_1k_tokens_output=6.00,
            avg_latency_ms=1500,
            reasoning_quality=0.95,
            code_quality=0.92,
            speed_score=0.70,
            preferred_tasks=[TaskType.REASONING, TaskType.CODE_GENERATION],
            api_key_env_var="OPENAI_API_KEY"
        ))
        
        # Claude 3.7 Sonnet - Thinking model option
        self.register_model(ModelCapabilities(
            name="Claude 3.7 Sonnet",
            provider="anthropic",
            model_id="claude-3.7-sonnet",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=16384,
            context_window=200000,
            cost_per_1k_tokens_input=4.00,
            cost_per_1k_tokens_output=20.00,
            avg_latency_ms=1200,
            reasoning_quality=0.96,
            code_quality=0.96,
            speed_score=0.78,
            preferred_tasks=[TaskType.REASONING, TaskType.CODE_GENERATION],
            api_key_env_var="ANTHROPIC_API_KEY"
        ))
        
        # ================================================================
        # TIER 3: NATIVE & FAST MODELS
        # ================================================================
        
        # Cursor Composer 1 - Native model optimized for Composer
        self.register_model(ModelCapabilities(
            name="Composer 1",
            provider="cursor",
            model_id="composer-1",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=False,
            max_tokens=16384,
            context_window=128000,
            cost_per_1k_tokens_input=0.10,
            cost_per_1k_tokens_output=0.30,
            avg_latency_ms=200,
            reasoning_quality=0.88,
            code_quality=0.92,
            speed_score=0.98,
            preferred_tasks=[TaskType.CODE_GENERATION, TaskType.CODE_EDIT],
            api_key_env_var="CURSOR_API_KEY"
        ))
        
        # Gemini 3 Pro - Massive context window for large projects
        self.register_model(ModelCapabilities(
            name="Gemini 3 Pro",
            provider="google",
            model_id="gemini-3-pro",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=32768,
            context_window=2000000,  # 2M token context window
            cost_per_1k_tokens_input=2.00,
            cost_per_1k_tokens_output=8.00,
            avg_latency_ms=1500,
            reasoning_quality=0.96,
            code_quality=0.94,
            speed_score=0.72,
            preferred_tasks=[TaskType.REASONING, TaskType.MULTIMODAL, TaskType.CODE_GENERATION],
            api_key_env_var="GOOGLE_API_KEY"
        ))
        
        # Gemini 3 Flash - Fast with large context
        self.register_model(ModelCapabilities(
            name="Gemini 3 Flash",
            provider="google",
            model_id="gemini-3-flash",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=16384,
            context_window=1000000,  # 1M token context window
            cost_per_1k_tokens_input=0.10,
            cost_per_1k_tokens_output=0.40,
            avg_latency_ms=400,
            reasoning_quality=0.88,
            code_quality=0.90,
            speed_score=0.95,
            preferred_tasks=[TaskType.QUESTION_ANSWERING, TaskType.SUMMARIZATION, TaskType.MULTIMODAL],
            api_key_env_var="GOOGLE_API_KEY"
        ))
        
        # ================================================================
        # TIER 4: LEGACY/BUDGET MODELS (Still Available)
        # ================================================================
        
        # GPT-4o - Previous flagship, still excellent
        self.register_model(ModelCapabilities(
            name="GPT-4o",
            provider="openai",
            model_id="gpt-4o",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=16384,
            context_window=128000,
            cost_per_1k_tokens_input=2.50,
            cost_per_1k_tokens_output=10.00,
            avg_latency_ms=800,
            reasoning_quality=0.95,
            code_quality=0.95,
            speed_score=0.85,
            preferred_tasks=[TaskType.REASONING, TaskType.CODE_GENERATION, TaskType.MULTIMODAL],
            api_key_env_var="OPENAI_API_KEY"
        ))
        
        # GPT-4o-mini - Fast and cost-effective
        self.register_model(ModelCapabilities(
            name="GPT-4o-mini",
            provider="openai",
            model_id="gpt-4o-mini",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=16384,
            context_window=128000,
            cost_per_1k_tokens_input=0.15,
            cost_per_1k_tokens_output=0.60,
            avg_latency_ms=400,
            reasoning_quality=0.80,
            code_quality=0.85,
            speed_score=0.95,
            preferred_tasks=[TaskType.CODE_EDIT, TaskType.QUESTION_ANSWERING, TaskType.SUMMARIZATION],
            api_key_env_var="OPENAI_API_KEY"
        ))
        
        # Claude 3.5 Sonnet - Previous default, still great
        self.register_model(ModelCapabilities(
            name="Claude 3.5 Sonnet",
            provider="anthropic",
            model_id="claude-3-5-sonnet-20241022",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=8192,
            context_window=200000,
            cost_per_1k_tokens_input=3.00,
            cost_per_1k_tokens_output=15.00,
            avg_latency_ms=1000,
            reasoning_quality=0.96,
            code_quality=0.97,
            speed_score=0.80,
            preferred_tasks=[TaskType.REASONING, TaskType.CODE_GENERATION],
            api_key_env_var="ANTHROPIC_API_KEY"
        ))
        
        # Claude 3.5 Haiku - Fast budget option
        self.register_model(ModelCapabilities(
            name="Claude 3.5 Haiku",
            provider="anthropic",
            model_id="claude-3-5-haiku-20241022",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=False,
            max_tokens=8192,
            context_window=200000,
            cost_per_1k_tokens_input=0.80,
            cost_per_1k_tokens_output=4.00,
            avg_latency_ms=400,
            reasoning_quality=0.85,
            code_quality=0.88,
            speed_score=0.92,
            preferred_tasks=[TaskType.CODE_EDIT, TaskType.QUESTION_ANSWERING, TaskType.SUMMARIZATION],
            api_key_env_var="ANTHROPIC_API_KEY"
        ))
        
        # Gemini 2.0 Pro - Current stable Gemini
        self.register_model(ModelCapabilities(
            name="Gemini 2.0 Pro",
            provider="google",
            model_id="gemini-2.0-pro",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=8192,
            context_window=2000000,
            cost_per_1k_tokens_input=1.25,
            cost_per_1k_tokens_output=5.00,
            avg_latency_ms=1200,
            reasoning_quality=0.94,
            code_quality=0.92,
            speed_score=0.75,
            preferred_tasks=[TaskType.REASONING, TaskType.MULTIMODAL, TaskType.CODE_GENERATION],
            api_key_env_var="GOOGLE_API_KEY"
        ))
        
        # Gemini 2.0 Flash - Fast Gemini option
        self.register_model(ModelCapabilities(
            name="Gemini 2.0 Flash",
            provider="google",
            model_id="gemini-2.0-flash",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=True,
            max_tokens=8192,
            context_window=1000000,
            cost_per_1k_tokens_input=0.075,
            cost_per_1k_tokens_output=0.30,
            avg_latency_ms=500,
            reasoning_quality=0.82,
            code_quality=0.85,
            speed_score=0.90,
            preferred_tasks=[TaskType.QUESTION_ANSWERING, TaskType.SUMMARIZATION, TaskType.MULTIMODAL],
            api_key_env_var="GOOGLE_API_KEY"
        ))
        
        # DeepSeek V3 - Open source powerhouse
        self.register_model(ModelCapabilities(
            name="DeepSeek V3",
            provider="deepseek",
            model_id="deepseek-v3",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=False,
            max_tokens=8192,
            context_window=128000,
            cost_per_1k_tokens_input=0.14,
            cost_per_1k_tokens_output=0.28,
            avg_latency_ms=600,
            reasoning_quality=0.92,
            code_quality=0.94,
            speed_score=0.88,
            preferred_tasks=[TaskType.CODE_GENERATION, TaskType.REASONING],
            api_key_env_var="DEEPSEEK_API_KEY"
        ))
        
        # DeepSeek R1 - Reasoning-focused open source
        self.register_model(ModelCapabilities(
            name="DeepSeek R1",
            provider="deepseek",
            model_id="deepseek-r1",
            supports_reasoning=True,
            supports_code=True,
            supports_streaming=True,
            supports_multimodal=False,
            max_tokens=16384,
            context_window=128000,
            cost_per_1k_tokens_input=0.55,
            cost_per_1k_tokens_output=2.19,
            avg_latency_ms=1500,
            reasoning_quality=0.96,
            code_quality=0.92,
            speed_score=0.70,
            preferred_tasks=[TaskType.REASONING, TaskType.CODE_GENERATION],
            api_key_env_var="DEEPSEEK_API_KEY"
        ))
    
    def register_model(self, model: ModelCapabilities):
        """Register a model with the router."""
        self.models[model.model_id] = model
        logger.info(f"Registered model: {model.name} ({model.model_id})")
    
    def route(
        self,
        query: str,
        context: Optional[QueryContext] = None,
        strategy: str = "balanced",
        chat_summary: Optional[ChatSummary] = None,
        chat_history: Optional[List[Dict]] = None,
        summary_text: Optional[str] = None
    ) -> RoutingDecision:
        """
        Route a query to the best model.
        
        Args:
            query: The user query
            context: Optional pre-analyzed context
            strategy: Routing strategy ("balanced", "cost", "speed", "quality")
            chat_summary: Optional pre-analyzed chat summary
            chat_history: Optional list of chat messages to analyze
            summary_text: Optional text summary (e.g., from ledger.md)
        
        Returns:
            RoutingDecision with selected model and reasoning
        """
        # Analyze query if context not provided
        if context is None:
            context = self.analyzer.analyze(query)
        
        # Analyze chat summary if not provided but history is
        if chat_summary is None and chat_history is not None:
            chat_analyzer = ChatSummaryAnalyzer()
            chat_summary = chat_analyzer.analyze(chat_history)
        elif chat_summary is None and summary_text is not None:
            chat_analyzer = ChatSummaryAnalyzer()
            chat_summary = chat_analyzer.analyze_from_text(summary_text)
        
        # Attach chat summary to context
        context.chat_summary = chat_summary
        
        # Filter compatible models
        compatible_models = self._filter_compatible_models(context)
        
        if not compatible_models:
            raise ValueError("No compatible models found for this query")
        
        # Score models based on strategy (now includes chat summary signals)
        scored_models = self._score_models(compatible_models, context, strategy)
        
        # Select best model
        best_model, score = scored_models[0]
        
        # Get alternatives
        alternatives = [model for model, _ in scored_models[1:4]]  # Top 3 alternatives
        
        # Estimate cost and latency
        estimated_cost = self._estimate_cost(best_model, context)
        estimated_latency = best_model.avg_latency_ms
        
        # Generate reasoning (includes chat summary context)
        reasoning = self._generate_reasoning(best_model, context, strategy, score)
        
        decision = RoutingDecision(
            selected_model=best_model,
            confidence=score,
            reasoning=reasoning,
            alternatives=alternatives,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency
        )
        
        # Log routing decision
        self.routing_history.append({
            "query": query,
            "decision": decision.selected_model.model_id,
            "confidence": decision.confidence,
            "timestamp": time.time(),
            "chat_context": chat_summary.to_dict() if chat_summary else None
        })
        
        return decision
    
    def _filter_compatible_models(self, context: QueryContext) -> List[ModelCapabilities]:
        """Filter models that are compatible with the query requirements."""
        compatible = []
        
        for model in self.models.values():
            # Note: API key check is optional for routing (only needed for execution)
            # This allows routing decisions even when API keys aren't configured
            
            # Check streaming requirement
            if context.requires_streaming and not model.supports_streaming:
                continue
            
            # Check multimodal requirement
            if context.requires_multimodal and not model.supports_multimodal:
                continue
            
            # Check embeddings requirement
            if context.requires_embeddings and not model.supports_embeddings:
                continue
            
            # Check task compatibility
            if context.task_type and context.task_type in model.unsuitable_tasks:
                continue
            
            # Check context window
            if context.estimated_tokens > model.context_window:
                continue
            
            compatible.append(model)
        
        return compatible
    
    def _score_models(
        self,
        models: List[ModelCapabilities],
        context: QueryContext,
        strategy: str
    ) -> List[tuple]:
        """Score models based on strategy, context, and chat summary."""
        scored = []
        chat_summary = context.chat_summary
        
        for model in models:
            score = 0.0
            
            # Base score from quality
            if context.task_type == TaskType.REASONING:
                score += model.reasoning_quality * 0.4
            elif context.task_type in [TaskType.CODE_GENERATION, TaskType.CODE_EDIT]:
                score += model.code_quality * 0.4
            else:
                score += (model.reasoning_quality + model.code_quality) / 2 * 0.3
            
            # Task preference bonus
            if context.task_type in model.preferred_tasks:
                score += 0.2
            
            # Strategy-based scoring
            if strategy == "cost":
                # Prefer lower cost
                cost_score = 1.0 / (1.0 + model.cost_per_1k_tokens_input / 10.0)
                score += cost_score * 0.3
            elif strategy == "speed":
                # Prefer faster models
                score += model.speed_score * 0.3
            elif strategy == "quality":
                # Prefer higher quality
                quality_score = (model.reasoning_quality + model.code_quality) / 2
                score += quality_score * 0.3
            else:  # balanced
                # Balance cost, speed, and quality
                cost_score = 1.0 / (1.0 + model.cost_per_1k_tokens_input / 10.0)
                score += (cost_score * 0.1 + model.speed_score * 0.1 + 
                         (model.reasoning_quality + model.code_quality) / 2 * 0.1)
            
            # Complexity matching
            if context.complexity == Complexity.VERY_HIGH:
                score += model.reasoning_quality * 0.1
            elif context.complexity == Complexity.LOW:
                score += model.speed_score * 0.1
            
            # ========== CHAT SUMMARY SIGNALS ==========
            if chat_summary:
                # Deep context needs larger context window models
                if chat_summary.context_depth == "deep":
                    if model.context_window >= 200000:
                        score += 0.15
                    elif model.context_window >= 128000:
                        score += 0.08
                
                # Continuity requirement favors high-quality models
                if chat_summary.requires_continuity:
                    score += model.reasoning_quality * 0.1
                
                # Match dominant task type from history
                if chat_summary.dominant_task_type:
                    if chat_summary.dominant_task_type in model.preferred_tasks:
                        score += 0.1  # Bonus for matching chat history pattern
                
                # Error-heavy sessions benefit from reasoning models
                if chat_summary.success_rate < 0.8:
                    if model.reasoning_quality >= 0.95:
                        score += 0.1  # Prefer high-reasoning models for debugging
                
                # High complexity history suggests need for capable models
                if chat_summary.avg_complexity in [Complexity.HIGH, Complexity.VERY_HIGH]:
                    quality = (model.reasoning_quality + model.code_quality) / 2
                    if quality >= 0.95:
                        score += 0.1
                
                # Multiple files mentioned = multi-file task
                if len(chat_summary.files_mentioned) >= 5:
                    # Prefer models good at code (multi-file refactoring)
                    score += model.code_quality * 0.05
                
                # Long sessions benefit from context-efficient models
                if chat_summary.session_duration_mins > 30:
                    if model.context_window >= 128000:
                        score += 0.05
                
                # Multiple languages = complex polyglot task
                if len(chat_summary.languages_used) >= 2:
                    score += model.code_quality * 0.05
                
                # Topic-specific boosts
                if "authentication" in chat_summary.topics or "database" in chat_summary.topics:
                    # Security/data tasks need careful reasoning
                    score += model.reasoning_quality * 0.05
            
            # Normalize score to 0-1 range
            score = min(1.0, score)
            scored.append((model, score))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _estimate_cost(self, model: ModelCapabilities, context: QueryContext) -> float:
        """Estimate cost for the query."""
        input_cost = (context.estimated_tokens / 1000) * model.cost_per_1k_tokens_input
        # Assume output is ~50% of input for estimation
        output_tokens = context.estimated_tokens // 2
        output_cost = (output_tokens / 1000) * model.cost_per_1k_tokens_output
        return input_cost + output_cost
    
    def _generate_reasoning(
        self,
        model: ModelCapabilities,
        context: QueryContext,
        strategy: str,
        score: float
    ) -> str:
        """Generate human-readable reasoning for the selection."""
        reasons = []
        
        if context.task_type in model.preferred_tasks:
            reasons.append(f"Model is optimized for {context.task_type.value} tasks")
        
        if strategy == "cost":
            reasons.append("Selected for cost efficiency")
        elif strategy == "speed":
            reasons.append("Selected for low latency")
        elif strategy == "quality":
            reasons.append("Selected for highest quality")
        else:
            reasons.append("Selected for balanced performance")
        
        if context.complexity == Complexity.VERY_HIGH:
            reasons.append("High complexity requires strong reasoning capabilities")
        
        if context.requires_streaming:
            reasons.append("Streaming support required")
        
        if context.requires_multimodal:
            reasons.append("Multimodal capabilities required")
        
        # Add chat summary context to reasoning
        chat_summary = context.chat_summary
        if chat_summary:
            if chat_summary.context_depth == "deep":
                reasons.append(f"Deep context ({chat_summary.total_tokens_used:,} tokens) benefits from {model.context_window:,} context window")
            
            if chat_summary.requires_continuity:
                reasons.append("Task requires continuity with chat history")
            
            if chat_summary.dominant_task_type:
                reasons.append(f"Chat history shows {chat_summary.dominant_task_type.value} pattern")
            
            if chat_summary.success_rate < 0.8:
                reasons.append("Debugging session detected - high reasoning model selected")
            
            if len(chat_summary.files_mentioned) >= 5:
                reasons.append(f"Multi-file task ({len(chat_summary.files_mentioned)} files)")
            
            if len(chat_summary.languages_used) >= 2:
                reasons.append(f"Polyglot task ({', '.join(chat_summary.languages_used[:3])})")
            
            if chat_summary.topics:
                reasons.append(f"Topics: {', '.join(chat_summary.topics[:3])}")
        
        return "; ".join(reasons) if reasons else f"Selected {model.name} with confidence {score:.2f}"
    
    def load_config(self, config_path: str):
        """Load model configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for model_config in config.get("models", []):
            model = ModelCapabilities(**model_config)
            self.register_model(model)
    
    def save_config(self, config_path: str):
        """Save current model configuration to JSON file."""
        config = {
            "models": [model.to_dict() for model in self.models.values()]
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing decisions."""
        if not self.routing_history:
            return {"total_routes": 0}
        
        model_counts = {}
        for entry in self.routing_history:
            model_id = entry["decision"]
            model_counts[model_id] = model_counts.get(model_id, 0) + 1
        
        return {
            "total_routes": len(self.routing_history),
            "model_usage": model_counts,
            "avg_confidence": sum(e["confidence"] for e in self.routing_history) / len(self.routing_history)
        }

