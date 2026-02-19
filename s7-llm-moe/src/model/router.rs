/// Deterministic MoE Router — S7-LLM-MOE-140M
///
/// Routing is purely lexical: the router scans the raw decoded token string
/// for domain-specific trigger patterns.  No learned weights.  No softmax.
/// The routing function is a pure bijection from (tokens) → ExpertDomain,
/// making it V6-compliant (same input → same route, always).
///
/// Fold binding: ⟁COMPUTE_FOLD⟁ (routing happens inside BATCH lane compute).
/// CM-1 gate: must be applied before calling `route()` by the caller.

/// The four expert domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertDomain {
    Code    = 0,
    Math    = 1,
    Reason  = 2,
    General = 3,
}

impl ExpertDomain {
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            ExpertDomain::Code    => "CODE",
            ExpertDomain::Math    => "MATH",
            ExpertDomain::Reason  => "REASON",
            ExpertDomain::General => "GENERAL",
        }
    }
}

/// Code-domain trigger tokens.
const CODE_TRIGGERS: &[&str] = &[
    "{", "}", "def ", "fn ", "class ", "::", ";",
    "import ", "struct ", "=>", "let ", "var ", "func ",
    "for(", "while(", "#include", "pub fn", "async fn",
    "return ", "void ", "int ", "bool ", "string ",
];

/// Math-domain trigger tokens.
const MATH_TRIGGERS: &[&str] = &[
    "=", "Answer:", "Therefore", "∑", "∫", "√", "∞",
    "solve", "equation", "calculate", "formula",
    " + ", " - ", " × ", " ÷ ", " / ", " * ",
    "proof:", "lemma", "theorem", "corollary",
];

/// Reason-domain trigger tokens.
const REASON_TRIGGERS: &[&str] = &[
    "why ", "explain ", "because ", "therefore ", "reason ",
    "step ", "if ", " then ", "conclude", "implies",
    "first,", "second,", "finally,", "thus ",
    "given that", "it follows", "consider ",
];

/// Code digit characters (heuristic for math detection).
const DIGIT_CHARS: &[char] = &['0','1','2','3','4','5','6','7','8','9'];

/// The router.  No mutable state — routing is a pure function.
pub struct DeterministicRouter;

impl DeterministicRouter {
    pub fn new() -> Self {
        DeterministicRouter
    }

    /// Route a decoded token string to an expert domain.
    ///
    /// Priority order (first match wins):
    ///   1. Code  — structural syntax tokens
    ///   2. Math  — numeric or equation tokens
    ///   3. Reason — discourse tokens
    ///   4. General — fallback
    pub fn route(&self, token_text: &str) -> ExpertDomain {
        if self.matches_code(token_text) {
            return ExpertDomain::Code;
        }
        if self.matches_math(token_text) {
            return ExpertDomain::Math;
        }
        if self.matches_reason(token_text) {
            return ExpertDomain::Reason;
        }
        ExpertDomain::General
    }

    /// Route over a full decoded sequence (for context-aware token routing).
    /// Uses a sliding window of the last 64 decoded bytes.
    pub fn route_context(&self, context: &str) -> ExpertDomain {
        let window = if context.len() > 64 {
            &context[context.len() - 64..]
        } else {
            context
        };
        self.route(window)
    }

    fn matches_code(&self, text: &str) -> bool {
        CODE_TRIGGERS.iter().any(|t| text.contains(t))
    }

    fn matches_math(&self, text: &str) -> bool {
        // Trigger keyword match.
        if MATH_TRIGGERS.iter().any(|t| text.contains(t)) {
            return true;
        }
        // Digit density heuristic: >30% digit chars → likely math.
        let total = text.chars().count();
        if total == 0 { return false; }
        let digits = text.chars().filter(|c| DIGIT_CHARS.contains(c)).count();
        digits * 10 > total * 3
    }

    fn matches_reason(&self, text: &str) -> bool {
        let lower = text.to_ascii_lowercase();
        REASON_TRIGGERS.iter().any(|t| lower.contains(t))
    }
}

impl Default for DeterministicRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_code() {
        let r = DeterministicRouter::new();
        assert_eq!(r.route("def fibonacci(n):"), ExpertDomain::Code);
        assert_eq!(r.route("fn main() {"), ExpertDomain::Code);
    }

    #[test]
    fn routes_math() {
        let r = DeterministicRouter::new();
        assert_eq!(r.route("Therefore x = 42"), ExpertDomain::Math);
        assert_eq!(r.route("Answer: 123"), ExpertDomain::Math);
    }

    #[test]
    fn routes_reason() {
        let r = DeterministicRouter::new();
        assert_eq!(r.route("explain why the sky is blue"), ExpertDomain::Reason);
    }

    #[test]
    fn routes_general() {
        let r = DeterministicRouter::new();
        assert_eq!(r.route("hello world"), ExpertDomain::General);
    }

    #[test]
    fn deterministic_same_input_same_output() {
        let r = DeterministicRouter::new();
        let text = "import torch; class Model:";
        assert_eq!(r.route(text), r.route(text));
    }
}
