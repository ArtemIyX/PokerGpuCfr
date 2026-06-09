# Code Documentation Guide

This guide establishes the mandatory standards for documenting all code within the PokerGPU project (and any future modules). Consistent and comprehensive documentation is crucial for maintainability, onboarding new team members, and ensuring long-term project stability.

---

## 🎯 Guiding Principles

1.  **Goal:** Every module, class, and public function/method must be fully understandable by a developer unfamiliar with the codebase based solely on its docstrings.
2.  **Clarity over Brevity:** Prioritize descriptive accuracy over minimal token usage when writing documentation.
3.  **Mandatory Tooling:** Utilize type hinting extensively (Python 3.7+) and ensure all functions adhere to PEP 484 standards.
4.  **Docstring Standard (Primary):** Adopt the **NumPy Docstring Format**. This is mandatory for internal consistency, readability, and integration with scientific tools.


---

## 📚 Structure & Guidelines

### 1. Module Documentation (File Level)

Every Python file (`*.py`) that contains logic or definitions must begin with a module-level docstring immediately following the necessary imports.

**Content Requirements:**
*   A concise, one-sentence summary of what the module does.
*   A more detailed paragraph explaining its purpose, architecture role within the larger system (e.g., "This module handles all aspects of hand range evaluation."), and any key dependencies.
*   Any external files or resources it interacts with.

**Example:**
```python
"""pokergpu/state.py
    Manages the current game state, including board cards, pot size, 
    and active player information. This module is central to all decision-making 
    processes and must be updated whenever a game rule changes.

    :ivar current_board: The list of community cards on the table.
"""
```

### 2. Class Documentation

Every class definition must include a comprehensive docstring that clearly defines its scope, internal state (attributes), and responsibilities.

**Content Requirements:**
*   **Overall Purpose:** What does this class represent? Why was it created?
*   **Internal Structure/State (`:ivar`):** Document all significant instance variables (`self.attribute`). Include their type and a description of their expected range or purpose.
*   **Inheritance:** If the class inherits from another, document the reason for inheritance and any deviations from the parent's behavior.

### 3. Function & Method Documentation (Public API)

Every public function and method signature must be accompanied by a docstring detailing its contract: what inputs it expects, what output it guarantees, and what conditions might cause failure.
---

## 📚 Structure & Guidelines

### 1. Module Documentation (File Level)

Every Python file (`*.py`) that contains logic or definitions must begin with a module-level docstring immediately following the necessary imports.

**Content Requirements:**
*   A concise, one-sentence summary of what the module does.
*   A more detailed paragraph explaining its purpose, architecture role within the larger system (e.g., "This module handles all aspects of hand range evaluation."), and any key dependencies.
*   Any external files or resources it interacts with.

**Example:**
```python
"""pokergpu/state.py
    Manages the current game state, including board cards, pot size, 
    and active player information. This module is central to all decision-making 
    processes and must be updated whenever a game rule changes.

    :ivar current_board: The list of community cards on the table.
"""
```

### 2. Class Documentation

Every class definition must include a comprehensive docstring that clearly defines its scope, internal state (attributes), and responsibilities.

**Content Requirements:**
*   **Overall Purpose:** What does this class represent? Why was it created?
*   **Internal Structure/State (`:ivar`):** Document all significant instance variables (`self.attribute`). Include their type and a description of their expected range or purpose.
*   **Inheritance:** If the class inherits from another, document the reason for inheritance and any deviations from the parent's behavior.

### 3. Function & Method Documentation (Public API)

Every public function and method signature must be accompanied by a docstring detailing its contract: what inputs it expects, what output it guarantees, and what conditions might cause failure.

**Structure Requirements:**
*   **Summary:** One line description of the function's action.
*   **Parameters (`:param`):** For every parameter:
    *   Type (must match type hint).
    *   A clear, technical explanation of its role and acceptable values.
*   **Returns (`:return:`):**
    *   Type.
    *   Explanation of the returned value(s) and what they represent.
*   **Raises (`:raises:`):** Document any exceptions that this function is expected to raise (e.g., `ValueError`, `TypeError`).

---

## 📚 Structure & Guidelines

### 1. Module Documentation (File Level)

Every Python file (`*.py`) that contains logic or definitions must begin with a module-level docstring immediately following the necessary imports.

**Content Requirements:**
*   A concise, one-sentence summary of what the module does.
*   A more detailed paragraph explaining its purpose, architecture role within the larger system (e.g., "This module handles all aspects of hand range evaluation."), and any key dependencies.
*   Any external files or resources it interacts with.

**Example:**
```python
"""pokergpu/state.py
    Manages the current game state, including board cards, pot size, 
    and active player information. This module is central to all decision-making 
    processes and must be updated whenever a game rule changes.

    :ivar current_board: The list of community cards on the table.
"""
```

### 2. Class Documentation

Every class definition must include a comprehensive docstring that clearly defines its scope, internal state (attributes), and responsibilities.

**Content Requirements:**
*   **Overall Purpose:** What does this class represent? Why was it created?
*   **Internal Structure/State (`:ivar`):** Document all significant instance variables (`self.attribute`). Include their type and a description of their expected range or purpose.
*   **Inheritance:** If the class inherits from another, document the reason for inheritance and any deviations from the parent's behavior.

### 3. Function & Method Documentation (Public API)

Every public function and method signature must be accompanied by a docstring detailing its contract: what inputs it expects, what output it guarantees, and what conditions might cause failure.

**Structure Requirements:**
*   **Summary:** One line description of the function's action.
*   **Parameters (`:param`):** For every parameter:
    *   Type (must match type hint).
    *   A clear, technical explanation of its role and acceptable values.
*   **Returns (`:return:`):**
    *   Type.
    *   Explanation of the returned value(s) and what they represent.
*   **Raises (`:raises:`):** Document any exceptions that this function is expected to raise (e.g., `ValueError`, `TypeError`).
---

## 📚 Structure & Guidelines

### 1. Module Documentation (File Level)

Every Python file (`*.py`) that contains logic or definitions must begin with a module-level docstring immediately following the necessary imports.

**Content Requirements:**
*   A concise, one-sentence summary of what the module does.
*   A more detailed paragraph explaining its purpose, architecture role within the larger system (e.g., "This module handles all aspects of hand range evaluation."), and any key dependencies.
*   Any external files or resources it interacts with.

**Example:**
```python
"""pokergpu/state.py
    Manages the current game state, including board cards, pot size, 
    and active player information. This module is central to all decision-making 
    processes and must be updated whenever a game rule changes.

    :ivar current_board: The list of community cards on the table.
"""
```

### 2. Class Documentation

Every class definition must include a comprehensive docstring that clearly defines its scope, internal state (attributes), and responsibilities.

**Content Requirements:**
*   **Overall Purpose:** What does this class represent? Why was it created?
*   **Internal Structure/State (`:ivar`):** Document all significant instance variables (`self.attribute`). Include their type and a description of their expected range or purpose.
*   **Inheritance:** If the class inherits from another, document the reason for inheritance and any deviations from the parent's behavior.

### 3. Function & Method Documentation (Public API)

Every public function and method signature must be accompanied by a docstring detailing its contract: what inputs it expects, what output it guarantees, and what conditions might cause failure.

**Structure Requirements:**
*   **Summary:** One line description of the function's action.
*   **Parameters (`:param`):** For every parameter:
    *   Type (must match type hint).
    *   A clear, technical explanation of its role and acceptable values.
*   **Returns (`:return:`):**
    *   Type.
    *   Explanation of the returned value(s) and what they represent.
*   **Raises (`:raises:`):** Document any exceptions that this function is expected to raise (e.g., `ValueError`, `TypeError`).

---

## 🛠️ Implementation Examples & Best Practices

### A. Docstring Formatting (NumPy Style - Primary Standard)

```python
def calculate_leduc(history, player_id: int) -> float:
    """Calculates the expected value for a given player history using Leduc rules.

    This function processes the betting and card history to determine 
    the maximum potential EV achievable by the specified player in the current game state.

    Parameters
    ----------
    history : list[tuple]
        A chronological list of (action, amount) tuples representing 
        all actions taken so far.
    player_id : int
        The unique identifier of the player whose EV is being calculated.

    Returns
    -------
    float
        The calculated maximum expected value (EV). The result must be non-negative.

    Raises
    ------
    ValueError
        If history is empty or if the provided player_id does not exist 
        in the current game context.
    """
    # Implementation goes here...
    pass
```

### B. Alternative/Complementary Docstring Styles (Doxygen/Sphinx Compatibility)

While NumPy remains the primary standard, developers should be aware of other formats:

*   **Google Style:** Uses a concise parameter block (`Args:`). Good for quick reading but less structured than NumPy.
    Example: `Args: history (list[tuple]): A chronological list...`
*   **Doxygen/JavaDoc Style:** Often uses `@param`, `@return`, and `@throws`. This structure is crucial when interfacing with non-Python compiled languages or using specific documentation generators that favor these tags.

### C. Code Style & Readability Best Practices (Mandatory)

1.  **Constants:** Global constants must use `CAPITAL_SNAKE_CASE` (e.g., `MAX_DEPTH`).
2.  **Variables/Functions:** Must adhere to Python's standard `snake_case`.
3.  **Documentation Location:** Docstrings *must* be the first executable statement inside a module, class, or function body.

---

## 🧪 Testing Documentation & Coverage (Mandatory)

The relationship between code and its tests must be explicitly documented:

1.  **Test Linking:** For complex utility methods, document the primary test file(s) that cover the logic within the docstring's `Testing Notes` section (as shown in the example).
2.  **Coverage Goals:** The module documentation should ideally state expected code coverage goals and reference CI/CD checks for compliance.

```python
"""pokergpu/utils.py
... [Rest of Module Docstring] ...

Testing Notes:
- Core logic is covered by tests/test_utils.py (Minimum 95% line coverage required).
- Edge cases related to tie handling are specifically checked in tests/test_signatures.py.
"""
```

Every Python file (`*.py`) that contains logic or definitions must begin with a module-level docstring immediately following the necessary imports.

**Content Requirements:**
*   A concise, one-sentence summary of what the module does.
*   A more detailed paragraph explaining its purpose, architecture role within the larger system (e.g., "This module handles all aspects of hand range evaluation."), and any key dependencies.
*   Any external files or resources it interacts with.

**Example:**
```python
"""pokergpu/state.py
    Manages the current game state, including board cards, pot size, 
    and active player information. This module is central to all decision-making 
    processes and must be updated whenever a game rule changes.

    :ivar current_board: The list of community cards on the table.
"""
```

### 2. Class Documentation

Every class definition must include a comprehensive docstring that clearly defines its scope, internal state (attributes), and responsibilities.

**Content Requirements:**
*   **Overall Purpose:** What does this class represent? Why was it created?
*   **Internal Structure/State (`:ivar`):** Document all significant instance variables (`self.attribute`). Include their type and a description of their expected range or purpose.
*   **Inheritance:** If the class inherits from another, document the reason for inheritance and any deviations from the parent's behavior.

### 3. Function & Method Documentation (Public API)

Every public function and method signature must be accompanied by a docstring detailing its contract: what inputs it expects, what output it guarantees, and what conditions might cause failure.

**Structure Requirements:**
*   **Summary:** One line description of the function's action.
*   **Parameters (`:param`):** For every parameter:
    *   Type (must match type hint).
    *   A clear, technical explanation of its role and acceptable values.
*   **Returns (`:return:`):**
    *   Type.
    *   Explanation of the returned value(s) and what they represent.
*   **Raises (`:raises:`):** Document any exceptions that this function is expected to raise (e.g., `ValueError`, `TypeError`).

---

## 🛠️ Implementation Examples & Best Practices

### A. Docstring Formatting (NumPy Style)

```python
def calculate_leduc(history, player_id: int) -> float:
    """Calculates the expected value for a given player history using Leduc rules.

    This function processes the betting and card history to determine 
    the maximum potential EV achievable by the specified player in the current game state.

    Parameters
    ----------
    history : list[tuple]
        A chronological list of (action, amount) tuples representing 
        all actions taken so far.
    player_id : int
        The unique identifier of the player whose EV is being calculated.

    Returns
    -------
    float
        The calculated maximum expected value (EV). The result must be non-negative.

    Raises
    ------
    ValueError
        If history is empty or if the provided player_id does not exist 
        in the current game context.
    """
    # Implementation goes here...
    pass
```

### B. Type Hinting & Signature Compliance (MANDATORY)

All function signatures must be fully type-hinted to improve editor support and static analysis.

*   **Mandatory:** Use explicit types for all arguments (`arg: type`) and return values (`-> type`).
*   **Complex Types:** For containers, use `typing` module annotations (e.g., `list[str]`, `dict[int, float]`).

### C. Code Style & Readability

1.  **Constants:** Global constants should be named in `CAPITAL_SNAKE_CASE`.
2.  **Variables/Functions:** Should use `snake_case`.
3.  **Documentation Location:** Docstrings must be the first executable statement inside a module, class, or function body.

---
