# Coding Standards - Auto Workflow Platform

## 🎯 Mục đích

Document này định nghĩa coding standards và best practices cho toàn bộ Auto Workflow Platform. Tất cả contributors phải tuân thủ các quy tắc này.

## 📋 Mục lục

- [General Principles](#general-principles)
- [Python Standards](#python-standards)
- [TypeScript/JavaScript Standards](#typescriptjavascript-standards)
- [Database Standards](#database-standards)
- [API Design](#api-design)
- [Testing Standards](#testing-standards)
- [Security Guidelines](#security-guidelines)
- [Git Workflow](#git-workflow)
- [Documentation](#documentation)

---

## General Principles

### SOLID Principles

- **S**ingle Responsibility - Mỗi class/function có một trách nhiệm duy nhất
- **O**pen/Closed - Open for extension, closed for modification
- **L**iskov Substitution - Subclasses phải thay thế được base classes
- **I**nterface Segregation - Nhiều interfaces nhỏ thay vì một interface lớn
- **D**ependency Inversion - Depend on abstractions, not concretions

### DRY (Don't Repeat Yourself)

- Tránh code duplication
- Extract common logic vào shared functions/classes
- Sử dụng inheritance/composition hợp lý

### KISS (Keep It Simple, Stupid)

- Ưu tiên simplicity over complexity
- Code dễ đọc > code "clever"
- Avoid premature optimization

### YAGNI (You Aren't Gonna Need It)

- Chỉ implement những gì cần thiết bây giờ
- Không build features "cho tương lai"
- Refactor khi cần, không phải trước

---

## Python Standards

### Code Style

- **Formatter:** Black (line length: 100)
- **Linter:** Ruff
- **Type Checker:** mypy
- **Import Sorter:** isort

```python
# ✅ GOOD: Clean, readable, typed
from typing import List, Optional
from datetime import datetime

async def get_workflows(
    user_id: str,
    status: Optional[str] = None,
    limit: int = 100
) -> List[Workflow]:
    """Get workflows for a user.

    Args:
        user_id: The user ID
        status: Optional status filter
        limit: Maximum number of results

    Returns:
        List of Workflow objects
    """
    query = select(WorkflowModel).where(WorkflowModel.user_id == user_id)
    if status:
        query = query.where(WorkflowModel.status == status)
    query = query.limit(limit)

    result = await db.execute(query)
    return result.scalars().all()

# ❌ BAD: No types, no docstring, poor formatting
def get_workflows(user_id, status=None, limit=100):
    query = select(WorkflowModel).where(WorkflowModel.user_id==user_id)
    if status:
        query=query.where(WorkflowModel.status==status)
    return db.execute(query.limit(limit)).scalars().all()
```

### Naming Conventions

```python
# Classes: PascalCase
class WorkflowExecutor:
    pass

# Functions/Variables: snake_case
def execute_workflow(workflow_id: str):
    execution_result = ...

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Private: prefix với _
class MyClass:
    def _internal_method(self):
        pass
```

### Type Hints

```python
# ✅ GOOD: Always use type hints
def process_data(
    items: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Tuple[List[str], int]:
    ...

# ❌ BAD: No type hints
def process_data(items, threshold=0.5):
    ...
```

### Error Handling

```python
# ✅ GOOD: Specific exceptions, proper cleanup
from contextlib import asynccontextmanager

class WorkflowError(Exception):
    """Base exception for workflow errors"""
    pass

class WorkflowNotFoundError(WorkflowError):
    """Workflow not found"""
    pass

async def execute_workflow(workflow_id: str) -> Result:
    try:
        workflow = await get_workflow(workflow_id)
        if not workflow:
            raise WorkflowNotFoundError(f"Workflow {workflow_id} not found")

        async with resource_manager() as resources:
            result = await _execute(workflow, resources)

        return result

    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise WorkflowError("Execution failed") from e

# ❌ BAD: Catch-all, no cleanup
def execute_workflow(workflow_id):
    try:
        return execute(workflow_id)
    except:
        pass
```

### Async/Await

```python
# ✅ GOOD: Proper async usage
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_item(session, id) for id in ids]
        return await asyncio.gather(*tasks)

# ❌ BAD: Blocking in async function
async def fetch_data():
    return requests.get(url)  # Blocks event loop!
```

### Documentation

```python
# ✅ GOOD: Google-style docstrings
def calculate_score(
    metrics: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """Calculate weighted score from metrics.

    Args:
        metrics: Dictionary of metric names to values
        weights: Dictionary of metric names to weights

    Returns:
        Weighted score as float

    Raises:
        ValueError: If metrics and weights don't match

    Example:
        >>> calculate_score(
        ...     {"accuracy": 0.9, "speed": 0.8},
        ...     {"accuracy": 0.7, "speed": 0.3}
        ... )
        0.87
    """
    if set(metrics.keys()) != set(weights.keys()):
        raise ValueError("Metrics and weights must have same keys")

    return sum(metrics[k] * weights[k] for k in metrics)
```

---

## TypeScript/JavaScript Standards

### Code Style

- **Formatter:** Prettier
- **Linter:** ESLint (Airbnb config)
- **Type Checker:** TypeScript strict mode

```typescript
// ✅ GOOD: Typed, functional component
import { FC, useState, useEffect } from 'react';

interface WorkflowListProps {
  userId: string;
  onSelect: (workflowId: string) => void;
}

export const WorkflowList: FC<WorkflowListProps> = ({ userId, onSelect }) => {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchWorkflows = async () => {
      try {
        const data = await api.workflows.list(userId);
        setWorkflows(data);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    };

    fetchWorkflows();
  }, [userId]);

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <div className="workflow-list">
      {workflows.map((workflow) => (
        <WorkflowCard
          key={workflow.id}
          workflow={workflow}
          onClick={() => onSelect(workflow.id)}
        />
      ))}
    </div>
  );
};

// ❌ BAD: No types, poor structure
export const WorkflowList = ({ userId, onSelect }) => {
  const [workflows, setWorkflows] = useState([]);

  useEffect(() => {
    fetch(`/api/workflows?userId=${userId}`)
      .then(r => r.json())
      .then(setWorkflows);
  }, []);

  return <div>{workflows.map(w => <div onClick={() => onSelect(w.id)}>{w.name}</div>)}</div>;
};
```

### Naming Conventions

```typescript
// Interfaces/Types: PascalCase
interface User {
  id: string;
  name: string;
}

// Functions/Variables: camelCase
const fetchUserData = async (userId: string): Promise<User> => { ... };

// Constants: UPPER_SNAKE_CASE
const API_BASE_URL = 'https://api.example.com';

// Components: PascalCase
const UserProfile: FC = () => { ... };

// Enums: PascalCase
enum WorkflowStatus {
  Draft = 'draft',
  Running = 'running',
  Completed = 'completed',
}
```

---

## Database Standards

### Schema Design

```sql
-- ✅ GOOD: Clear naming, proper constraints
CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    graph_definition JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT check_status CHECK (status IN ('draft', 'running', 'completed', 'failed'))
);

CREATE INDEX idx_workflows_user_id ON workflows(user_id);
CREATE INDEX idx_workflows_status ON workflows(status);
CREATE INDEX idx_workflows_created_at ON workflows(created_at DESC);

-- ❌ BAD: Vague names, no constraints
CREATE TABLE wf (
    id INT PRIMARY KEY,
    nm VARCHAR(100),
    st VARCHAR(10),
    usr INT
);
```

### Migrations

```python
# ✅ GOOD: Reversible, safe migration
def upgrade():
    # Add column with default
    op.add_column('workflows', sa.Column('priority', sa.Integer(), nullable=True))

    # Populate existing rows
    op.execute("UPDATE workflows SET priority = 0 WHERE priority IS NULL")

    # Make NOT NULL after populating
    op.alter_column('workflows', 'priority', nullable=False)

    # Add index
    op.create_index('idx_workflows_priority', 'workflows', ['priority'])

def downgrade():
    op.drop_index('idx_workflows_priority')
    op.drop_column('workflows', 'priority')

# ❌ BAD: Not reversible, breaks existing data
def upgrade():
    op.add_column('workflows', sa.Column('priority', sa.Integer(), nullable=False))
    # Fails if table has data!

def downgrade():
    pass  # No rollback!
```

### Queries

```python
# ✅ GOOD: Use ORM, avoid N+1
from sqlalchemy.orm import selectinload

async def get_workflows_with_nodes(user_id: str):
    result = await db.execute(
        select(WorkflowModel)
        .options(selectinload(WorkflowModel.nodes))  # Eager load
        .where(WorkflowModel.user_id == user_id)
        .order_by(WorkflowModel.created_at.desc())
        .limit(100)
    )
    return result.scalars().all()

# ❌ BAD: N+1 queries
async def get_workflows_with_nodes(user_id):
    workflows = await db.execute(
        select(WorkflowModel).where(WorkflowModel.user_id == user_id)
    )
    for workflow in workflows:
        nodes = await db.execute(
            select(NodeModel).where(NodeModel.workflow_id == workflow.id)
        )  # N+1!
```

---

## API Design

### REST Endpoints

```python
# ✅ GOOD: RESTful, clear naming, proper status codes
@app.post("/api/v1/workflows", status_code=201)
async def create_workflow(
    workflow: WorkflowCreate,
    user: User = Depends(get_current_user)
) -> WorkflowResponse:
    """Create a new workflow."""
    try:
        created = await workflow_service.create(workflow, user.id)
        return WorkflowResponse.from_orm(created)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    user: User = Depends(get_current_user)
) -> WorkflowResponse:
    """Get workflow by ID."""
    workflow = await workflow_service.get(workflow_id, user.id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return WorkflowResponse.from_orm(workflow)

# ❌ BAD: Inconsistent, wrong status codes
@app.post("/workflow/new")  # Should be POST /api/v1/workflows
def create(data):
    return workflow_service.create(data), 200  # Should be 201

@app.get("/getWorkflow")  # Should be GET /api/v1/workflows/{id}
def get(id):
    return workflow_service.get(id)
```

### Request/Response Models

```python
# ✅ GOOD: Separate models for request/response
from pydantic import BaseModel, Field

class WorkflowCreate(BaseModel):
    """Request model for creating workflow"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Workflow",
                "description": "Process customer data"
            }
        }

class WorkflowResponse(BaseModel):
    """Response model for workflow"""
    id: str
    name: str
    description: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ❌ BAD: Same model for request/response
class Workflow(BaseModel):
    id: str  # Should not be in create request
    name: str
    created_at: datetime  # Should not be in create request
```

---

## Testing Standards

### Unit Tests

```python
# ✅ GOOD: Clear, isolated, comprehensive
import pytest
from unittest.mock import Mock, patch

class TestWorkflowExecutor:
    @pytest.fixture
    def executor(self):
        return WorkflowExecutor(
            db=Mock(),
            tool_registry=Mock()
        )

    @pytest.fixture
    def sample_workflow(self):
        return Workflow(
            id="wf_123",
            nodes=[
                Node(id="n1", type="task", tool="print"),
                Node(id="n2", type="task", tool="log", dependencies=["n1"])
            ]
        )

    @pytest.mark.asyncio
    async def test_execute_linear_workflow(self, executor, sample_workflow):
        """Test execution of linear workflow."""
        result = await executor.execute(sample_workflow)

        assert result.status == "success"
        assert len(result.node_results) == 2
        assert result.node_results[0].node_id == "n1"
        assert result.node_results[1].node_id == "n2"

    @pytest.mark.asyncio
    async def test_execute_with_failing_node(self, executor, sample_workflow):
        """Test workflow execution when node fails."""
        executor.tool_registry.get.side_effect = [
            Mock(execute=Mock(side_effect=Exception("Tool failed"))),
            Mock()
        ]

        result = await executor.execute(sample_workflow)

        assert result.status == "failed"
        assert result.error is not None

# ❌ BAD: No fixtures, unclear, incomplete
def test_execute():
    executor = WorkflowExecutor()
    result = executor.execute({"id": "wf_123"})
    assert result  # What does this test?
```

### Test Coverage

- **Minimum:** 80% code coverage
- **Unit tests:** Business logic, utilities
- **Integration tests:** API endpoints, database operations
- **E2E tests:** Critical user workflows

---

## Security Guidelines

### Authentication

```python
# ✅ GOOD: Secure password handling
from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12
)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# ❌ BAD: Insecure hashing
import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()  # NEVER!
```

### Secrets Management

```python
# ✅ GOOD: Environment variables
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    jwt_secret: str
    api_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# ❌ BAD: Hardcoded secrets
DATABASE_URL = "postgresql://user:password@localhost/db"  # NEVER!
JWT_SECRET = "my-secret-key"  # NEVER!
```

### Input Validation

```python
# ✅ GOOD: Validate and sanitize
from pydantic import BaseModel, validator

class WorkflowCreate(BaseModel):
    name: str

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        if len(v) > 255:
            raise ValueError("Name too long")
        # Sanitize
        return v.strip()

# ❌ BAD: No validation
def create_workflow(name):
    db.execute(f"INSERT INTO workflows (name) VALUES ('{name}')")  # SQL injection!
```

---

## Git Workflow

### Branching Strategy

```
main                    # Production-ready code
  ├── develop          # Integration branch
  │   ├── feature/xxx  # New features
  │   ├── fix/xxx      # Bug fixes
  │   └── refactor/xxx # Code refactoring
  └── hotfix/xxx       # Critical production fixes
```

### Commit Messages

```bash
# ✅ GOOD: Conventional Commits
feat(workflow-engine): add retry logic for failed nodes
fix(api): handle null values in workflow response
docs(readme): update installation instructions
test(executor): add tests for parallel execution
refactor(planner): extract LLM prompt builder

# ❌ BAD: Vague messages
fixed bug
update code
changes
wip
```

### Pull Request

```markdown
## Description

Add retry logic for workflow node execution with exponential backoff.

## Changes

- Add RetryHandler class
- Implement exponential backoff strategy
- Add retry configuration to NodeConfig
- Update executor to use retry handler

## Testing

- Unit tests for RetryHandler
- Integration tests with failing nodes
- Tested with 3, 5, 10 retry attempts

## Checklist

- [x] Tests added/updated
- [x] Documentation updated
- [x] Code follows style guide
- [x] No breaking changes
```

---

## Documentation

### Code Comments

```python
# ✅ GOOD: Explain WHY, not WHAT
# Use binary search since workflows are sorted by created_at
# This reduces O(n) to O(log n) for large workflow lists
idx = binary_search(workflows, target_date)

# ❌ BAD: Obvious comments
# Increment i by 1
i += 1
```

### README Structure

````markdown
# Service Name

Brief description of what this service does.

## Features

- Feature 1
- Feature 2

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 15+

### Installation

```bash
pip install -r requirements.txt
```
````

### Configuration

```bash
cp .env.example .env
# Edit .env with your settings
```

### Running

```bash
uvicorn app.main:app --reload
```

## API Documentation

API docs available at: http://localhost:8000/docs

## Testing

```bash
pytest tests/
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md)

## License

See [LICENSE](../LICENSE)

````

---

## Pre-commit Checklist

Trước khi commit/push code, đảm bảo:

- [ ] Code passes linting (`black`, `ruff`, `eslint`)
- [ ] All tests pass (`pytest`, `jest`)
- [ ] Type checking passes (`mypy`, `tsc`)
- [ ] No sensitive data (API keys, passwords) in code
- [ ] Documentation updated nếu cần
- [ ] Commit message follows convention
- [ ] Code reviewed (self-review first)

---

## Tools & Automation

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
````

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov
      - run: black --check .
      - run: ruff check .
      - run: mypy .
```

---

## Enforcement

- **Code Review:** Mọi PR phải được review bởi ít nhất 1 team member
- **CI/CD:** PR không pass CI không được merge
- **Documentation:** PR thay đổi API/behavior phải update docs
- **Testing:** PR phải có tests cho new code

## Questions?

Nếu có thắc mắc về coding standards, hỏi trong team chat hoặc tạo issue.

---

**Last Updated:** February 4, 2026
**Maintained by:** Platform Team