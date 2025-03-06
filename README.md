# Code Review Platform Development Outline

## Project Overview
This project aims to create a web-based code review platform that integrates with GitLab and GitHub, leveraging AI-powered code analysis. The backend will be built with FastAPI, and the frontend (to be added later) will provide a user-friendly interface. The platform will support reviewing pull requests (GitHub) and merge requests (GitLab), offering detailed line-by-line feedback and summaries powered by AI (e.g., OpenAI or Ollama).

### Key Features
- Integration with GitLab and GitHub for fetching and reviewing code changes.
- AI-powered code reviews with detailed comments and summaries.
- RESTFul API backend using FastAPI.
- Support for both CLI and web-based usage.
- Extensible architecture for adding more AI providers or repository platforms.
- User authentication and project management (future enhancement).

### Tech Stack
- **Backend**: Python, FastAPI, GitHub/GitLab APIs, OpenAI/Ollama APIs.
- **Frontend** (Future): React, Axios for API calls, Tailwind CSS or similar.

## Development Roadmap

### Phase 1: Backend Foundation (Current Code Base)
**Goal**: Establish a working backend with core functionality based on the provided code.
- [x] Set up FastAPI application with lifespan management for background tasks.
- [x] Implement configuration loading from `config.json` and `.env`.
- [x] Create abstract `BaseReview` class with GitHub (`GitHubReview`) and GitLab (`GitLabReview`) implementations.
- [x] Add AI client abstraction (`AIClient`) supporting OpenAI and Ollama.
- [x] Implement code review logic with detailed comments and summaries.
- [x] Add `/review` API endpoint for reviewing code changes.
- [x] Set up CLI interface for manual testing.
- [x] Implement background task to periodically check for new GitHub pull requests.

**Next Steps**:
- Test the current code with real GitHub and GitLab repositories.
- Fix any bugs in the review parsing or comment submission logic.
- Add logging to track errors and review progress.

### Phase 2: Backend Enhancements
**Goal**: Improve robustness, scalability, and usability of the backend.
- [ ] **Authentication**: Add API key or OAuth-based authentication for secure access.
  - Use `fastapi.security` for API key management.
  - Integrate GitHub/GitLab OAuth for user login.
- [ ] **Database Integration**: Store review history and user settings.
  - Use SQLite or PostgreSQL with SQLAlchemy.
  - Create tables for users, reviews, and repository configs.
- [ ] **Queue System**: Handle large review requests asynchronously.
  - Integrate `Celery` with Redis or RabbitMQ for task queuing.
  - Move AI review processing to background tasks.
- [ ] **Error Handling**: Improve exception handling and user feedback.
  - Add detailed error responses in API.
  - Log errors to a file or external service (e.g., Sentry).
- [ ] **Rate Limiting**: Prevent abuse of the API.
  - Use `slowapi` or similar to enforce limits.
- [ ] **Configuration API**: Allow runtime updates to AI provider/model via API.

**Deliverables**:
- Secure and scalable backend API.
- Database schema and initial data population.
- Documentation for new endpoints (e.g., `/auth`, `/config`).

### Phase 3: Frontend Development
**Goal**: Build a basic web interface for interacting with the backend.
- [ ] **Setup**: Choose a frontend framework (e.g., React, Vue.js) and set up the project.
- [ ] **Authentication UI**: Login page with GitHub/GitLab OAuth.
- [ ] **Dashboard**: Display list of connected repositories and recent reviews.
- [ ] **Review Page**: Show pull/merge request details with AI comments and summary.
- [ ] **Settings Page**: Configure AI provider, model, and repository settings.
- [ ] **API Integration**: Use Axios or Fetch to connect to FastAPI endpoints.
- [ ] **Styling**: Apply a CSS framework (e.g., Tailwind CSS) for a clean UI.

**Deliverables**:
- Working frontend prototype.
- Basic navigation and styling.
- Integration with backend API endpoints.

### Phase 4: Advanced Features
**Goal**: Add features to enhance user experience and platform capabilities.
- [ ] **Multi-Repository Support**: Allow users to connect multiple GitHub/GitLab repositories.
- [ ] **Custom Prompts**: Enable users to define custom AI review prompts via UI.
- [ ] **Review History**: Show past reviews with filtering and search.
- [ ] **Notifications**: Email or webhook notifications for new reviews.
- [ ] **Team Features**: Support for multiple users per project with roles (e.g., reviewer, admin).
- [ ] **Performance Optimization**: Cache AI responses and optimize diff parsing.

**Deliverables**:
- Enhanced platform functionality.
- User documentation for advanced features.

### Phase 5: Deployment and Testing
**Goal**: Prepare the platform for production use.
- [ ] **Testing**: Write unit and integration tests for backend (e.g., `pytest`).
- [ ] **CI/CD**: Set up GitHub Actions or GitLab CI for automated testing and deployment.
- [ ] **Deployment**: Deploy backend to a cloud provider (e.g., Heroku, AWS, DigitalOcean).
- [ ] **Monitoring**: Add logging and monitoring (e.g., Prometheus, Grafana).
- [ ] **Documentation**: Write full API docs with OpenAPI/Swagger and user guide.

**Deliverables**:
- Production-ready application.
- Comprehensive test suite.
- Deployment scripts and documentation.


## License
[Apache 2.0 License](LICENSE)
