
# AI Act Compliance Documentation

## Application Documentation Template

### Key Links
- [Project Documentation](#) `[PLACEHOLDER: Insert documentation access link]`
- [Additional Resources](#) `[PLACEHOLDER: Link to additional resources, if available]`

### General Information
**Application Name**: RAG, search and solve math  
**Description**: Utility functions and tools for building and querying a simple RAG pipeline. This application supports document retrieval, embedding creation, prompt formatting, and answers generation while enabling additional capabilities such as RAG retrieval and image generation.  
**Purpose**: Solving mathematical problems, retrieving contextual documents, and generating content based on queries.  

### Risk Classification
**Risk Level**: `[PLACEHOLDER: Determine risk level according to AI Act requirements (e.g., low, medium, high risk)]`  
**Reasoning**: `[PLACEHOLDER: Provide reasoning behind the risk classification]`  

### Application Functionality
#### Utility Functions
- **Settings**: Configuration setup for RAG utilities, including chunk size, retrieval methods, and embedding details.
- **build_faiss_vectorstore**: Creates and persists FAISS index from document chunks.
- **format_docs_for_prompt**: Prepares a prompt context string with source citations.
- **get_contexts_for_question**: Retrieves the top-k document chunks relevant to a query.
- **get_embeddings**: Initializes an Azure OpenAI embeddings client.
- **split_documents**: Splits documents into manageable chunks for better retrieval.
- **rag_answer**: Executes a RAG pipeline for answering a user query.

#### Tools
- **RagTool**: Performs RAG retrieval given a question and the number of documents to return.
- **ImageGenerationTool**: Generates images using Azure OpenAI's DALL-E API.

### Models and Datasets
#### Models
[PLACEHOLDER: Fill in details on the models, such as the Azure OpenAI embeddings, the DALL-E model for image generation, etc.]

#### Datasets
[PLACEHOLDER: Provide details on the datasets used for retrieval, training, or testing, if applicable.]

### Deployment
#### Infrastructure and Environment Details
**Environment Variables**:
- `AZURE_API_BASE`
- `AZURE_API_KEY`
- `AZURE_API_VERSION`
- `MODEL`

**FAISS Index Directory**: `faiss_index_example`  
**Deployment Methodology**: `[PLACEHOLDER: Describe deployment methodologies, e.g., cloud-based infrastructure, CI/CD pipelines used.]`

#### Integration with External Systems
- **Integrated APIs**: Azure OpenAI API for embeddings and DALL-E image generation  
- **Interaction**: External services facilitate retrieval, chunk indexing, and answer generation. `[PLACEHOLDER: Add detailed API interaction data.]`

### Deployment Plan
[PLACEHOLDER: Outline the deployment process, timeline, and stages.]

### Lifecycle Management
#### Risk Management System
[PLACEHOLDER: Describe the risk management system applied during the lifecycle of the application, including monitoring and mitigation strategies.]

### Testing and Validation (Accuracy, Robustness, Cybersecurity)
#### Accuracy Throughout the Lifecycle
[PLACEHOLDER: Explain how accuracy is validated and maintained during operation.]

#### Robustness
- **Features**: Ability to handle diverse query formats, overlapping document contexts, and embedding variability.  
- **Testing**: `[PLACEHOLDER: Provide details on robustness testing performed and results.]`

#### Cybersecurity
- **Environment Security**: Utilizes API keys and environment variables for authentication.  
- `[PLACEHOLDER: Add details on cybersecurity best practices and protocols followed.]`

### Human Oversight
[PLACEHOLDER: Mention mechanisms for human oversight, such as manual validation checks or user feedback processes.]

### Incident Management
#### Troubleshooting AI Application Deployment
[PLACEHOLDER: Detailed steps for identifying and resolving application issues.]

#### EU Declaration of Conformity
[PLACEHOLDER: Detail conformity processes implemented according to EU AI regulations.]

#### Standards Applied
[PLACEHOLDER: Mention applicable standards (e.g., ISO/IEC standards, GDPR compliance).]

### Documentation Metadata
**Template Version**: `[PLACEHOLDER: Specify version of the template used for documentation.]`  
**Documentation Authors**: `[PLACEHOLDER: Include author details.]`

---
**End of Document**
```

This draft organizes the available project information and introduces placeholders for sections requiring additional input.