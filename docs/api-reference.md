# MCP Tools

Used by AI assistants (OpenCode, Claude Desktop, Cursor, Windsurf, Cline) when configured as an MCP server.

### search_document_tool(question, document?)

Search the knowledge base and return formatted text fragments.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | Natural language query |
| `document` | string | No | Filter to specific document name |

**Returns:** Formatted string with numbered text blocks and Polish source citations (`Źródło: document, chapter, str. X-Y`).

### search_document_raw(question, document?)

Search and return structured JSON with relevance scores.

**Parameters:** Same as `search_document_tool`.

**Returns:** JSON string with array of objects containing `text`, `book`, `chapter`, `start_page`, `end_page`, and `score`.

### search_document_trace(question, document?)

Search documents with full pipeline trace showing retrieval and reranking details.

**Parameters:** Same as `search_document_tool`.

**Returns:** Detailed trace showing which text chunks were retrieved by the bi-encoder (with cosine scores), and how the cross-encoder reranker reordered them.

### list_documents_tool()

List all indexed document collections.

**Returns:** Formatted string listing available document names.

### get_collection_info(collection_name)

Get details about a specific indexed collection.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `collection_name` | string | Yes | Name of the collection |

**Returns:** Number of chunks, which documents it contains, and what chapters/sections are indexed.

### delete_document(collection_name)

Delete a document/collection from the knowledge base.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `collection_name` | string | Yes | Name of the collection to delete |

**Returns:** Confirmation message.

### ingest_document(file_path, reindex?)

Ingest any supported document format into the knowledge base.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | Yes | Absolute path to document |
| `reindex` | bool | No | Delete existing collection first (default: false) |

**Returns:** Status message with document name and chunk count.

### ingest_folder(directory, reindex?)

Ingest all supported documents from a directory.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `directory` | string | Yes | Absolute path to directory |
| `reindex` | bool | No | Re-index all documents (default: false) |

**Returns:** Summary with per-file results (indexed/skipped/error) and totals.
