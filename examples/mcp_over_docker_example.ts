/**
 * Filesystem MCP Server with LangGraph Example
 *
 * This example demonstrates how to use the Filesystem MCP server with LangGraph
 * to create a structured workflow for complex file operations.
 *
 * The graph-based approach allows:
 * 1. Clear separation of responsibilities (reasoning vs execution)
 * 2. Conditional routing based on file operation types
 * 3. Structured handling of complex multi-file operations
 */

import { logger, MultiServerMCPClient } from '../src';
import { runExample as runFileSystemExample } from './filesystem_langgraph_example';

async function runExample() {
  const client = new MultiServerMCPClient({
    filesystem: {
      transport: 'stdio',
      command: 'docker',
      args: [
        'run',
        '-i',
        '--rm',
        '-v',
        'mcp-filesystem-data:/projects',
        'mcp/filesystem',
        '/projects',
      ],
    },
  });

  await runFileSystemExample(client);
}

const isMainModule = import.meta.url === `file://${process.argv[1]}`;
if (isMainModule) {
  runExample().catch(error => logger.error('Setup error:', error));
}
