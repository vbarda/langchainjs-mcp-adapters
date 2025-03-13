/**
 * Multiple MCP Servers Example - Firecrawl with Math Server
 *
 * This example demonstrates using multiple MCP servers from a single configuration file.
 * It includes both the Firecrawl server for web scraping and the Math server for calculations.
 */

import { ChatOpenAI } from '@langchain/openai';
import { StateGraph, END, START, MessagesAnnotation } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { HumanMessage, AIMessage, BaseMessage } from '@langchain/core/messages';
import { StructuredToolInterface } from '@langchain/core/tools';
import { z } from 'zod';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import logger from '../src/logger.js';

// MCP client imports
import { MultiServerMCPClient } from '../src/index.js';

// Load environment variables from .env file
dotenv.config();

// Path for our multiple servers config file
const multipleServersConfigPath = path.join(
  process.cwd(),
  'examples',
  'multiple_servers_config.json'
);

/**
 * Create a configuration file for multiple MCP servers
 */
function createMultipleServersConfigFile() {
  const configContent = {
    servers: {
      // Firecrawl server configuration
      firecrawl: {
        transport: 'stdio',
        command: 'npx',
        args: ['-y', 'firecrawl-mcp'],
        env: {
          FIRECRAWL_API_KEY: process.env.FIRECRAWL_API_KEY || '',
          FIRECRAWL_RETRY_MAX_ATTEMPTS: '3',
        },
      },
      // Math server configuration
      math: {
        transport: 'stdio',
        command: 'python',
        args: [path.join(process.cwd(), 'examples', 'math_server.py')],
      },
    },
  };

  fs.writeFileSync(multipleServersConfigPath, JSON.stringify(configContent, null, 2));
  logger.info(`Created multiple servers configuration file at ${multipleServersConfigPath}`);
}

/**
 * Example demonstrating how to use multiple MCP servers with LangGraph agent flows
 * This example creates and loads a configuration file with multiple servers
 */
async function runExample() {
  let client: MultiServerMCPClient | null = null;

  try {
    // Create the multiple servers configuration file
    createMultipleServersConfigFile();

    logger.info('Initializing MCP client from multiple servers configuration file...');

    // Create a client from the configuration file
    client = MultiServerMCPClient.fromConfigFile(multipleServersConfigPath);

    // Initialize connections to all servers in the configuration
    await client.initializeConnections();
    logger.info('Connected to servers from multiple servers configuration');

    // Get all tools from all servers
    const mcpTools = client.getTools() as StructuredToolInterface<z.ZodObject<any>>[];

    if (mcpTools.length === 0) {
      throw new Error('No tools found');
    }

    logger.info(
      `Loaded ${mcpTools.length} MCP tools: ${mcpTools.map(tool => tool.name).join(', ')}`
    );

    // Create an OpenAI model and bind the tools
    const model = new ChatOpenAI({
      modelName: process.env.OPENAI_MODEL_NAME || 'gpt-4o',
      temperature: 0,
    }).bindTools(mcpTools);

    // Create a tool node for the LangGraph
    const toolNode = new ToolNode(mcpTools);

    // ================================================
    // Create a LangGraph agent flow
    // ================================================
    console.log('\n=== CREATING LANGGRAPH AGENT FLOW ===');

    // Define the function that calls the model
    const llmNode = async (state: typeof MessagesAnnotation.State) => {
      console.log('Calling LLM with messages:', state.messages.length);
      const response = await model.invoke(state.messages);
      return { messages: [response] };
    };

    // Create a new graph with MessagesAnnotation
    const workflow = new StateGraph(MessagesAnnotation);

    // Add the nodes to the graph
    workflow.addNode('llm', llmNode);
    workflow.addNode('tools', toolNode);

    // Add edges - these define how nodes are connected
    workflow.addEdge(START as any, 'llm' as any);
    workflow.addEdge('tools' as any, 'llm' as any);

    // Conditional routing to end or continue the tool loop
    workflow.addConditionalEdges('llm' as any, state => {
      const lastMessage = state.messages[state.messages.length - 1];
      const aiMessage = lastMessage as AIMessage;

      if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
        console.log('Tool calls detected, routing to tools node');
        return 'tools' as any;
      }

      console.log('No tool calls, ending the workflow');
      return END as any;
    });

    // Compile the graph
    const app = workflow.compile();

    // Define queries that will use both servers
    const queries = [
      'What is 25 multiplied by 18?',
      'Scrape the content from https://example.com and count how many paragraphs are there',
      'If I have 42 items and each costs $7.50, what is the total cost?',
    ];

    // Test the LangGraph agent with the queries
    console.log('\n=== RUNNING LANGGRAPH AGENT ===');

    for (const query of queries) {
      console.log(`\nQuery: ${query}`);

      // Run the LangGraph agent with the query
      const result = await app.invoke({
        messages: [new HumanMessage(query)],
      });

      // Display the final response
      const finalMessage = result.messages[result.messages.length - 1];
      console.log(`\nResult: ${finalMessage.content}`);
    }
  } catch (error) {
    console.error('Error:', error);
    process.exit(1); // Exit with error code
  } finally {
    // Close all client connections
    if (client) {
      await client.close();
      console.log('\nClosed all connections');
    }

    // Clean up our config file
    if (fs.existsSync(multipleServersConfigPath)) {
      fs.unlinkSync(multipleServersConfigPath);
      logger.info(`Cleaned up multiple servers configuration file at ${multipleServersConfigPath}`);
    }

    // Exit process after a short delay to allow for cleanup
    setTimeout(() => {
      console.log('Example completed, exiting process.');
      process.exit(0);
    }, 500);
  }
}

// Run the example
runExample();
