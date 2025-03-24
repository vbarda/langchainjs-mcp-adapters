import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import type { StructuredToolInterface } from "@langchain/core/tools";
import debug from "debug";
import { loadMcpTools } from "./tools.js";

// Read package name from package.json
let debugLog: debug.Debugger;
function getDebugLog() {
  if (!debugLog) {
    debugLog = debug("@langchain/mcp-adapters:client");
  }
  return debugLog;
}

/**
 * Configuration for stdio transport connection
 */
export interface StdioConnection {
  transport: "stdio";
  command: string;
  args: string[];
  env?: Record<string, string>;
  encoding?: string;
  encodingErrorHandler?: "strict" | "ignore" | "replace";
  /**
   * Additional restart settings
   */
  restart?: {
    /**
     * Whether to automatically restart the process if it exits
     */
    enabled?: boolean;
    /**
     * Maximum number of restart attempts
     */
    maxAttempts?: number;
    /**
     * Delay in milliseconds between restart attempts
     */
    delayMs?: number;
  };
}

/**
 * Configuration for SSE transport connection
 */
export interface SSEConnection {
  transport: "sse";
  url: string;
  headers?: Record<string, string>;
  useNodeEventSource?: boolean;
  /**
   * Additional reconnection settings
   */
  reconnect?: {
    /**
     * Whether to automatically reconnect if the connection is lost
     */
    enabled?: boolean;
    /**
     * Maximum number of reconnection attempts
     */
    maxAttempts?: number;
    /**
     * Delay in milliseconds between reconnection attempts
     */
    delayMs?: number;
  };
}

/**
 * Check if a configuration is for a stdio connection
 */
export function isStdioConnection(config: unknown): config is StdioConnection {
  // When transport is missing, default to stdio if it has command and args
  // OR when transport is explicitly set to 'stdio'
  return (
    typeof config === "object" &&
    config !== null &&
    (!("transport" in config) || config.transport === "stdio") &&
    "command" in config &&
    (!("args" in config) || Array.isArray(config.args))
  );
}

/**
 * Check if a configuration is for an SSE connection
 */
export function isSSEConnection(config: unknown): config is SSEConnection {
  // Only consider it an SSE connection if transport is explicitly set to 'sse'
  return (
    typeof config === "object" &&
    config !== null &&
    "transport" in config &&
    config.transport === "sse" &&
    "url" in config &&
    typeof config.url === "string"
  );
}

/**
 * Union type for all transport connection types
 */
export type Connection = StdioConnection | SSEConnection;

/**
 * MCP configuration file format
 */
export interface MCPConfig {
  servers: Record<string, Connection>;
}

/**
 * Error class for MCP client operations
 */
export class MCPClientError extends Error {
  constructor(message: string, public readonly serverName?: string) {
    super(message);
    this.name = "MCPClientError";
  }
}

/**
 * Client for connecting to multiple MCP servers and loading LangChain-compatible tools.
 */
export class MultiServerMCPClient {
  private _clients: Record<string, Client> = {};

  private _serverNameToTools: Record<string, StructuredToolInterface[]> = {};

  private _connections?: Record<string, Connection>;

  private _cleanupFunctions: Array<() => Promise<void>> = [];

  private _transportInstances: Record<
    string,
    StdioClientTransport | SSEClientTransport
  > = {};

  /**
   * Create a new MultiServerMCPClient.
   *
   * @param connections - Optional connections to initialize
   */
  constructor(connections: Record<string, Connection>) {
    if (Object.keys(connections).length === 0) {
      throw new MCPClientError("No connections provided");
    }

    this._connections = MultiServerMCPClient._processConnections(connections);
  }

  /**
   * Proactively initialize connections to all servers. This will be called automatically when
   * methods requiring an active connection (like {@link getTools} or {@link getClient}) are called,
   * but you can call it directly to ensure all connections are established before using the tools.
   *
   * @returns A map of server names to arrays of tools
   * @throws {MCPClientError} If initialization fails
   */
  async initializeConnections(): Promise<
    Record<string, StructuredToolInterface[]>
  > {
    if (!this._connections || Object.keys(this._connections).length === 0) {
      throw new MCPClientError("No connections to initialize");
    }

    const connectionsToInit: [string, Connection][] = Array.from(
      Object.entries(this._connections).filter(
        ([serverName]) => this._clients[serverName] === undefined
      )
    );

    for (const [serverName, connection] of connectionsToInit) {
      getDebugLog()(
        `INFO: Initializing connection to server "${serverName}"...`
      );

      if (connection.transport === "stdio") {
        await this._initializeStdioConnection(serverName, connection);
      } else if (connection.transport === "sse") {
        await this._initializeSSEConnection(serverName, connection);
      } else {
        // This should never happen due to the validation in the constructor
        throw new MCPClientError(
          `Unsupported transport type for server "${serverName}"`,
          serverName
        );
      }
    }

    return this._serverNameToTools;
  }

  /**
   * Get tools from specified servers as a flattened array.
   *
   * @param servers - Optional array of server names to filter tools by.
   *                 If not provided, returns tools from all servers.
   * @returns A flattened array of tools from the specified servers (or all servers)
   */
  async getTools(...servers: string[]): Promise<StructuredToolInterface[]> {
    await this.initializeConnections();
    if (servers.length === 0) {
      return this._getAllToolsAsFlatArray();
    }
    return this._getToolsFromServers(servers);
  }

  /**
   * Get a the MCP client for a specific server. Useful for fetching prompts or resources from that server.
   *
   * @param serverName - The name of the server
   * @returns The client for the server, or undefined if the server is not connected
   */
  async getClient(serverName: string): Promise<Client | undefined> {
    await this.initializeConnections();
    return this._clients[serverName];
  }

  /**
   * Close all connections.
   */
  async close(): Promise<void> {
    getDebugLog()(`INFO: Closing all MCP connections...`);

    for (const cleanup of this._cleanupFunctions) {
      try {
        await cleanup();
      } catch (error) {
        getDebugLog()(`ERROR: Error during cleanup: ${error}`);
      }
    }

    this._cleanupFunctions = [];
    this._clients = {};
    this._serverNameToTools = {};
    this._transportInstances = {};

    getDebugLog()(`INFO: All MCP connections closed`);
  }

  /**
   * Process connection configurations
   *
   * @param connections - Raw connection configurations
   * @returns Processed connection configurations
   */
  private static _processConnections(
    connections: Record<string, Partial<Connection>>
  ): Record<string, Connection> {
    const processedConnections: Record<string, Connection> = {};

    for (const [serverName, config] of Object.entries(connections)) {
      if (typeof config !== "object" || config === null) {
        getDebugLog()(
          `WARN: Invalid configuration for server "${serverName}". Skipping.`
        );
        continue;
      }

      // Determine the connection type and process accordingly
      if (isStdioConnection(config)) {
        processedConnections[serverName] =
          MultiServerMCPClient._processStdioConfig(serverName, config);
      } else if (isSSEConnection(config)) {
        processedConnections[serverName] =
          MultiServerMCPClient._processSSEConfig(serverName, config);
      } else {
        throw new MCPClientError(
          `Server "${serverName}" has invalid or unsupported configuration. Skipping.`
        );
      }
    }

    return processedConnections;
  }

  /**
   * Process stdio connection configuration
   */
  private static _processStdioConfig(
    serverName: string,
    config: Partial<StdioConnection>
  ): StdioConnection {
    if (!config.command || typeof config.command !== "string") {
      throw new MCPClientError(
        `Missing or invalid command for server "${serverName}"`
      );
    }

    if (config.args !== undefined && !Array.isArray(config.args)) {
      throw new MCPClientError(
        `Invalid args for server "${serverName} - must be an array of strings`
      );
    }

    if (
      config.args !== undefined &&
      !config.args.every((arg) => typeof arg === "string")
    ) {
      throw new MCPClientError(
        `Invalid args for server "${serverName} - must be an array of strings`
      );
    }

    // Always set transport to 'stdio' regardless of whether it was in the original config
    const stdioConfig: StdioConnection = {
      transport: "stdio",
      command: config.command,
      args: config.args ?? [],
    };

    if (config.env && typeof config.env !== "object") {
      throw new MCPClientError(
        `Invalid env for server "${serverName} - must be an object of key-value pairs`
      );
    }

    if (
      config.env &&
      typeof config.env === "object" &&
      Array.isArray(config.env)
    ) {
      throw new MCPClientError(
        `Invalid env for server "${serverName} - must be an object of key-value pairs`
      );
    }

    if (
      config.env &&
      typeof config.env === "object" &&
      !Object.values(config.env).every((value) => typeof value === "string")
    ) {
      throw new MCPClientError(
        `Invalid env for server "${serverName} - must be an object of key-value pairs with string values`
      );
    }

    // Add optional properties if they exist
    if (config.env && typeof config.env === "object") {
      stdioConfig.env = config.env;
    }

    if (config.encoding !== undefined && typeof config.encoding !== "string") {
      throw new MCPClientError(
        `Invalid encoding for server "${serverName} - must be a string`
      );
    }

    if (typeof config.encoding === "string") {
      stdioConfig.encoding = config.encoding;
    }

    if (
      config.encodingErrorHandler !== undefined &&
      !["strict", "ignore", "replace"].includes(config.encodingErrorHandler)
    ) {
      throw new MCPClientError(
        `Invalid encodingErrorHandler for server "${serverName} - must be one of: strict, ignore, replace`
      );
    }

    if (
      ["strict", "ignore", "replace"].includes(
        config.encodingErrorHandler ?? ""
      )
    ) {
      stdioConfig.encodingErrorHandler = config.encodingErrorHandler as
        | "strict"
        | "ignore"
        | "replace";
    }

    // Add restart configuration if present
    if (config.restart && typeof config.restart !== "object") {
      throw new MCPClientError(
        `Invalid restart for server "${serverName} - must be an object`
      );
    }

    if (config.restart && typeof config.restart === "object") {
      if (
        config.restart.enabled !== undefined &&
        typeof config.restart.enabled !== "boolean"
      ) {
        throw new MCPClientError(
          `Invalid restart.enabled for server "${serverName} - must be a boolean`
        );
      }

      stdioConfig.restart = {
        enabled: Boolean(config.restart.enabled),
      };

      if (
        config.restart.maxAttempts !== undefined &&
        typeof config.restart.maxAttempts !== "number"
      ) {
        throw new MCPClientError(
          `Invalid restart.maxAttempts for server "${serverName} - must be a number`
        );
      }

      if (typeof config.restart.maxAttempts === "number") {
        stdioConfig.restart.maxAttempts = config.restart.maxAttempts;
      }

      if (
        config.restart.delayMs !== undefined &&
        typeof config.restart.delayMs !== "number"
      ) {
        throw new MCPClientError(
          `Invalid restart.delayMs for server "${serverName} - must be a number`
        );
      }

      if (typeof config.restart.delayMs === "number") {
        stdioConfig.restart.delayMs = config.restart.delayMs;
      }
    }

    return stdioConfig;
  }

  /**
   * Process SSE connection configuration
   */
  private static _processSSEConfig(
    serverName: string,
    config: SSEConnection
  ): SSEConnection {
    if (!config.url || typeof config.url !== "string") {
      throw new MCPClientError(
        `Missing or invalid url for server "${serverName}"`
      );
    }

    try {
      const url = new URL(config.url);
      if (!url.protocol.startsWith("http")) {
        throw new MCPClientError(
          `Invalid url for server "${serverName} - must be a valid HTTP or HTTPS URL`
        );
      }
    } catch {
      throw new MCPClientError(
        `Invalid url for server "${serverName} - must be a valid URL`
      );
    }

    if (!config.transport || config.transport !== "sse") {
      throw new MCPClientError(
        `Invalid transport for server "${serverName} - must be 'sse'`
      );
    }

    const sseConfig: SSEConnection = {
      transport: "sse",
      url: config.url,
    };

    if (config.headers && typeof config.headers !== "object") {
      throw new MCPClientError(
        `Invalid headers for server "${serverName} - must be an object`
      );
    }

    if (
      config.headers &&
      typeof config.headers === "object" &&
      Array.isArray(config.headers)
    ) {
      throw new MCPClientError(
        `Invalid headers for server "${serverName} - must be an object of key-value pairs`
      );
    }

    if (
      config.headers &&
      typeof config.headers === "object" &&
      !Object.values(config.headers).every((value) => typeof value === "string")
    ) {
      throw new MCPClientError(
        `Invalid headers for server "${serverName} - must be an object of key-value pairs with string values`
      );
    }

    // Add optional headers if they exist
    if (config.headers && typeof config.headers === "object") {
      sseConfig.headers = config.headers;
    }

    if (
      config.useNodeEventSource !== undefined &&
      typeof config.useNodeEventSource !== "boolean"
    ) {
      throw new MCPClientError(
        `Invalid useNodeEventSource for server "${serverName} - must be a boolean`
      );
    }

    // Add optional useNodeEventSource flag if it exists
    if (typeof config.useNodeEventSource === "boolean") {
      sseConfig.useNodeEventSource = config.useNodeEventSource;
    }

    if (config.reconnect && typeof config.reconnect !== "object") {
      throw new MCPClientError(
        `Invalid reconnect for server "${serverName} - must be an object`
      );
    }

    // Add reconnection configuration if present
    if (config.reconnect && typeof config.reconnect === "object") {
      if (
        config.reconnect.enabled !== undefined &&
        typeof config.reconnect.enabled !== "boolean"
      ) {
        throw new MCPClientError(
          `Invalid reconnect.enabled for server "${serverName} - must be a boolean`
        );
      }

      sseConfig.reconnect = {
        enabled: Boolean(config.reconnect.enabled),
      };

      if (
        config.reconnect.maxAttempts !== undefined &&
        typeof config.reconnect.maxAttempts !== "number"
      ) {
        throw new MCPClientError(
          `Invalid reconnect.maxAttempts for server "${serverName} - must be a number`
        );
      }

      if (typeof config.reconnect.maxAttempts === "number") {
        sseConfig.reconnect.maxAttempts = config.reconnect.maxAttempts;
      }

      if (
        config.reconnect.delayMs !== undefined &&
        typeof config.reconnect.delayMs !== "number"
      ) {
        throw new MCPClientError(
          `Invalid reconnect.delayMs for server "${serverName} - must be a number`
        );
      }

      if (typeof config.reconnect.delayMs === "number") {
        sseConfig.reconnect.delayMs = config.reconnect.delayMs;
      }
    }

    return sseConfig;
  }

  /**
   * Initialize a stdio connection
   */
  private async _initializeStdioConnection(
    serverName: string,
    connection: StdioConnection
  ): Promise<void> {
    const { command, args, env, restart } = connection;

    getDebugLog()(
      `DEBUG: Creating stdio transport for server "${serverName}" with command: ${command} ${args.join(
        " "
      )}`
    );

    const transport = new StdioClientTransport({
      command,
      args,
      env,
    });

    this._transportInstances[serverName] = transport;

    const client = new Client({
      name: "langchain-mcp-adapter",
      version: "0.1.0",
    });

    try {
      await client.connect(transport);

      // Set up auto-restart if configured
      if (restart?.enabled) {
        this._setupStdioRestart(serverName, transport, connection, restart);
      }
    } catch (error) {
      throw new MCPClientError(
        `Failed to connect to stdio server "${serverName}": ${error}`,
        serverName
      );
    }

    this._clients[serverName] = client;

    const cleanup = async () => {
      getDebugLog()(
        `DEBUG: Closing stdio transport for server "${serverName}"`
      );
      await transport.close();
    };

    this._cleanupFunctions.push(cleanup);

    // Load tools for this server
    await this._loadToolsForServer(serverName, client);
  }

  /**
   * Set up stdio restart handling
   */
  private _setupStdioRestart(
    serverName: string,
    transport: StdioClientTransport,
    connection: StdioConnection,
    restart: NonNullable<StdioConnection["restart"]>
  ): void {
    const originalOnClose = transport.onclose;
    // eslint-disable-next-line no-param-reassign, @typescript-eslint/no-misused-promises
    transport.onclose = async () => {
      if (originalOnClose) {
        await originalOnClose();
      }

      // Only attempt restart if we haven't cleaned up
      if (this._clients[serverName]) {
        getDebugLog()(
          `INFO: Process for server "${serverName}" exited, attempting to restart...`
        );
        await this._attemptReconnect(
          serverName,
          connection,
          restart.maxAttempts,
          restart.delayMs
        );
      }
    };
  }

  /**
   * Initialize an SSE connection
   */
  private async _initializeSSEConnection(
    serverName: string,
    connection: SSEConnection
  ): Promise<void> {
    const { url, headers, useNodeEventSource, reconnect } = connection;

    getDebugLog()(
      `DEBUG: Creating SSE transport for server "${serverName}" with URL: ${url}`
    );

    try {
      const transport = await this._createSSETransport(
        serverName,
        url,
        headers,
        useNodeEventSource
      );
      this._transportInstances[serverName] = transport;

      const client = new Client({
        name: "langchain-mcp-adapter",
        version: "0.1.0",
      });

      try {
        await client.connect(transport);

        // Set up auto-reconnect if configured
        if (reconnect?.enabled) {
          this._setupSSEReconnect(serverName, transport, connection, reconnect);
        }
      } catch (error) {
        throw new MCPClientError(
          `Failed to connect to SSE server "${serverName}": ${error}`,
          serverName
        );
      }

      this._clients[serverName] = client;

      const cleanup = async () => {
        getDebugLog()(
          `DEBUG: Closing SSE transport for server "${serverName}"`
        );
        await transport.close();
      };

      this._cleanupFunctions.push(cleanup);

      // Load tools for this server
      await this._loadToolsForServer(serverName, client);
    } catch (error) {
      throw new MCPClientError(
        `Failed to create SSE transport for server "${serverName}": ${error}`,
        serverName
      );
    }
  }

  /**
   * Create an SSE transport with appropriate EventSource implementation
   */
  private async _createSSETransport(
    serverName: string,
    url: string,
    headers?: Record<string, string>,
    useNodeEventSource?: boolean
  ): Promise<SSEClientTransport> {
    if (!headers) {
      // Simple case - no headers, use default transport
      return new SSEClientTransport(new URL(url));
    }

    getDebugLog()(
      `DEBUG: Using custom headers for SSE transport to server "${serverName}"`
    );

    // If useNodeEventSource is true, try Node.js implementations
    if (useNodeEventSource) {
      return await this._createNodeEventSourceTransport(
        serverName,
        url,
        headers
      );
    }

    // For browser environments, use the basic requestInit approach
    getDebugLog()(
      `DEBUG: Using browser EventSource for server "${serverName}". Headers may not be applied correctly.`
    );
    getDebugLog()(
      `DEBUG: For better headers support in browsers, consider using a custom SSE implementation.`
    );

    return new SSEClientTransport(new URL(url), {
      requestInit: { headers },
    });
  }

  /**
   * Create an EventSource transport for Node.js environments
   */
  private async _createNodeEventSourceTransport(
    serverName: string,
    url: string,
    headers: Record<string, string>
  ): Promise<SSEClientTransport> {
    // First try to use extended-eventsource which has better headers support
    try {
      const ExtendedEventSourceModule = await import("extended-eventsource");
      const ExtendedEventSource = ExtendedEventSourceModule.EventSource;

      getDebugLog()(
        `DEBUG: Using Extended EventSource for server "${serverName}"`
      );
      getDebugLog()(
        `DEBUG: Setting headers for Extended EventSource: ${JSON.stringify(
          headers
        )}`
      );

      // Override the global EventSource with the extended implementation
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (globalThis as any).EventSource = ExtendedEventSource;

      // For Extended EventSource, create the SSE transport
      return new SSEClientTransport(new URL(url), {
        // Pass empty options for test compatibility
        eventSourceInit: {},
        requestInit: {},
      });
    } catch (extendedError) {
      // Fall back to standard eventsource if extended-eventsource is not available
      getDebugLog()(
        `DEBUG: Extended EventSource not available, falling back to standard EventSource: ${extendedError}`
      );

      try {
        // Dynamically import the eventsource package
        // eslint-disable-next-line import/no-extraneous-dependencies
        const EventSourceModule = await import("eventsource");
        const EventSource =
          "default" in EventSourceModule
            ? EventSourceModule.default
            : EventSourceModule.EventSource;

        getDebugLog()(
          `DEBUG: Using Node.js EventSource for server "${serverName}"`
        );
        getDebugLog()(
          `DEBUG: Setting headers for EventSource: ${JSON.stringify(headers)}`
        );

        // Override the global EventSource with the Node.js implementation
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (globalThis as any).EventSource = EventSource;

        // Create transport with headers correctly configured for Node.js EventSource
        return new SSEClientTransport(new URL(url), {
          // Pass the headers to both eventSourceInit and requestInit for compatibility
          requestInit: { headers },
        });
      } catch (nodeError) {
        getDebugLog()(
          `WARN: Failed to load EventSource packages for server "${serverName}". Headers may not be applied to SSE connection: ${nodeError}`
        );

        // Last resort fallback
        return new SSEClientTransport(new URL(url), {
          requestInit: { headers },
        });
      }
    }
  }

  /**
   * Set up SSE reconnect handling
   */
  private _setupSSEReconnect(
    serverName: string,
    transport: SSEClientTransport,
    connection: SSEConnection,
    reconnect: NonNullable<SSEConnection["reconnect"]>
  ): void {
    const originalOnClose = transport.onclose;
    // eslint-disable-next-line @typescript-eslint/no-misused-promises, no-param-reassign
    transport.onclose = async () => {
      if (originalOnClose) {
        await originalOnClose();
      }

      // Only attempt reconnect if we haven't cleaned up
      if (this._clients[serverName]) {
        getDebugLog()(
          `INFO: SSE connection for server "${serverName}" closed, attempting to reconnect...`
        );
        await this._attemptReconnect(
          serverName,
          connection,
          reconnect.maxAttempts,
          reconnect.delayMs
        );
      }
    };
  }

  /**
   * Load tools for a specific server
   */
  private async _loadToolsForServer(
    serverName: string,
    client: Client
  ): Promise<void> {
    try {
      getDebugLog()(`DEBUG: Loading tools for server "${serverName}"...`);
      const tools = await loadMcpTools(serverName, client);
      this._serverNameToTools[serverName] = tools;
      getDebugLog()(
        `INFO: Successfully loaded ${tools.length} tools from server "${serverName}"`
      );
    } catch (error) {
      throw new MCPClientError(
        `Failed to load tools from server "${serverName}": ${error}`
      );
    }
  }

  /**
   * Attempt to reconnect to a server after a connection failure.
   *
   * @param serverName - The name of the server to reconnect to
   * @param connection - The connection configuration
   * @param maxAttempts - Maximum number of reconnection attempts
   * @param delayMs - Delay in milliseconds between reconnection attempts
   * @private
   */
  private async _attemptReconnect(
    serverName: string,
    connection: Connection,
    maxAttempts = 3,
    delayMs = 1000
  ): Promise<void> {
    let connected = false;
    let attempts = 0;

    // Clean up previous connection resources
    this._cleanupServerResources(serverName);

    while (
      !connected &&
      (maxAttempts === undefined || attempts < maxAttempts)
    ) {
      attempts += 1;
      getDebugLog()(
        `INFO: Reconnection attempt ${attempts}${
          maxAttempts ? `/${maxAttempts}` : ""
        } for server "${serverName}"`
      );

      try {
        // Wait before attempting to reconnect
        if (delayMs) {
          await new Promise((resolve) => {
            setTimeout(resolve, delayMs);
          });
        }

        // Initialize just this connection based on its type
        if (connection.transport === "stdio") {
          await this._initializeStdioConnection(serverName, connection);
        } else if (connection.transport === "sse") {
          await this._initializeSSEConnection(serverName, connection);
        }

        // Check if connected
        if (this._clients[serverName]) {
          connected = true;
          getDebugLog()(
            `INFO: Successfully reconnected to server "${serverName}"`
          );
        }
      } catch (error) {
        getDebugLog()(
          `ERROR: Failed to reconnect to server "${serverName}" (attempt ${attempts}): ${error}`
        );
      }
    }

    if (!connected) {
      getDebugLog()(
        `ERROR: Failed to reconnect to server "${serverName}" after ${attempts} attempts`
      );
    }
  }

  /**
   * Clean up resources for a specific server
   */
  private _cleanupServerResources(serverName: string): void {
    delete this._clients[serverName];
    delete this._serverNameToTools[serverName];
    delete this._transportInstances[serverName];
  }

  /**
   * Get all tools from all servers as a flat array.
   *
   * @returns A flattened array of all tools
   */
  private _getAllToolsAsFlatArray(): StructuredToolInterface[] {
    const allTools: StructuredToolInterface[] = [];
    for (const tools of Object.values(this._serverNameToTools)) {
      allTools.push(...tools);
    }
    return allTools;
  }

  /**
   * Get tools from specific servers as a flat array.
   *
   * @param serverNames - Names of servers to get tools from
   * @returns A flattened array of tools from the specified servers
   */
  private _getToolsFromServers(
    serverNames: string[]
  ): StructuredToolInterface[] {
    const allTools: StructuredToolInterface[] = [];
    for (const serverName of serverNames) {
      const tools = this._serverNameToTools[serverName];
      if (tools) {
        allTools.push(...tools);
      }
    }
    return allTools;
  }
}
