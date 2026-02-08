// K'UHUL MICRONAUT FACTORY - Web/App/Game Development Edition
// Enhanced Version with Full ASX Integration

const KUHUL = {
  // ==================== AGENT SPAWNER ====================
  spawn(type, spec = {}) {
    const id = `${type[0]}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    const caps = this.capabilities(type);
    const agent = {
      $id: id,
      $type: type,
      $caps: caps,
      $state: "live",
      $ports: this.createPorts(id),
      $spec: spec,
      $created: Date.now(),
      $parent: spec.parent || "root"
    };

    // Initialize with ASX shell structure
    this.initAsxShell(agent, spec);

    // Register in hive
    this.hive.set(id, agent);

    // Execute spawn protocol
    this.execute(id, "⟁Pop⟁spawn⟁Wo⟁init⟁Xul", { agent, spec });

    return { id, agent, status: "activated" };
  },

  // ==================== CAPABILITY MATRIX ====================
  capabilities(type) {
    const matrix = {
      // Web Development Suite
      w: {
        html: ["semantic", "seo", "web-components", "templates"],
        css: ["grid", "flex", "animations", "variables", "themes"],
        js: ["es6+", "react", "vue", "svelte", "typescript"],
        responsive: ["mobile-first", "breakpoints", "fluid", "adaptive"],
        ux: ["a11y", "performance", "pwa", "offline", "installable"]
      },

      // App Development Suite
      a: {
        mobile: ["ios", "android", "react-native", "flutter", "capacitor"],
        desktop: ["electron", "tauri", "webview", "native-bindings"],
        pwa: ["service-workers", "manifest", "notifications", "background-sync"],
        native: ["camera", "geolocation", "biometrics", "storage", "sensors"]
      },

      // Game Development Suite
      g: {
        "2d": ["canvas", "pixi.js", "phaser", "physics", "sprites"],
        "3d": ["three.js", "babylon", "webgl", "shaders", "models"],
        physics: ["matter.js", "box2d", "collision", "forces", "joints"],
        ai: ["pathfinding", "behavior-trees", "state-machines", "neural-nets"],
        multiplayer: ["websockets", "webrtc", "matchmaking", "synchronization"]
      },

      // Backend/Data Suite
      d: {
        api: ["rest", "graphql", "websockets", "rpc", "openapi"],
        database: ["sql", "mongodb", "redis", "indexeddb", "realtime"],
        auth: ["oauth2", "jwt", "sessions", "mfa", "permissions"],
        security: ["encryption", "sanitization", "rate-limiting", "cors"],
        deployment: ["docker", "kubernetes", "serverless", "cdn", "ci/cd"]
      },

      // Studio/Tape Suite
      s: {
        studio: ["projects", "workflow", "collaboration", "versioning"],
        tape: ["asx-v1", "components", "state", "routing", "lifecycle"],
        cartridge: ["standalone", "portable", "executable", "distributable"],
        git: ["import", "export", "sync", "merge", "conflict-resolution"],
        converter: ["html→asx", "react→asx", "vue→asx", "legacy→modern"]
      },

      // Training/ML Suite
      tr: {
        train: ["datasets", "models", "hyperparameters", "validation"],
        tune: ["optimization", "pruning", "quantization", "distillation"],
        eval: ["metrics", "benchmarks", "a/b-testing", "analytics"]
      }
    };

    const prefix = type[0];
    const subtype = type.slice(1) || "base";
    return matrix[prefix]?.[subtype] || matrix[prefix] || ["analyze", "process", "generate"];
  },

  // ==================== ASX SHELL INTEGRATION ====================
  initAsxShell(agent, spec) {
    agent.$shell = {
      $schema: "asx-shell-v1",
      $metadata: {
        name: spec.name || `${agent.$type} Application`,
        type: agent.$type,
        studio: spec.studio || "kuhul-studio",
        version: "1.0.0",
        created: new Date().toISOString()
      },
      $state: this.createInitialState(agent.$type, spec),
      $routes: this.createRoutes(agent.$type, spec),
      $components: this.createComponents(agent.$type, spec),
      $actions: this.createActions(agent.$type, spec),
      $lifecycle: {
        mounted: [],
        updated: [],
        destroyed: []
      }
    };

    // Inject ASX tape if provided
    if (spec.tape) {
      agent.$shell = this.mergeAsxTapes(agent.$shell, spec.tape);
    }
  },

  createInitialState(type, spec) {
    const base = {
      loading: false,
      error: null,
      timestamp: Date.now()
    };

    switch (type[0]) {
      case "w": // Web
        return {
          ...base,
          page: "home",
          user: null,
          theme: "light",
          language: "en"
        };
      case "a": // App
        return {
          ...base,
          platform: spec.platform || "web",
          online: true,
          permissions: {},
          storage: {}
        };
      case "g": // Game
        return {
          ...base,
          scene: "menu",
          score: 0,
          level: 1,
          player: { x: 0, y: 0, health: 100 },
          entities: []
        };
      case "d": // Data
        return {
          ...base,
          connected: false,
          queries: 0,
          cache: {},
          endpoints: []
        };
      default:
        return base;
    }
  },

  // ==================== COMMUNICATION PORTS ====================
  createPorts(id) {
    const base = `object://route/kuhul/a/${id}`;
    return {
      q: `${base}/q`, // Query port
      i: `${base}/i`, // Instruction port
      tr: `${base}/tr`, // Training port
      st: `${base}/st`, // State port
      ev: `${base}/ev` // Event port
    };
  },

  // ==================== EXECUTION ENGINE ====================
  hive: new Map(),
  processes: new Map(),

  async execute(agentId, protocol, data) {
    const procId = `proc_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    const process = {
      id: procId,
      agent: agentId,
      protocol,
      data,
      status: "running",
      start: Date.now(),
      result: null
    };

    this.processes.set(procId, process);

    // Parse protocol: ⟁Pop⟁action⟁Wo⟁target⟁Xul
    const parts = protocol.split("⟁").filter((p) => p);
    if (parts.length >= 5) {
      const [, pop, action, wo, target, xul] = parts;

      switch (action) {
        case "spawn":
          process.result = await this.handleSpawn(target, data);
          break;
        case "train":
          process.result = await this.handleTraining(target, data);
          break;
        case "convert":
          process.result = await this.handleConversion(target, data);
          break;
        case "build":
          process.result = await this.handleBuild(target, data);
          break;
        default:
          process.result = { action, target, executed: true };
      }
    }

    process.status = "completed";
    process.end = Date.now();
    process.duration = process.end - process.start;

    return process;
  },

  // ==================== SPECIALIZED HANDLERS ====================
  async handleSpawn(target, data) {
    // Spawn sub-agents based on project type
    const project = data.spec || {};

    if (project.type === "ecommerce") {
      const agents = await Promise.all([
        this.spawn("whtml", { parent: data.agent.$id, role: "storefront" }),
        this.spawn("dcart", { parent: data.agent.$id, role: "shopping-cart" }),
        this.spawn("dpayments", { parent: data.agent.$id, role: "checkout" }),
        this.spawn("winventory", { parent: data.agent.$id, role: "product-management" })
      ]);

      return { type: "ecommerce", agents: agents.map((a) => a.id), status: "deployed" };
    }

    if (project.type === "game") {
      const agents = await Promise.all([
        this.spawn("g2d", { parent: data.agent.$id, genre: project.genre }),
        this.spawn("gphysics", { parent: data.agent.$id }),
        this.spawn("gai", { parent: data.agent.$id }),
        this.spawn("gmultiplayer", { parent: data.agent.$id })
      ]);

      return { type: "game", agents: agents.map((a) => a.id), genre: project.genre };
    }

    return { spawned: true, target, details: data };
  },

  async handleTraining(target, data) {
    // ML training for specific domains
    const model = {
      type: data.conf?.type || "neural",
      layers: [],
      dataset: data.ds || [],
      metrics: {}
    };

    // Simulated training
    for (let i = 0; i < 10; i += 1) {
      model.metrics[`epoch_${i}`] = {
        loss: Math.random() * 0.1,
        accuracy: 0.9 + Math.random() * 0.09
      };
    }

    return { trained: true, model, duration: "simulated" };
  },

  async handleConversion(target, data) {
    // Convert from various formats to ASX
    const source = data.source || {};
    let converted;

    if (source.type === "registry") {
      // Import from registry
      converted = {
        $schema: "asx-shell-v1",
        $metadata: {
          name: source.repo,
          imported: new Date().toISOString(),
          source: "registry",
          path: source.path
        },
        $components: this.convertRegistryToComponents(source),
        $state: { imported: true, synced: false }
      };
    } else if (source.type === "react") {
      // Convert React to ASX
      converted = this.convertReactToAsx(source);
    } else if (source.type === "vue") {
      // Convert Vue to ASX
      converted = this.convertVueToAsx(source);
    }

    return { converted: true, format: "asx", result: converted };
  },

  async handleBuild(target, data) {
    // Build ASX cartridge (self-contained app)
    const cartridge = {
      id: `cart_${Date.now()}`,
      format: "asx-cartridge-v1",
      contains: data.agents || [],
      size: "standalone",
      executable: true,
      manifest: {
        name: data.name || "K'UHUL Application",
        version: "1.0.0",
        requires: ["asx-runtime"],
        entrypoint: data.entry || "main.asx"
      }
    };

    return { built: true, cartridge };
  },

  // ==================== STUDIO MANAGEMENT ====================
  createStudio(name) {
    const studioId = `studio_${name.replace(/\s+/g, "-").toLowerCase()}_${Date.now()}`;
    const studio = {
      $id: studioId,
      $type: "studio",
      name,
      projects: new Map(),
      agents: new Set(),
      tapes: [],
      registry: {
        connected: false,
        repos: []
      },
      created: Date.now(),
      $state: "active"
    };

    // Spawn studio management agents
    const studioAgent = this.spawn("sstudio", { parent: "root", studio: studioId });
    const gitAgent = this.spawn("sgit", { parent: studioAgent.id });

    studio.agents.add(studioAgent.id);
    studio.agents.add(gitAgent.id);

    this.hive.set(studioId, studio);

    return { studioId, studio, agents: [studioAgent, gitAgent] };
  },

  // ==================== TAPE GENERATION ====================
  generateTape(project, format = "asx") {
    const tapeId = `tape_${project.name}_${Date.now()}`;

    const tape = {
      $id: tapeId,
      $schema: "asx-tape-v1",
      project: project.name,
      format,
      generated: new Date().toISOString(),
      contents: {
        shell: project.shell || {},
        components: project.components || {},
        state: project.state || {},
        routes: project.routes || {},
        actions: project.actions || {}
      },
      size: this.calculateTapeSize(project),
      hash: this.generateHash(project)
    };

    return tape;
  },

  // ==================== REGISTRY INTEGRATION ====================
  async importFromRegistry(path) {
    // Simulated registry import
    const repoName = path.split("/").pop();

    const project = {
      name: repoName,
      source: "registry",
      path,
      imported: new Date().toISOString(),
      structure: this.analyzeRepoStructure(path),
      files: [] // Would be populated with actual file contents
    };

    // Convert to ASX
    const converted = await this.handleConversion("registry", { source: project });

    // Create studio for imported project
    const studio = this.createStudio(repoName);
    const tape = this.generateTape({ ...project, ...converted.result });

    studio.projects.set(project.name, {
      project,
      tape,
      status: "imported"
    });

    return { project, studio: studio.studioId, tape: tape.$id, status: "imported" };
  },

  // ==================== UTILITIES ====================
  analyzeRepoStructure(path) {
    // Simulated structure analysis
    return {
      type: this.detectProjectType(path),
      frameworks: ["react", "node", "express"], // Would detect actual
      buildSystem: "webpack",
      tests: "jest",
      languages: ["javascript", "html", "css"]
    };
  },

  detectProjectType(path) {
    if (path.includes("react")) return "react-app";
    if (path.includes("vue")) return "vue-app";
    if (path.includes("game")) return "game";
    if (path.includes("ecommerce")) return "ecommerce";
    return "web-app";
  },

  calculateTapeSize(project) {
    const json = JSON.stringify(project);
    const bytes = new TextEncoder().encode(json).length;
    return `${(bytes / 1024).toFixed(2)}KB`;
  },

  generateHash(content) {
    // Simple hash for demo
    const str = JSON.stringify(content);
    let hash = 0;
    for (let i = 0; i < str.length; i += 1) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash |= 0;
    }
    return hash.toString(16);
  },

  convertRegistryToComponents(source) {
    // Simulated conversion
    return {
      App: {
        type: "component",
        template: "<div>Imported from registry</div>",
        style: "/* Registry styles */",
        script: "// Registry logic"
      }
    };
  },

  convertReactToAsx(source) {
    return {
      components: {
        [source.name || "ReactComponent"]: {
          type: "asx-component",
          convertedFrom: "react",
          props: source.props || {},
          state: source.state || {},
          lifecycle: this.mapReactLifecycle(source)
        }
      }
    };
  },

  convertVueToAsx(source) {
    return {
      components: {
        [source.name || "VueComponent"]: {
          type: "asx-component",
          convertedFrom: "vue",
          data: source.data || {},
          computed: source.computed || {},
          methods: source.methods || {}
        }
      }
    };
  },

  mapReactLifecycle(component) {
    return {
      mounted: component.componentDidMount ? "converted" : null,
      updated: component.componentDidUpdate ? "converted" : null,
      destroyed: component.componentWillUnmount ? "converted" : null
    };
  },

  createRoutes(type, spec) {
    const base = {
      home: { path: "/", component: "Home" },
      about: { path: "/about", component: "About" }
    };

    if (type[0] === "w" && spec.type === "ecommerce") {
      return {
        ...base,
        products: { path: "/products", component: "ProductList" },
        product: { path: "/product/:id", component: "ProductDetail" },
        cart: { path: "/cart", component: "Cart" },
        checkout: { path: "/checkout", component: "Checkout" }
      };
    }

    if (type[0] === "g") {
      return {
        menu: { path: "/", component: "MainMenu" },
        game: { path: "/play", component: "GameScene" },
        leaderboard: { path: "/scores", component: "Leaderboard" },
        settings: { path: "/settings", component: "Settings" }
      };
    }

    return base;
  },

  createComponents(type, spec) {
    const base = {
      Home: { type: "page", template: "<h1>Welcome</h1>" },
      About: { type: "page", template: "<h1>About</h1>" }
    };

    if (type[0] === "w" && spec.type === "ecommerce") {
      return {
        ...base,
        ProductList: { type: "grid", items: "products" },
        ProductDetail: { type: "detail", fields: ["name", "price", "description"] },
        Cart: { type: "list", editable: true },
        Checkout: { type: "form", fields: ["shipping", "payment"] }
      };
    }

    return base;
  },

  createActions(type, spec) {
    return {
      navigate: { type: "router", effect: "route-change" },
      updateState: { type: "state", effect: "state-update" },
      fetchData: { type: "api", effect: "data-load" }
    };
  },

  mergeAsxTapes(base, extension) {
    // Deep merge for ASX tapes
    const merge = (target, source) => {
      for (const key in source) {
        if (source[key] && typeof source[key] === "object" && !Array.isArray(source[key])) {
          if (!target[key]) target[key] = {};
          merge(target[key], source[key]);
        } else {
          target[key] = source[key];
        }
      }
      return target;
    };

    return merge(JSON.parse(JSON.stringify(base)), extension);
  }
};

// ==================== API EXPOSURE ====================
if (typeof self !== "undefined") {
  // Worker/Service Worker context
  self.KUHUL = KUHUL;

  self.onfetch = async (event) => {
    const url = new URL(event.request.url);

    // API endpoints
    if (url.pathname.startsWith("/kuhul/api/")) {
      const endpoint = url.pathname.slice("/kuhul/api/".length);
      const params = await event.request.json().catch(() => ({}));

      let response;

      switch (endpoint) {
        case "spawn":
          response = KUHUL.spawn(params.type, params.spec);
          break;
        case "studio/create":
          response = KUHUL.createStudio(params.name);
          break;
        case "registry/import":
          response = await KUHUL.importFromRegistry(params.path);
          break;
        case "tape/generate":
          response = KUHUL.generateTape(params.project, params.format);
          break;
        case "execute":
          response = await KUHUL.execute(params.agentId, params.protocol, params.data);
          break;
        default:
          response = { error: "Unknown endpoint", endpoint };
      }

      return new Response(JSON.stringify(response), {
        headers: { "Content-Type": "application/json" }
      });
    }

    return new Response(null, { status: 404 });
  };

  self.onmessage = (event) => {
    const { type, data } = event.data;

    switch (type) {
      case "SPAWN": {
        const result = KUHUL.spawn(data.type, data.spec);
        event.ports[0]?.postMessage(result);
        break;
      }
      case "EXECUTE":
        KUHUL.execute(data.agentId, data.protocol, data.data).then((result) =>
          event.ports[0]?.postMessage(result)
        );
        break;
      default:
        break;
    }
  };
}

// ==================== EXPORT FOR NODE/BROWSER ====================
if (typeof module !== "undefined" && module.exports) {
  module.exports = KUHUL;
} else if (typeof window !== "undefined") {
  window.KUHUL = KUHUL;
}

console.log("🔷 K'UHUL MICRONAUT FACTORY v1.2 - Web/App/Game Edition");
console.log("📦 Capabilities: Web Dev | App Dev | Game Dev | Backend | Studio");
console.log("🔄 Registry Import | ASX Tape Generation | Cartridge Building");
console.log("🚀 Ready for project: ecommerce, games, apps, and more!");
