/**
 * KUHUL 3D Inference Cluster — Three.js Visualization
 *
 * Renders the 1000-node inference cluster as a 3D spatial lattice:
 *   - Nodes as cubes (colored by type: compute/routing/storage/verification/control)
 *   - Folds as translucent planes slicing through the grid
 *   - Token flow as particles moving between active nodes
 *   - MoE pipeline stages highlighted in sequence
 *
 * Integrates with InferenceCluster (inference_cluster.js) for live data.
 * Can run in browser (index.html) or headless (SVG export for VM-1 projection).
 *
 * Dependencies: three.js (CDN or npm)
 */

// ---------------------------------------------------------------------------
// Configuration (mirrors models.toml visualization section)
// ---------------------------------------------------------------------------

const VIZ_CONFIG = {
  scene: {
    width: 1200,
    height: 800,
    background: 0x0a0a0f,
    cameraFov: 60,
    cameraDistance: 25,
  },
  nodes: {
    compute:      { color: 0x00ff88, size: 0.15 },
    routing:      { color: 0x4488ff, size: 0.15 },
    storage:      { color: 0xff8844, size: 0.15 },
    verification: { color: 0xff44ff, size: 0.15 },
    control:      { color: 0xffff00, size: 0.15 },
  },
  particles: {
    size: 0.05,
    speed: 2.0,
    color: 0xffffff,
    trailLength: 8,
  },
  foldPlanes: {
    enabled: true,
    opacity: 0.08,
    gridVisible: true,
  },
  glow: {
    activeIntensity: 2.0,
    idleIntensity: 0.3,
    pulseSpeed: 0.02,
  },
};

// Micronaut labels for the pipeline HUD
const PIPELINE_STAGES = [
  { id: 'PM-1', label: 'Perceive',  fold: '⟁DATA_FOLD⟁',    color: 0x4488ff },
  { id: 'CM-1', label: 'Gate',      fold: '⟁CONTROL_FOLD⟁',  color: 0xffff00 },
  { id: 'TM-1', label: 'Schedule',  fold: '⟁TIME_FOLD⟁',     color: 0x4488ff },
  { id: 'HM-1', label: 'Normalize', fold: '⟁STATE_FOLD⟁',    color: 0xff8844 },
  { id: 'MM-1', label: 'Infer',     fold: '⟁COMPUTE_FOLD⟁',  color: 0x00ff88 },
  { id: 'XM-1', label: 'Expand',    fold: '⟁PATTERN_FOLD⟁',  color: 0x00ff88 },
  { id: 'SM-1', label: 'Seal',      fold: '⟁STORAGE_FOLD⟁',  color: 0xff8844 },
  { id: 'VM-2', label: 'Verify',    fold: '⟁META_FOLD⟁',     color: 0xff44ff },
  { id: 'VM-1', label: 'Render',    fold: '⟁UI_FOLD⟁',       color: 0x4488ff },
];

// ---------------------------------------------------------------------------
// Kuhul3DCluster class
// ---------------------------------------------------------------------------

export class Kuhul3DCluster {
  /**
   * @param {HTMLElement|null} container — DOM element for the canvas (null for headless)
   * @param {InferenceCluster|null} cluster — reference to the inference cluster for live data
   */
  constructor(container = null, cluster = null) {
    this.container = container;
    this.cluster = cluster;
    this.config = VIZ_CONFIG;
    this.gridSize = cluster ? cluster.gridSize : 10;

    // Three.js objects (initialized in init())
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    // Scene graph
    this.nodeMeshes = new Map();      // nodeId → THREE.Mesh
    this.edgeLines = [];              // active connection lines
    this.particles = [];              // token flow particles
    this.foldPlanes = new Map();      // foldName → THREE.Mesh
    this.pipelineHighlight = null;    // current MoE stage highlight

    // Animation state
    this.clock = null;
    this.animating = false;
    this.currentStage = -1;           // pipeline stage index for animation
    this.frameCount = 0;
  }

  /**
   * Initialize the Three.js scene.
   * Call this after the DOM is ready (or skip for headless SVG export).
   */
  async init() {
    // Dynamic import — works in browser or Node.js with three installed
    let THREE;
    try {
      THREE = await import('three');
    } catch {
      console.warn('[Kuhul3D] three.js not available — using headless mode');
      this._headless = true;
      return this._initHeadless();
    }

    this.THREE = THREE;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.config.scene.background);

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      this.config.scene.cameraFov,
      this.config.scene.width / this.config.scene.height,
      0.1,
      1000
    );
    this.camera.position.set(
      this.config.scene.cameraDistance,
      this.config.scene.cameraDistance * 0.8,
      this.config.scene.cameraDistance
    );
    this.camera.lookAt(this.gridSize / 2, this.gridSize / 2, this.gridSize / 2);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.config.scene.width, this.config.scene.height);
    this.renderer.setPixelRatio(typeof window !== 'undefined' ? window.devicePixelRatio : 1);

    if (this.container) {
      this.container.appendChild(this.renderer.domElement);
    }

    // Orbit controls (if available)
    try {
      const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js');
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.target.set(this.gridSize / 2, this.gridSize / 2, this.gridSize / 2);
    } catch {
      // OrbitControls not available — camera is static
    }

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404050, 0.5);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(20, 30, 20);
    this.scene.add(directionalLight);

    // Clock
    this.clock = new THREE.Clock();

    // Build the visual elements
    this._buildNodes();
    if (this.config.foldPlanes.enabled) {
      this._buildFoldPlanes();
    }

    console.log('[Kuhul3D] Scene initialized');
    return this;
  }

  /**
   * Headless init — generates data for SVG export without Three.js.
   */
  _initHeadless() {
    this._headless = true;
    console.log('[Kuhul3D] Headless mode — SVG export available via exportSVG()');
    return this;
  }

  // -------------------------------------------------------------------------
  // Scene building
  // -------------------------------------------------------------------------

  /**
   * Build node meshes from the inference cluster data.
   */
  _buildNodes() {
    const THREE = this.THREE;
    const geometry = new THREE.BoxGeometry(
      this.config.nodes.compute.size,
      this.config.nodes.compute.size,
      this.config.nodes.compute.size
    );

    const nodesData = this.cluster ? this.cluster.getNodesArray() : this._generateDefaultNodes();

    for (const nodeData of nodesData) {
      const typeConfig = this.config.nodes[nodeData.type] || this.config.nodes.compute;
      const material = new THREE.MeshPhongMaterial({
        color: typeConfig.color,
        transparent: true,
        opacity: nodeData.active ? 1.0 : this.config.glow.idleIntensity,
        emissive: typeConfig.color,
        emissiveIntensity: nodeData.active ? this.config.glow.activeIntensity : 0.1,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(nodeData.position.x, nodeData.position.y, nodeData.position.z);
      mesh.userData = nodeData;

      this.scene.add(mesh);
      this.nodeMeshes.set(nodeData.id, mesh);
    }
  }

  /**
   * Build translucent fold planes that slice through the grid.
   */
  _buildFoldPlanes() {
    const THREE = this.THREE;
    const planeSize = this.gridSize + 2;

    // Create planes at different Y-levels for the sovereignty folds
    const foldLevels = [
      { fold: '⟁CONTROL_FOLD⟁',  y: 0,  color: 0xffff00 },
      { fold: '⟁COMPUTE_FOLD⟁',  y: 3,  color: 0x00ff88 },
      { fold: '⟁STORAGE_FOLD⟁',  y: 5,  color: 0xff8844 },
      { fold: '⟁META_FOLD⟁',     y: 7,  color: 0xff44ff },
      { fold: '⟁UI_FOLD⟁',       y: 9,  color: 0x4488ff },
    ];

    for (const { fold, y, color } of foldLevels) {
      const planeGeometry = new THREE.PlaneGeometry(planeSize, planeSize);
      const planeMaterial = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: this.config.foldPlanes.opacity,
        side: THREE.DoubleSide,
      });

      const plane = new THREE.Mesh(planeGeometry, planeMaterial);
      plane.rotation.x = -Math.PI / 2;
      plane.position.set(this.gridSize / 2, y, this.gridSize / 2);
      plane.userData = { fold };

      this.scene.add(plane);
      this.foldPlanes.set(fold, plane);
    }
  }

  /**
   * Generate default node positions when no cluster is provided.
   */
  _generateDefaultNodes() {
    const nodes = [];
    const types = ['compute', 'routing', 'storage', 'verification', 'control'];
    let idx = 0;

    for (let x = 0; x < this.gridSize; x++) {
      for (let y = 0; y < this.gridSize; y++) {
        for (let z = 0; z < this.gridSize; z++) {
          nodes.push({
            id: `node_${x}_${y}_${z}`,
            type: types[idx % types.length],
            position: { x, y, z },
            active: false,
            load: 0,
          });
          idx++;
        }
      }
    }
    return nodes;
  }

  // -------------------------------------------------------------------------
  // Update & Animation
  // -------------------------------------------------------------------------

  /**
   * Update node visuals from live cluster state.
   */
  updateFromCluster() {
    if (!this.cluster || this._headless) return;

    const nodesData = this.cluster.getNodesArray();

    for (const nodeData of nodesData) {
      const mesh = this.nodeMeshes.get(nodeData.id);
      if (!mesh) continue;

      const typeConfig = this.config.nodes[nodeData.type] || this.config.nodes.compute;

      // Update opacity and emissive based on active state
      mesh.material.opacity = nodeData.active ? 1.0 : this.config.glow.idleIntensity;
      mesh.material.emissiveIntensity = nodeData.active
        ? this.config.glow.activeIntensity * (0.5 + nodeData.load * 0.5)
        : 0.1;

      // Scale active nodes slightly
      const scale = nodeData.active ? 1.0 + nodeData.load * 0.5 : 1.0;
      mesh.scale.set(scale, scale, scale);
    }

    // Update edge lines
    this._updateEdges();
  }

  /**
   * Update active edge connections.
   */
  _updateEdges() {
    const THREE = this.THREE;

    // Remove old edges
    for (const line of this.edgeLines) {
      this.scene.remove(line);
    }
    this.edgeLines = [];

    if (!this.cluster) return;

    const edges = this.cluster.getActiveEdges();
    const material = new THREE.LineBasicMaterial({
      color: 0x224466,
      transparent: true,
      opacity: 0.3,
    });

    for (const edge of edges) {
      const points = [
        new THREE.Vector3(edge.fromPos.x, edge.fromPos.y, edge.fromPos.z),
        new THREE.Vector3(edge.toPos.x, edge.toPos.y, edge.toPos.z),
      ];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geometry, material);
      this.scene.add(line);
      this.edgeLines.push(line);
    }
  }

  /**
   * Add a token particle that travels between two nodes.
   */
  addTokenParticle(fromNodeId, toNodeId) {
    if (this._headless) return;

    const THREE = this.THREE;
    const fromMesh = this.nodeMeshes.get(fromNodeId);
    const toMesh = this.nodeMeshes.get(toNodeId);
    if (!fromMesh || !toMesh) return;

    const geometry = new THREE.SphereGeometry(this.config.particles.size, 8, 8);
    const material = new THREE.MeshBasicMaterial({
      color: this.config.particles.color,
      transparent: true,
      opacity: 0.9,
    });

    const particle = new THREE.Mesh(geometry, material);
    particle.position.copy(fromMesh.position);
    particle.userData = {
      from: fromMesh.position.clone(),
      to: toMesh.position.clone(),
      progress: 0,
      speed: this.config.particles.speed,
    };

    this.scene.add(particle);
    this.particles.push(particle);
  }

  /**
   * Highlight a pipeline stage (glow the fold's nodes).
   */
  highlightStage(stageIndex) {
    if (stageIndex < 0 || stageIndex >= PIPELINE_STAGES.length) return;

    const stage = PIPELINE_STAGES[stageIndex];
    this.currentStage = stageIndex;

    // Dim all nodes first
    for (const mesh of this.nodeMeshes.values()) {
      mesh.material.emissiveIntensity = 0.05;
      mesh.material.opacity = 0.2;
    }

    // Highlight the active fold's nodes
    if (this.cluster) {
      const foldNodes = this.cluster.foldIndex.get(stage.fold) || [];
      for (const nodeId of foldNodes) {
        const mesh = this.nodeMeshes.get(nodeId);
        if (mesh) {
          mesh.material.emissiveIntensity = this.config.glow.activeIntensity;
          mesh.material.opacity = 1.0;
        }
      }
    }

    // Highlight the fold plane
    for (const [fold, plane] of this.foldPlanes) {
      plane.material.opacity = fold === stage.fold
        ? this.config.foldPlanes.opacity * 3
        : this.config.foldPlanes.opacity;
    }
  }

  /**
   * Animate one frame.
   */
  animate() {
    if (this._headless) return;

    this.frameCount++;
    const delta = this.clock ? this.clock.getDelta() : 0.016;

    // Update particle positions
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.userData.progress += p.userData.speed * delta;

      if (p.userData.progress >= 1.0) {
        this.scene.remove(p);
        this.particles.splice(i, 1);
      } else {
        p.position.lerpVectors(p.userData.from, p.userData.to, p.userData.progress);
        p.material.opacity = 1.0 - p.userData.progress * 0.5;
      }
    }

    // Pulse active nodes
    for (const mesh of this.nodeMeshes.values()) {
      if (mesh.userData.active) {
        const pulse = Math.sin(this.frameCount * this.config.glow.pulseSpeed) * 0.3 + 0.7;
        mesh.material.emissiveIntensity *= pulse;
      }
    }

    // Update controls
    if (this.controls) {
      this.controls.update();
    }

    // Render
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  /**
   * Start the animation loop.
   */
  startAnimation() {
    if (this._headless) return;
    this.animating = true;

    const loop = () => {
      if (!this.animating) return;
      this.animate();
      requestAnimationFrame(loop);
    };
    loop();
  }

  /**
   * Stop the animation loop.
   */
  stopAnimation() {
    this.animating = false;
  }

  // -------------------------------------------------------------------------
  // SVG Export (VM-1 projection — headless compatible)
  // -------------------------------------------------------------------------

  /**
   * Export the current cluster state as an SVG spatial lattice.
   * This is the VM-1 rendering_projection tool output.
   * Works in headless mode without Three.js.
   */
  exportSVG() {
    const nodesData = this.cluster ? this.cluster.getNodesArray() : this._generateDefaultNodes();
    const svgWidth = this.config.scene.width;
    const svgHeight = this.config.scene.height;
    const scale = 40;
    const offsetX = svgWidth / 2 - (this.gridSize * scale) / 2;
    const offsetY = svgHeight / 2 - (this.gridSize * scale) / 2;

    // Isometric projection
    const project = (x, y, z) => ({
      px: offsetX + (x - z) * scale * 0.866,
      py: offsetY + (x + z) * scale * 0.5 - y * scale * 0.8,
    });

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" `;
    svg += `viewBox="0 0 ${svgWidth} ${svgHeight}" `;
    svg += `width="${svgWidth}" height="${svgHeight}" `;
    svg += `data-fold="⟁UI_FOLD⟁" data-micronaut="VM-1" data-projection="true">\n`;
    svg += `  <rect width="100%" height="100%" fill="${colorToHex(this.config.scene.background)}"/>\n`;
    svg += `  <g id="cluster-nodes">\n`;

    // Sort by depth for painter's algorithm
    const sorted = [...nodesData].sort((a, b) => {
      const da = a.position.x + a.position.z - a.position.y;
      const db = b.position.x + b.position.z - b.position.y;
      return da - db;
    });

    for (const node of sorted) {
      const { px, py } = project(node.position.x, node.position.y, node.position.z);
      const typeConfig = this.config.nodes[node.type] || this.config.nodes.compute;
      const colorHex = colorToHex(typeConfig.color);
      const size = typeConfig.size * scale;
      const opacity = node.active ? 1.0 : 0.3;

      svg += `    <rect x="${(px - size / 2).toFixed(1)}" y="${(py - size / 2).toFixed(1)}" `;
      svg += `width="${size.toFixed(1)}" height="${size.toFixed(1)}" `;
      svg += `fill="${colorHex}" opacity="${opacity}" `;
      svg += `data-node-id="${node.id}" data-type="${node.type}" data-fold="${node.fold || ''}" `;
      svg += `data-active="${node.active}" data-load="${node.load || 0}"/>\n`;
    }

    svg += `  </g>\n`;

    // Pipeline stage labels
    svg += `  <g id="pipeline-hud" transform="translate(20, 20)">\n`;
    for (let i = 0; i < PIPELINE_STAGES.length; i++) {
      const stage = PIPELINE_STAGES[i];
      const active = i === this.currentStage;
      const cy = 20 + i * 22;
      svg += `    <text x="0" y="${cy}" fill="${colorToHex(stage.color)}" `;
      svg += `font-family="monospace" font-size="12" `;
      svg += `opacity="${active ? 1.0 : 0.4}">`;
      svg += `${active ? '▶' : '○'} ${stage.id}: ${stage.label}</text>\n`;
    }
    svg += `  </g>\n`;

    svg += `</svg>`;
    return svg;
  }

  // -------------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------------

  dispose() {
    this.stopAnimation();
    if (this.renderer) {
      this.renderer.dispose();
    }
    this.nodeMeshes.clear();
    this.edgeLines = [];
    this.particles = [];
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function colorToHex(color) {
  if (typeof color === 'string') return color;
  return '#' + color.toString(16).padStart(6, '0');
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export async function createKuhul3D(container = null, cluster = null) {
  const viz = new Kuhul3DCluster(container, cluster);
  await viz.init();
  return viz;
}

export default Kuhul3DCluster;
