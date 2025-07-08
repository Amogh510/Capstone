const path = require('path');
const { walk } = require('./fileWalker');
const { parseFile } = require('./astParser');
const { outputDir } = require('./config');
const { writeJsonFile, normalizePath } = require('./utils/fileUtils');

const { extractComponents } = require('./extractors/componentExtractor');
const { extractStates } = require('./extractors/stateExtractor');
const { extractProps } = require('./extractors/propExtractor');
const { extractHooks } = require('./extractors/hookExtractor');
const { extractEventHandlers } = require('./extractors/eventHandlerExtractor');
const { extractJsxElements } = require('./extractors/jsxElementExtractor');
const { extractContexts } = require('./extractors/contextExtractor');
const { extractRoutes } = require('./extractors/routeExtractor');

const { buildFdg } = require('./fdgBuilder');
const { buildKg } = require('./kgNodeBuilder');

async function main() {
  console.log('Starting analysis...');
  const files = walk();
  const kgNodesByFile = new Map();

  for (const filePath of files) {
    const normalizedPath = normalizePath(filePath);
    const ast = parseFile(filePath);
    if (!ast) continue;

    const fileNodes = [];

    const components = extractComponents({ ast, filePath: normalizedPath });
    console.log(`File: ${normalizedPath} - Components found: ${components.length}`);
    if (components.length > 0) {
      components.forEach(comp => console.log(`  - ${comp.name} (${comp.exportType})`));
    }
    fileNodes.push(...components);

    // Extract all related data for each component
    components.forEach((component) => {
      const componentName = component.name;
      
      // Extract states, props, hooks, event handlers, and JSX elements
      const states = extractStates({ ast, componentName });
      const props = extractProps({ ast, componentName, filePath: normalizedPath });
      const hooks = extractHooks({ ast, componentName });
      const eventHandlers = extractEventHandlers({ ast, componentName });
      const jsxElements = extractJsxElements({ ast, componentName });
      
      // Add all extracted data to file nodes
      fileNodes.push(...states, ...props, ...hooks, ...eventHandlers, ...jsxElements);
      
      // Update the component node with its related data using IDs
      component.props = props.map(prop => prop.id);
      component.hooksUsed = hooks.map(hook => hook.id);
      // Note: contextUsed will be updated later when we extract contexts
    });

    // Extract contexts and routes
    const contexts = extractContexts({ ast, filePath: normalizedPath });
    const { routes, declareRoutes } = extractRoutes({ ast, filePath: normalizedPath });
    fileNodes.push(...contexts, ...routes);

    // Update component nodes with context information using IDs
    components.forEach((component) => {
      const componentName = component.name;
      const componentContexts = contexts.filter(context => 
        context.componentName === componentName
      );
      component.contextUsed = componentContexts.map(context => context.id);
    });

    kgNodesByFile.set(normalizedPath, fileNodes);
  }

  const fdg = buildFdg(files, kgNodesByFile);
  let kg = buildKg(kgNodesByFile);

  // Deduplicate props, hooksUsed, and contextUsed for each component node
  kg.nodes.forEach(node => {
    if (node.type === 'Component') {
      if (Array.isArray(node.props)) {
        node.props = Array.from(new Set(node.props));
      }
      if (Array.isArray(node.hooksUsed)) {
        node.hooksUsed = Array.from(new Set(node.hooksUsed));
      }
      if (Array.isArray(node.contextUsed)) {
        node.contextUsed = Array.from(new Set(node.contextUsed));
      }
    }
  });

  // Update declareRoutes in FDG
  fdg.nodes.forEach((node) => {
    const fileKgNodes = kgNodesByFile.get(node.fileId);
    if (fileKgNodes) {
      const hasRoutes = fileKgNodes.some((kgNode) => kgNode.type === 'Route');
      if (hasRoutes) {
        node.declareRoutes = true;
      }
    }
  });

  writeJsonFile(path.join(outputDir, 'fdg.json'), fdg);
  writeJsonFile(path.join(outputDir, 'kg.json'), kg);

  console.log('Analysis complete.');
}

main().catch((error) => console.error('Unhandled error:', error)); 